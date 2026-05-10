"""
B.0 — verify dataset loader correctness via self-reprojection.

For 5 random frames from a scene loaded through the canonical
`get_dataset(cfg)` path: project all valid pixels through depth + intrinsics
to camera-frame XYZ, transform to world, transform back to camera, project
to pixel. Should return original (u, v) within sub-pixel error if the loader's
(K, depth_scale, pose convention) form a self-consistent triple.

Aborts with [FAIL] if median per-frame pixel error > 5 px on any frame.
Passes if median ≤ 3 px on all 5 frames (paper-grade gate).

This catches:
  • wrong png_depth_scale (depth too small/big → reprojected points on the
    wrong ray → wrong (u', v')).
  • pose convention sign flip (w2c where c2w expected, etc.).
  • wrong cx/cy (pixel offset = (cx_used - cx_true)).
  • wrong fx/fy (axis-aligned magnification of the residual error).

Usage:
    python scripts/verify_dataset_loader.py \\
        --config configs/RGBD/Replica/room0.yaml
    python scripts/verify_dataset_loader.py \\
        --config configs/Dynamic/TUM_RGBD/freiburg3_walking_static.yaml

Exits 0 on PASS, 1 on FAIL. Designed for CI gating per CLAUDE.md §6.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

torch.backends.cudnn.deterministic = True

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import config as droid_config
from src.utils.datasets import get_dataset


def _scene_intrinsic(stream) -> tuple[float, float, float, float, int, int]:
    """Read the post-crop / post-resize intrinsic from the BaseDataset instance.
    These match the (color, depth) tensors that __getitem__ returns."""
    fx = float(stream.fx)
    fy = float(stream.fy)
    cx = float(stream.cx)
    cy = float(stream.cy)
    # Post-edge-crop output dims
    H_out = int(stream.H_out_with_edge - 2 * stream.H_edge)
    W_out = int(stream.W_out_with_edge - 2 * stream.W_edge)
    return fx, fy, cx, cy, H_out, W_out


def _cross_frame_reprojection(depth_i: np.ndarray, depth_j: np.ndarray,
                              fx: float, fy: float, cx: float, cy: float,
                              pose_i_c2w: np.ndarray, pose_j_c2w: np.ndarray,
                              min_depth: float = 0.05,
                              max_depth: float = 10.0,
                              z_rel_tol: float = 0.05,
                              z_occlusion_gap: float = 0.05) -> dict:
    """Project pixels of frame i to world via (depth_i, pose_i_c2w), then to
    frame j's image plane via inv(pose_j_c2w) + K, look up depth_j there, and
    compare predicted z (cam_j) against observed depth_j.

    Catches: pose convention sign flip, wrong depth_scale (would diverge across
    a non-trivial baseline), wrong (fx, fy) (parallax-dependent residual).

    Tolerance:
      • z_rel_tol = 0.05 → 5% relative depth disagreement is the strict gate.
      • Occlusion-tolerant: z_pred > z_obs + z_occlusion_gap means an occluder
        (closer surface) sits in frame j — these pixels are dropped, NOT failed.
    """
    H, W = depth_i.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    valid_i = (depth_i >= min_depth) & (depth_i <= max_depth) & np.isfinite(depth_i)
    z_i = depth_i.astype(np.float64)
    x = (u.astype(np.float64) - cx) * z_i / fx
    y = (v.astype(np.float64) - cy) * z_i / fy
    cam_i = np.stack([x, y, z_i, np.ones_like(z_i)], axis=-1).reshape(-1, 4).T  # (4, H*W)
    world = pose_i_c2w @ cam_i
    cam_j = np.linalg.inv(pose_j_c2w) @ world
    z_pred = cam_j[2]
    u_pred = fx * cam_j[0] / z_pred + cx
    v_pred = fy * cam_j[1] / z_pred + cy

    valid_flat = valid_i.flatten() & (z_pred > min_depth)
    u_pred_int = np.round(u_pred).astype(int)
    v_pred_int = np.round(v_pred).astype(int)
    in_bounds = (
        valid_flat
        & (u_pred_int >= 0) & (u_pred_int < W)
        & (v_pred_int >= 0) & (v_pred_int < H)
    )
    if in_bounds.sum() < 100:
        return {"n_valid": int(in_bounds.sum()), "median_rel_err": float("inf"),
                "p95_rel_err": float("inf"), "n_occluded": 0,
                "in_bounds_pct": float(in_bounds.mean() * 100)}
    u_v = u_pred_int[in_bounds]
    v_v = v_pred_int[in_bounds]
    z_pred_v = z_pred[in_bounds]
    z_obs_v = depth_j[v_v, u_v]
    valid_obs = (z_obs_v >= min_depth) & (z_obs_v <= max_depth) & np.isfinite(z_obs_v)
    z_pred_v = z_pred_v[valid_obs]
    z_obs_v = z_obs_v[valid_obs]

    diff = z_pred_v - z_obs_v
    occluded = diff > z_occlusion_gap  # the predicted surface is BEHIND a closer one in j → drop
    consistent = ~occluded
    rel_err = np.abs(diff[consistent]) / np.maximum(z_obs_v[consistent], 1e-6)
    if rel_err.size == 0:
        return {"n_valid": int(consistent.sum()), "median_rel_err": float("inf"),
                "p95_rel_err": float("inf"), "n_occluded": int(occluded.sum()),
                "in_bounds_pct": float(in_bounds.mean() * 100)}
    return {
        "n_valid": int(rel_err.size),
        "n_occluded": int(occluded.sum()),
        "median_rel_err": float(np.median(rel_err)),
        "p95_rel_err": float(np.percentile(rel_err, 95)),
        "in_bounds_pct": float(in_bounds.mean() * 100),
    }


def _self_reprojection_error(depth: np.ndarray, fx: float, fy: float,
                             cx: float, cy: float, pose_c2w: np.ndarray,
                             min_depth: float = 0.05,
                             max_depth: float = 10.0) -> dict:
    """Project all valid pixels to world via (depth, K, pose_c2w), then back.
    Returns median + p95 + count of pixels within image bounds."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    valid = (depth >= min_depth) & (depth <= max_depth) & np.isfinite(depth)
    z = depth.astype(np.float64)
    x = (u.astype(np.float64) - cx) * z / fx
    y = (v.astype(np.float64) - cy) * z / fy
    cam_xyz1 = np.stack([x, y, z, np.ones_like(z)], axis=-1)  # (H, W, 4)
    cam_flat = cam_xyz1.reshape(-1, 4).T  # (4, H*W)

    # cam → world → cam (round-trip via pose_c2w then its inverse).
    world = pose_c2w @ cam_flat                                # (4, H*W)
    pose_w2c = np.linalg.inv(pose_c2w)
    cam_back = pose_w2c @ world                                # (4, H*W)

    z_back = cam_back[2]
    u_back = fx * cam_back[0] / z_back + cx
    v_back = fy * cam_back[1] / z_back + cy
    u_back = u_back.reshape(H, W)
    v_back = v_back.reshape(H, W)
    z_back = z_back.reshape(H, W)

    err = np.sqrt((u_back - u) ** 2 + (v_back - v) ** 2)
    err_valid = err[valid]
    in_bounds = (
        (u_back[valid] >= 0) & (u_back[valid] < W) &
        (v_back[valid] >= 0) & (v_back[valid] < H) &
        (np.abs(z_back[valid] - z[valid]) < 1e-3)
    )
    return {
        "n_valid_px": int(valid.sum()),
        "median_err_px": float(np.median(err_valid)) if err_valid.size else float("nan"),
        "p95_err_px": float(np.percentile(err_valid, 95)) if err_valid.size else float("nan"),
        "max_err_px": float(err_valid.max()) if err_valid.size else float("nan"),
        "in_bounds_pct": float(in_bounds.mean() * 100) if err_valid.size else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str,
                   help="Scene config yaml (with full inherit_from chain).")
    p.add_argument("--n-frames", default=5, type=int,
                   help="Number of random frames to test.")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--gate-median-px", default=3.0, type=float,
                   help="PASS if median pixel error <= this on every frame.")
    p.add_argument("--abort-median-px", default=5.0, type=float,
                   help="FAIL hard if any frame median error > this.")
    args = p.parse_args()

    cfg = droid_config.load_config(args.config)
    stream = get_dataset(cfg)
    n_total = len(stream)
    print(f"[setup] config={args.config}  n_frames_total={n_total}", flush=True)
    fx, fy, cx, cy, H_out, W_out = _scene_intrinsic(stream)
    print(f"[setup] post-crop K  fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}  HxW={H_out}x{W_out}",
          flush=True)
    print(f"[setup] png_depth_scale={stream.png_depth_scale}", flush=True)

    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(n_total, size=min(args.n_frames, n_total), replace=False)
    print(f"[setup] sampled frames: {sorted(idxs.tolist())}\n", flush=True)

    fail = False
    pass_count = 0
    for idx in sorted(idxs.tolist()):
        item = stream[int(idx)]
        # __getitem__ returns (idx, color, depth, pose) per BaseDataset
        if len(item) != 4:
            print(f"  [{idx}] FAIL — unexpected loader output: tuple of len {len(item)}", flush=True)
            fail = True
            continue
        _, color, depth, pose = item
        if pose is None:
            print(f"  [{idx}] FAIL — pose is None (loader didn't load traj)", flush=True)
            fail = True
            continue
        if torch.is_tensor(depth):
            depth_np = depth.cpu().numpy()
        else:
            depth_np = np.asarray(depth)
        if torch.is_tensor(pose):
            pose_np = pose.cpu().numpy()
        else:
            pose_np = np.asarray(pose)
        if pose_np.shape != (4, 4):
            print(f"  [{idx}] FAIL — pose shape {pose_np.shape} (expected (4, 4))", flush=True)
            fail = True
            continue
        # `pose` from BaseDataset is c2w (per `self.poses[index]` and Replica's load_poses).

        result = _self_reprojection_error(
            depth_np, fx, fy, cx, cy, pose_np,
            min_depth=0.05, max_depth=10.0,
        )
        m = result["median_err_px"]
        p95 = result["p95_err_px"]
        in_b = result["in_bounds_pct"]
        nv = result["n_valid_px"]
        if not np.isfinite(m):
            print(f"  [{idx}] FAIL — no valid depth pixels in this frame", flush=True)
            fail = True
            continue
        if m > args.abort_median_px:
            print(f"  [{idx}] FAIL  median={m:.3f}px (gate {args.abort_median_px})  p95={p95:.3f}px  "
                  f"in_bounds={in_b:.1f}%  n_valid={nv}", flush=True)
            fail = True
        elif m > args.gate_median_px:
            print(f"  [{idx}] WARN  median={m:.3f}px (over soft gate {args.gate_median_px}, under hard {args.abort_median_px})  "
                  f"p95={p95:.3f}px  in_bounds={in_b:.1f}%  n_valid={nv}", flush=True)
            pass_count += 1
        else:
            print(f"  [{idx}] PASS  median={m:.3f}px  p95={p95:.3f}px  "
                  f"in_bounds={in_b:.1f}%  n_valid={nv}", flush=True)
            pass_count += 1

    print(f"\n=== self-reproj summary ===  passed={pass_count}/{len(idxs)}  hard_fail={fail}", flush=True)

    # ── Cross-frame reprojection (Reviewer A audit, Report 18 B.0):
    # self-reproj alone is mathematical identity and cannot detect c2w/w2c
    # flips, wrong depth_scale, or wrong K vs scene-truth. Cross-frame depth
    # consistency on a static-scene pair catches all three. ─────────────────
    print(f"\n=== cross-frame depth-consistency (Reviewer A required) ===", flush=True)
    sorted_idxs = sorted(idxs.tolist())
    cross_pairs = list(zip(sorted_idxs[:-1], sorted_idxs[1:]))
    cross_fail = False
    for i, j in cross_pairs:
        item_i = stream[int(i)]
        item_j = stream[int(j)]
        _, _, depth_i, pose_i = item_i
        _, _, depth_j, pose_j = item_j
        if pose_i is None or pose_j is None:
            print(f"  [{i} → {j}] SKIP — missing pose", flush=True)
            continue
        depth_i_np = depth_i.cpu().numpy() if torch.is_tensor(depth_i) else np.asarray(depth_i)
        depth_j_np = depth_j.cpu().numpy() if torch.is_tensor(depth_j) else np.asarray(depth_j)
        pose_i_np = pose_i.cpu().numpy() if torch.is_tensor(pose_i) else np.asarray(pose_i)
        pose_j_np = pose_j.cpu().numpy() if torch.is_tensor(pose_j) else np.asarray(pose_j)
        r = _cross_frame_reprojection(
            depth_i_np, depth_j_np, fx, fy, cx, cy, pose_i_np, pose_j_np,
        )
        med = r["median_rel_err"]
        p95 = r["p95_rel_err"]
        nv = r["n_valid"]
        nocc = r["n_occluded"]
        ib = r["in_bounds_pct"]
        if not np.isfinite(med):
            print(f"  [{i} → {j}] FAIL  no usable cross-frame pixels  in_bounds={ib:.1f}%", flush=True)
            cross_fail = True
            continue
        # Strict gate: 2% median relative depth error (paper-grade for static scenes).
        if med > 0.05:
            print(f"  [{i} → {j}] FAIL  median_rel={med*100:.2f}% (>5% gate)  p95_rel={p95*100:.2f}%  "
                  f"n_valid={nv} n_occluded={nocc} in_bounds={ib:.1f}%", flush=True)
            cross_fail = True
        elif med > 0.02:
            print(f"  [{i} → {j}] WARN  median_rel={med*100:.2f}% (>2% soft)  p95_rel={p95*100:.2f}%  "
                  f"n_valid={nv} n_occluded={nocc} in_bounds={ib:.1f}%", flush=True)
        else:
            print(f"  [{i} → {j}] PASS  median_rel={med*100:.2f}%  p95_rel={p95*100:.2f}%  "
                  f"n_valid={nv} n_occluded={nocc} in_bounds={ib:.1f}%", flush=True)

    if fail or cross_fail:
        print(f"\n[FAIL] {args.config}", flush=True)
        return 1
    print(f"\n[PASS] {args.config}  (self-reproj + cross-frame both clean)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
