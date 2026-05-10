"""
B.0 — verify dataset loader correctness via self- + cross-frame reprojection.

Two tests, both must PASS:

  1. SELF-REPROJECTION (sanity).
     For 5 random frames: project all valid pixels through depth + K to camera
     XYZ, transform to world via pose_c2w, back to camera via inv(pose_c2w),
     project to (u, v). Should return identity within sub-pixel error.
     Catches: arithmetic bugs, NaN/Inf in poses, wrong tensor shapes.
     Does NOT catch: wrong K, wrong depth_scale, sign-flipped pose convention
     (these all cancel under a self-consistent round-trip).

  2. CROSS-FRAME DEPTH CONSISTENCY (the actual gate, per Reviewer-2 audit).
     Fixed small-parallax pairs (0,5), (50,55), (500,505), (1000,1005),
     (1500,1505) — 5-frame baseline preserves >50% of frame i in-bounds and
     exercises near-field parallax sensitive to fx/fy. For each pair (i, j):
     project depth_i to world, then to image_j, look up depth_j at predicted
     pixel, check predicted z (cam_j) ≈ observed z. Median rel-err <= 2%
     AND p95 rel-err <= 10% required.
     Catches: pose-convention sign flips, wrong depth_scale (would diverge
     across non-trivial baselines), wrong (fx, fy) (parallax-dependent).

Reviewer-2 audit (Report 18 B.0) hardenings:
  #1  Single source-of-truth gate threshold (was 2%/5%/5% inconsistently).
  #2/#3  z>0 + isfinite guards before division (catches sign-flipped poses
         that would otherwise silently produce huge-finite garbage).
  #4  Small-parallax fixed pairs (was random-then-consecutive — produced
      [178→866] with only 0.4% of frame i in-bounds).
  #5  Loader-shape drift assertion in `_scene_intrinsic`.
  #7  p95 secondary gate (median is blind to mid-magnitude depth_scale errors
      masked by dynamic-scene tails).

NOT TESTED HERE (open issue from Reviewer 2 #8, Report 18 B.0):
  Pose-convention FALSIFICATION (OpenGL c2w vs OpenCV c2w). Both pass under
  a self-consistent round-trip + cross-frame test because both are c2w; the
  test only proves c2w (not w2c). Distinguishing OpenGL from OpenCV requires
  an external anchor — e.g., project the GT mesh into image i and compare
  RGB photometrically, or check pose[:3,1] gravity direction against scene
  prior. Plan B B.5 must audit downstream BA / RADIO consumers for handedness
  consistency before publication.

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


def _scene_intrinsic(stream, sample_depth_shape: tuple[int, int]) -> tuple[float, float, float, float, int, int]:
    """Read the post-crop / post-resize intrinsic from the BaseDataset instance.
    These match the (color, depth) tensors that __getitem__ returns.

    Reviewer-2 audit fix #5: assert the loader's reported H/W match an actual
    depth tensor's shape. Catches loader-shape drift (resize / crop config
    drift) before it silently mis-tests downstream reprojection math.
    """
    fx = float(stream.fx)
    fy = float(stream.fy)
    cx = float(stream.cx)
    cy = float(stream.cy)
    H_out = int(stream.H_out_with_edge - 2 * stream.H_edge)
    W_out = int(stream.W_out_with_edge - 2 * stream.W_edge)
    H_d, W_d = sample_depth_shape
    assert (H_out, W_out) == (H_d, W_d), (
        f"loader-shape drift: stream reports {(H_out, W_out)} but depth tensor "
        f"is {sample_depth_shape}; verifier math would silently mis-test"
    )
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
    # Reviewer-2 audit fix #3: guard z>0 + finite BEFORE division so a
    # sign-flipped pose can't silently produce huge-finite garbage.
    safe = (z_back > min_depth) & np.isfinite(z_back)
    u_back = np.full_like(z_back, np.nan)
    v_back = np.full_like(z_back, np.nan)
    u_back[safe] = fx * cam_back[0][safe] / z_back[safe] + cx
    v_back[safe] = fy * cam_back[1][safe] / z_back[safe] + cy
    u_back = u_back.reshape(H, W)
    v_back = v_back.reshape(H, W)
    z_back = z_back.reshape(H, W)

    err = np.sqrt((u_back - u) ** 2 + (v_back - v) ** 2)
    err_valid = err[valid & np.isfinite(err)]
    in_bounds = (
        (u_back[valid] >= 0) & (u_back[valid] < W) &
        (v_back[valid] >= 0) & (v_back[valid] < H) &
        (np.abs(z_back[valid] - z[valid]) < 1e-3) &
        np.isfinite(u_back[valid]) & np.isfinite(v_back[valid])
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
                   help="Number of random frames to test (self-reproj only).")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--gate-median-px", default=3.0, type=float,
                   help="PASS if median pixel error <= this on every frame.")
    p.add_argument("--abort-median-px", default=5.0, type=float,
                   help="FAIL hard if any frame median error > this.")
    p.add_argument("--gate-cross-rel", default=0.02, type=float,
                   help="Cross-frame median relative depth-error gate (Reviewer-2 #1: "
                        "default tightened to 2%% — was inconsistently 2%/5%/5% across "
                        "code/comment/print before).")
    p.add_argument("--gate-cross-p95", default=0.10, type=float,
                   help="Cross-frame p95 relative depth-error gate (Reviewer-2 #7: "
                        "median-only is blind to mid-magnitude errors).")
    p.add_argument("--gate-cross-min-overlap-pct", default=20.0, type=float,
                   help="Reviewer-2 #4: each cross-frame pair must have >=this %% of "
                        "frame-i pixels reproject in-bounds, else the pair is "
                        "dismissed as too-large-baseline (statistically meaningless).")
    args = p.parse_args()

    cfg = droid_config.load_config(args.config)
    stream = get_dataset(cfg)
    n_total = len(stream)
    print(f"[setup] config={args.config}  n_frames_total={n_total}", flush=True)
    # Probe one frame to discover depth shape; pass to intrinsic check.
    _, _, _probe_depth, _ = stream[0]
    probe_shape = (_probe_depth.shape[-2], _probe_depth.shape[-1])
    fx, fy, cx, cy, H_out, W_out = _scene_intrinsic(stream, probe_shape)
    print(f"[setup] post-crop K  fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}  HxW={H_out}x{W_out}",
          flush=True)
    print(f"[setup] png_depth_scale={stream.png_depth_scale}", flush=True)

    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(n_total, size=min(args.n_frames, n_total), replace=False)
    print(f"[setup] sampled frames (self-reproj): {sorted(idxs.tolist())}\n", flush=True)

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

    # ── Cross-frame reprojection (Reviewer-2 audit fix #4 + #7) ──────────
    # Pre-tightening: random-sample-then-consecutive-pair gave [178→866] with
    # only 2961 pixels (0.4% of frame) — statistically meaningless and biased
    # toward the far-plane subset most tolerant to pose error. Now uses fixed
    # SMALL-PARALLAX pairs (5-frame baseline) → preserves >50% of frame i in
    # bounds, exercises near-field parallax which is sensitive to fx/fy.
    SMALL_PARALLAX_OFFSETS = [(0, 5), (50, 55), (500, 505), (1000, 1005), (1500, 1505)]
    cross_pairs = [(i, j) for (i, j) in SMALL_PARALLAX_OFFSETS
                   if i < n_total and j < n_total]
    print(f"\n=== cross-frame depth-consistency (small-parallax fixed pairs) ===", flush=True)
    print(f"  pairs={cross_pairs}  median-gate={args.gate_cross_rel*100:.1f}%  "
          f"p95-gate={args.gate_cross_p95*100:.1f}%  min-overlap={args.gate_cross_min_overlap_pct:.0f}%",
          flush=True)
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
        # Reviewer-2 audit #4: reject pairs whose overlap is too small to be
        # a meaningful test (less than min-overlap-pct of frame-i pixels in-bounds).
        if ib < args.gate_cross_min_overlap_pct:
            print(f"  [{i} → {j}] SKIP  in_bounds={ib:.1f}% < gate {args.gate_cross_min_overlap_pct:.0f}%  "
                  f"(too-large baseline; not a meaningful test)", flush=True)
            continue
        # Reviewer-2 audit #1+#7: aligned median + p95 gates.
        med_pct = med * 100
        p95_pct = p95 * 100
        med_fail = med > args.gate_cross_rel
        p95_fail = p95 > args.gate_cross_p95
        if med_fail or p95_fail:
            tag = "FAIL"
            if med_fail and p95_fail:
                why = f"median>{args.gate_cross_rel*100:.1f}% AND p95>{args.gate_cross_p95*100:.1f}%"
            elif med_fail:
                why = f"median>{args.gate_cross_rel*100:.1f}%"
            else:
                why = f"p95>{args.gate_cross_p95*100:.1f}%"
            print(f"  [{i} → {j}] {tag}  median_rel={med_pct:.2f}%  p95_rel={p95_pct:.2f}%  "
                  f"n_valid={nv} n_occluded={nocc} in_bounds={ib:.1f}%  ({why})", flush=True)
            cross_fail = True
        else:
            print(f"  [{i} → {j}] PASS  median_rel={med_pct:.2f}%  p95_rel={p95_pct:.2f}%  "
                  f"n_valid={nv} n_occluded={nocc} in_bounds={ib:.1f}%", flush=True)

    if fail or cross_fail:
        print(f"\n[FAIL] {args.config}", flush=True)
        return 1
    print(f"\n[PASS] {args.config}  (self-reproj + cross-frame both clean)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
