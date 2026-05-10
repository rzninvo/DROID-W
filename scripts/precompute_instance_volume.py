"""
B.3 — accumulate per-track 3D point cloud (world frame) across all KFs.

For every global track from B.2, walk every per-KF instance, unproject mask
pixels via depth + pose to world XYZ, and concatenate.

POSE CONVENTION: world XYZ is in OpenGL c2w convention (X right, Y up, Z
back), inherited from BaseDataset.Replica.load_poses (which deliberately
does NOT apply NICE-SLAM's Y/Z flip — see datasets.py:Replica docstring).
B.5 mesh-vertex projection MUST apply the OpenGL→OpenCV flip at entry
when comparing against NICE-SLAM-rendered Replica meshes.

STRIDE NOTE: This script uses pixel stride=4 by default (~6.25%/instance,
~50 pts per 800-px instance). B.2's voxel-IoU pass uses stride=8 (cheaper,
just for merge-test). Two-strides-by-design — change in lockstep.

Inputs:
    panoptic.npz   : per-KF instance masks (B.1)
    tracks.npz     : global_track_id per instance (B.2)
    radseg_features.npz : kf_global_indices for pose lookup
    scene config   : dataset stream for depth + pose

Output (<panoptic.parent>/instance_volumes.npz):
    points          : (P_total, 3) float32  — world XYZ, all tracks concat.
    point_src_kf    : (P_total,)   int32    — source KF (local idx) per point.
                      Reviewer audit (Report 18 B.3): needed for per-frame
                      mIoU debugging at B.5; +33 MB on room0 (free at compress).
    track_offsets   : (G+1,)       int64    — points[off[g]:off[g+1]] is track g.
                      INVARIANT: track_offsets[g] == track_offsets[g+1] is legal
                      (empty track); consumers MUST guard:
                          if track_n_points[g] == 0: continue
    track_id        : (G,)         int64    — same as global_track_id; cross-ref.
    track_n_points  : (G,)         int64    — bookkeeping.
    voxel_size      : float                 — 0.05 m (recorded; subsample uses it).
    scene           : str                   — config path.

Subsample is VOXEL-GRID at `voxel_size` (Reviewer audit, Report 18 B.3 #2):
random-array-index subsampling biased toward densely-photographed near-camera
regions; voxel-grid downsample preserves spatial uniformity which is what
B.5 mesh-vertex assignment cares about.

Verification gate (Plan B B.3):
  - 1-5 M total points per Replica scene.
  - 50-5000 per-instance mean.
  - 1% sample re-projects into source KF within 3 px (validated by reviewer).

Usage (cvg):
    python scripts/precompute_instance_volume.py \\
        --panoptic Outputs/Replica/room0/panoptic.npz \\
        --tracks   Outputs/Replica/room0/tracks.npz \\
        --features Outputs/Replica/room0/radseg_features.npz \\
        --config   configs/RGBD/Replica/room0.yaml \\
        --output   Outputs/Replica/room0/instance_volumes.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

torch.backends.cudnn.deterministic = True

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _instance_world_points(mask: np.ndarray, depth: np.ndarray, pose_c2w: np.ndarray,
                           fx: float, fy: float, cx: float, cy: float,
                           min_depth: float = 0.05, max_depth: float = 10.0,
                           stride: int = 4) -> np.ndarray:
    """Subsampled world-frame point cloud for `mask`. Returns (M, 3) float32.

    Mirrors `scripts/build_panoptic_tracks.py:_instance_world_points` but with
    a coarser default stride (4 → ~12 pts per 200-px mask vs B.2's stride=8).
    """
    if not mask.any():
        return np.zeros((0, 3), dtype=np.float32)
    ys, xs = np.where(mask)
    if stride > 1:
        ys = ys[::stride]
        xs = xs[::stride]
    z = depth[ys, xs].astype(np.float64)
    valid = (z >= min_depth) & (z <= max_depth) & np.isfinite(z)
    if not valid.any():
        return np.zeros((0, 3), dtype=np.float32)
    z = z[valid]
    xs = xs[valid].astype(np.float64)
    ys = ys[valid].astype(np.float64)
    x_cam = (xs - cx) * z / fx
    y_cam = (ys - cy) * z / fy
    cam = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=0)
    world = pose_c2w @ cam
    return world[:3].T.astype(np.float32)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--panoptic", required=True, type=str)
    p.add_argument("--tracks", required=True, type=str)
    p.add_argument("--features", required=True, type=str)
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--output", default=None, type=str,
                   help="Default: <panoptic.parent>/instance_volumes.npz")
    p.add_argument("--stride", default=4, type=int,
                   help="Per-mask pixel stride. 4 → ~6.25%% pixels per instance, "
                        "~12 pts per 200-px instance, ~50 pts per 800-px instance.")
    p.add_argument("--voxel-size", default=0.05, type=float,
                   help="Recorded in metadata for downstream voxel-aware consumers.")
    p.add_argument("--max-points-per-track", default=200_000, type=int,
                   help="Cap. Tracks with more points get random-subsampled at save.")
    args = p.parse_args()

    pan_path = Path(args.panoptic);   feat_path = Path(args.features)
    trk_path = Path(args.tracks);     cfg_path = Path(args.config)
    if not pan_path.is_absolute(): pan_path = REPO_ROOT / pan_path
    if not feat_path.is_absolute(): feat_path = REPO_ROOT / feat_path
    if not trk_path.is_absolute(): trk_path = REPO_ROOT / trk_path
    if not cfg_path.is_absolute(): cfg_path = REPO_ROOT / cfg_path
    out_path = Path(args.output) if args.output else pan_path.parent / "instance_volumes.npz"
    if not out_path.is_absolute(): out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[setup] panoptic={pan_path}\n[setup] tracks={trk_path}", flush=True)
    print(f"[setup] features={feat_path}\n[setup] config={cfg_path}", flush=True)

    pan = np.load(pan_path)
    masks = pan["masks"]                                # (K, H, W) uint8
    per_kf_offsets = pan["per_kf_offsets"]
    K_total = masks.shape[0]
    H, W = int(masks.shape[1]), int(masks.shape[2])

    trk = np.load(trk_path)
    global_track_id = trk["global_track_id"]            # (K_total,) int64
    G = int(trk["n_tracks_total"])
    print(f"[load] K_total={K_total}  G={G}  HxW={H}x{W}", flush=True)

    from src import config as droid_config
    from src.utils.datasets import get_dataset
    cfg = droid_config.load_config(str(cfg_path))
    stream = get_dataset(cfg)
    fx, fy, cx, cy = float(stream.fx), float(stream.fy), float(stream.cx), float(stream.cy)

    # ── Pass 1: accumulate points per track + per-track source-KF arrays ───
    per_track_points: List[List[np.ndarray]] = [[] for _ in range(G)]
    per_track_src_kf: List[List[np.ndarray]] = [[] for _ in range(G)]
    cached_kf = -1
    cached_depth = None
    cached_pose = None
    n_skipped = 0

    t0 = time.time()
    for inst in range(K_total):
        gid = int(global_track_id[inst])
        if gid < 0:
            n_skipped += 1
            continue
        kf_local = int(np.searchsorted(per_kf_offsets[1:], inst, side="right"))
        if kf_local != cached_kf:
            _, _, depth_t, pose_t = stream[kf_local]
            cached_depth = depth_t.cpu().numpy() if torch.is_tensor(depth_t) else np.asarray(depth_t)
            cached_pose = pose_t.cpu().numpy() if torch.is_tensor(pose_t) else np.asarray(pose_t)
            cached_kf = kf_local
        m = masks[inst].astype(bool)
        pts = _instance_world_points(m, cached_depth, cached_pose, fx, fy, cx, cy,
                                     stride=args.stride)
        if pts.shape[0] == 0:
            continue
        per_track_points[gid].append(pts)
        # Reviewer audit #1: persist source KF per point for B.5 debugging.
        per_track_src_kf[gid].append(np.full(pts.shape[0], kf_local, dtype=np.int32))

        if (inst + 1) % 1000 == 0 or inst == K_total - 1:
            el = time.time() - t0
            eta = el / max(1, inst + 1) * (K_total - inst - 1)
            print(f"  [{inst+1:5d}/{K_total}]  ({el:5.1f}s elapsed, ~{eta:5.1f}s ETA)",
                  flush=True)
    print(f"[unproject] {n_skipped} unassigned instances skipped", flush=True)

    # ── Pass 2: pack into ragged arrays + voxel-grid subsample to cap ──────
    rng = np.random.default_rng(42)
    track_arrays: List[np.ndarray] = []
    track_src_arrays: List[np.ndarray] = []
    track_n_points = np.zeros(G, dtype=np.int64)
    for gid in range(G):
        if not per_track_points[gid]:
            track_arrays.append(np.zeros((0, 3), dtype=np.float32))
            track_src_arrays.append(np.zeros((0,), dtype=np.int32))
            continue
        cat = np.concatenate(per_track_points[gid], axis=0)
        cat_src = np.concatenate(per_track_src_kf[gid], axis=0)
        if cat.shape[0] > args.max_points_per_track:
            # Reviewer audit #2: voxel-grid subsample preserves spatial
            # uniformity (B.5 mesh-vertex projection cares about near-corner
            # coverage, which uniform-array-index sampling under-represents).
            voxel_keys = np.floor(cat / args.voxel_size).astype(np.int64)
            # Pack 3 ints into a 1-D string key and unique-by-first.
            # (Scalable up to ~1e6 points; for room0 we have ~200k.)
            packed = (voxel_keys[:, 0] * 1_000_003 + voxel_keys[:, 1]) * 1_000_003 \
                     + voxel_keys[:, 2]
            _, unique_idx = np.unique(packed, return_index=True)
            unique_idx.sort()
            cat = cat[unique_idx]
            cat_src = cat_src[unique_idx]
            # If still over cap after voxel-dedupe, random-cap from remaining.
            if cat.shape[0] > args.max_points_per_track:
                idx = rng.choice(cat.shape[0], size=args.max_points_per_track, replace=False)
                cat = cat[idx]
                cat_src = cat_src[idx]
        track_arrays.append(cat)
        track_src_arrays.append(cat_src)
        track_n_points[gid] = cat.shape[0]

    points_concat = np.concatenate(track_arrays, axis=0).astype(np.float32) if track_arrays else \
                    np.zeros((0, 3), dtype=np.float32)
    src_kf_concat = np.concatenate(track_src_arrays, axis=0).astype(np.int32) if track_src_arrays else \
                    np.zeros((0,), dtype=np.int32)
    track_offsets = np.zeros(G + 1, dtype=np.int64)
    np.cumsum(track_n_points, out=track_offsets[1:])

    track_id = np.arange(G, dtype=np.int64)

    np.savez_compressed(
        out_path,
        points=points_concat,
        point_src_kf=src_kf_concat,
        track_offsets=track_offsets,
        track_id=track_id,
        track_n_points=track_n_points,
        voxel_size=np.float32(args.voxel_size),
        scene=np.array(str(cfg_path)),
    )
    size_mb = out_path.stat().st_size / 1024 / 1024
    P_total = int(points_concat.shape[0])
    print(f"\n[done] {G} tracks, {P_total:,} total points "
          f"(mean {P_total/max(1,G):.0f}/track, max {int(track_n_points.max()) if G else 0})",
          flush=True)
    print(f"[save] {out_path}  ({size_mb:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
