"""
B.2 — cross-keyframe instance tracking → global IDs.

Inputs:
    panoptic.npz (from B.1)              : per-KF instances (mask, lang_emb_pca, score, kf_global_idx)
    radseg_features.npz (Phase B')        : kf_global_indices for pose lookup
    scene config (Replica or TUM)        : enables get_dataset(cfg) → poses + depth + K

Outputs (written to <panoptic.npz.parent>/tracks.npz):
    global_track_id   : (K_total,)  int64    — one ID per per-KF instance, dense in [0, G).
    track_emb_pca     : (G, 256)    fp16     — visibility-weighted fused embedding per track.
    track_kf_count    : (G,)        int64    — number of KFs each track was seen in.
    track_total_vis   : (G,)        int64    — total mask area (pixels) summed across views.
    track_centroid_w  : (G, 3)      fp32     — world-frame centroid (mean across views).
    n_tracks_total    : int                  — G.
    scene_id_offset   : int                  — global_track_id + scene_id_offset gives a globally-unique ID.

Algorithm — hybrid 2D-cosine + 3D-voxel merge (OVI-MAP §3.3 + ConceptGraphs):
  1. 2D pre-pass: walk KFs in dataset-frame order; apply MaskTracker
     (appearance + IoU + box-IoU; cosine on lang_emb_pca features).
  2. 3D voxel-vote merge: for each preliminary track, compute world-frame
     centroid per KF via depth-unproject + pose. Two tracks A and B merge if
       cos(e_A, e_B) >= 0.78  AND  voxel-set overlap (5cm) >= 0.30.
  3. Visibility-weighted fused embedding (OVI-MAP §3.2 Eq. 2):
       e_g = (sum_v vis_v * e_v) / sum_v vis_v;  L2-normalize.

Cross-scene leak guard: emit `scene_id_offset` so multi-scene eval can do
`unique_id = scene_offset + global_track_id` without collision.

Usage:
    python scripts/build_panoptic_tracks.py \\
        --panoptic Outputs/Replica/room0/panoptic.npz \\
        --features Outputs/Replica/room0/radseg_features.npz \\
        --config   configs/RGBD/Replica/room0.yaml \\
        --output   Outputs/Replica/room0/tracks.npz \\
        --scene-id 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

torch.backends.cudnn.deterministic = True

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.mono_priors.mask_classifier import MaskTracker


def _bbox_from_mask(mask: np.ndarray) -> List[float]:
    """[x1, y1, x2, y2] tight bbox of a bool mask. Returns [0,0,0,0] if empty."""
    if not mask.any():
        return [0.0, 0.0, 0.0, 0.0]
    ys, xs = np.where(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def _instance_centroid_world(mask: np.ndarray, depth: np.ndarray, pose_c2w: np.ndarray,
                             fx: float, fy: float, cx: float, cy: float,
                             min_depth: float = 0.05, max_depth: float = 10.0) -> np.ndarray:
    """Median XYZ in world frame of pixels under `mask` (median ≫ mean for noise robustness)."""
    if not mask.any():
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    H, W = mask.shape
    ys, xs = np.where(mask)
    z = depth[ys, xs].astype(np.float64)
    valid = (z >= min_depth) & (z <= max_depth) & np.isfinite(z)
    if not valid.any():
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    z = z[valid]
    xs = xs[valid].astype(np.float64)
    ys = ys[valid].astype(np.float64)
    x_cam = (xs - cx) * z / fx
    y_cam = (ys - cy) * z / fy
    cam = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=0)            # (4, M)
    world = pose_c2w @ cam                                                # (4, M)
    return np.median(world[:3], axis=1).astype(np.float32)


def _voxel_iou(centroids_a: np.ndarray, centroids_b: np.ndarray, voxel_size: float) -> float:
    """Voxel-set IoU. centroids_*: (M, 3). Quantize, set-intersect, set-union."""
    if centroids_a.shape[0] == 0 or centroids_b.shape[0] == 0:
        return 0.0
    a = set(map(tuple, np.floor(centroids_a / voxel_size).astype(np.int64)))
    b = set(map(tuple, np.floor(centroids_b / voxel_size).astype(np.int64)))
    inter = len(a & b)
    union = len(a | b)
    return float(inter / union) if union > 0 else 0.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--panoptic", required=True, type=str)
    p.add_argument("--features", required=True, type=str)
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--output", default=None, type=str,
                   help="Default: <panoptic.parent>/tracks.npz")
    p.add_argument("--scene-id", default=0, type=int,
                   help="Multi-scene leak guard (cross-scene global IDs disjoint).")
    p.add_argument("--scene-id-block", default=100_000, type=int,
                   help="scene_id_offset = scene_id * scene_id_block.")
    p.add_argument("--track-thresh-2d", default=0.55, type=float,
                   help="MaskTracker cosine + IoU score gate.")
    p.add_argument("--track-max-age", default=8, type=int,
                   help="MaskTracker drop-after-N-KFs.")
    p.add_argument("--merge-cos-thresh", default=0.78, type=float,
                   help="3D merge: cos(e_A, e_B) gate (OVI-MAP §3.3).")
    p.add_argument("--merge-voxel-iou-thresh", default=0.30, type=float,
                   help="3D merge: voxel-IoU gate (OVI-MAP §3.3).")
    p.add_argument("--voxel-size", default=0.05, type=float,
                   help="Voxel grid (m). 5 cm matches OVI-MAP / ConceptGraphs.")
    p.add_argument("--min-track-kf-count", default=1, type=int,
                   help="Drop tracks seen in fewer than N KFs.")
    args = p.parse_args()

    pan_path = Path(args.panoptic)
    feat_path = Path(args.features)
    cfg_path = Path(args.config)
    if not pan_path.is_absolute(): pan_path = REPO_ROOT / pan_path
    if not feat_path.is_absolute(): feat_path = REPO_ROOT / feat_path
    if not cfg_path.is_absolute(): cfg_path = REPO_ROOT / cfg_path
    out_path = Path(args.output) if args.output else pan_path.parent / "tracks.npz"
    if not out_path.is_absolute(): out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene_id_offset = int(args.scene_id) * int(args.scene_id_block)

    print(f"[setup] panoptic={pan_path}", flush=True)
    print(f"[setup] features={feat_path}  config={cfg_path}", flush=True)
    print(f"[setup] scene_id={args.scene_id}  scene_id_offset={scene_id_offset}", flush=True)

    pan = np.load(pan_path)
    masks = pan["masks"]                                # (K, H, W) uint8
    lang_emb = pan["lang_emb_pca"]                      # (K, 256) fp16
    scores = pan["score"]                                # (K,) fp16
    areas = pan["area"]                                  # (K,) int64
    kf_global_idx = pan["kf_global_idx"]                 # (K,) int64
    per_kf_offsets = pan["per_kf_offsets"]               # (N+1,) int64
    K_total = masks.shape[0]
    N_kfs = len(per_kf_offsets) - 1
    H, W = int(masks.shape[1]), int(masks.shape[2])
    print(f"[panoptic] K_total={K_total}  N_kfs={N_kfs}  HxW={H}x{W}", flush=True)

    feat = np.load(feat_path)
    kf_global_indices_feat = feat["kf_global_indices"] if "kf_global_indices" in feat.files \
        else feat["kf_indices"]
    print(f"[features] kf_global_indices range {int(kf_global_indices_feat.min())}..{int(kf_global_indices_feat.max())}",
          flush=True)

    # Dataset stream — for poses + depth lookup at the per-instance KF.
    from src import config as droid_config
    from src.utils.datasets import get_dataset
    cfg = droid_config.load_config(str(cfg_path))
    stream = get_dataset(cfg)
    fx, fy, cx, cy = float(stream.fx), float(stream.fy), float(stream.cx), float(stream.cy)
    print(f"[stream] N_frames={len(stream)}  K  fx={fx:.2f} cx={cx:.2f}", flush=True)

    # ── 2D PRE-PASS: appearance-based MaskTracker per KF ───────────────────
    print(f"\n=== 2D MaskTracker pass (cosine+IoU @ thr={args.track_thresh_2d}) ===", flush=True)
    tracker = MaskTracker(match_thresh=args.track_thresh_2d, max_age=args.track_max_age)
    track_ids_per_inst = np.full(K_total, -1, dtype=np.int64)
    t0 = time.time()
    for kf in range(N_kfs):
        start, end = int(per_kf_offsets[kf]), int(per_kf_offsets[kf + 1])
        detections = []
        for inst in range(start, end):
            m = masks[inst].astype(bool)
            box = _bbox_from_mask(m)
            emb = np.asarray(lang_emb[inst], dtype=np.float32)
            # MaskTracker expects clip_emb on instance dict; we pass pca-256
            # (same cosine semantics, just smaller dim — confirmed on f3+rep).
            detections.append({
                "mask": m, "box": box, "label": "object", "clip_emb": emb,
            })
        out = tracker.update(detections)
        for k, det in enumerate(out):
            track_ids_per_inst[start + k] = int(det["track_id"])

    n_2d_tracks = len(set(track_ids_per_inst.tolist()))
    print(f"[2D] {n_2d_tracks} preliminary tracks  ({time.time()-t0:.1f}s)", flush=True)

    # ── COMPUTE 3D CENTROIDS PER (track_id, kf) ────────────────────────────
    # For voxel-IoU merge we need the world XYZ of every per-KF instance under
    # each prelim track.
    print(f"\n=== unproject centroids to world frame ===", flush=True)
    # Cache (depth, pose) per dataset frame to avoid repeated disk reads.
    track_centroids: Dict[int, List[np.ndarray]] = {}
    track_areas: Dict[int, List[int]] = {}
    track_embs: Dict[int, List[np.ndarray]] = {}
    track_kfs: Dict[int, List[int]] = {}

    # stream is already strided: stream[i] returns the i-th KF (local position),
    # NOT frame `i` of the original undecimated dataset. kf_global_idx stores
    # the original frame number (for mesh-GT lookup downstream), but for stream
    # depth/pose access we use the local KF position directly.
    cached_kf: int = -1
    cached_depth: np.ndarray = None
    cached_pose: np.ndarray = None
    t0 = time.time()
    for inst in range(K_total):
        kf_local = int(np.searchsorted(per_kf_offsets[1:], inst, side="right"))
        if kf_local != cached_kf:
            _, _, depth_t, pose_t = stream[kf_local]
            cached_depth = depth_t.cpu().numpy() if torch.is_tensor(depth_t) else np.asarray(depth_t)
            cached_pose = pose_t.cpu().numpy() if torch.is_tensor(pose_t) else np.asarray(pose_t)
            cached_kf = kf_local
        m = masks[inst].astype(bool)
        c = _instance_centroid_world(m, cached_depth, cached_pose, fx, fy, cx, cy)
        if not np.isfinite(c).all():
            continue
        tid = int(track_ids_per_inst[inst])
        track_centroids.setdefault(tid, []).append(c)
        track_areas.setdefault(tid, []).append(int(areas[inst]))
        track_embs.setdefault(tid, []).append(np.asarray(lang_emb[inst], dtype=np.float32))
        track_kfs.setdefault(tid, []).append(int(kf_global_idx[inst]))
    print(f"[3D] centroids computed for {len(track_centroids)} tracks  ({time.time()-t0:.1f}s)",
          flush=True)

    # ── 3D MERGE: cos-and-voxel-IoU → connected components → final tracks ──
    print(f"\n=== 3D voxel-vote merge (cos≥{args.merge_cos_thresh}, voxel-IoU≥{args.merge_voxel_iou_thresh}) ===",
          flush=True)
    track_ids_sorted = sorted(track_centroids.keys())
    # Per-track fused embedding (visibility-weighted) for the merge cosine test.
    fused: Dict[int, np.ndarray] = {}
    voxels: Dict[int, np.ndarray] = {}
    for tid in track_ids_sorted:
        embs = np.stack(track_embs[tid])                               # (M, 256)
        weights = np.asarray(track_areas[tid], dtype=np.float32)
        if weights.sum() <= 0:
            continue
        e = (embs * weights[:, None]).sum(axis=0) / weights.sum()
        e = e / (np.linalg.norm(e) + 1e-8)
        fused[tid] = e
        voxels[tid] = np.stack(track_centroids[tid])

    # Union-find over track IDs.
    parent = {tid: tid for tid in fused}
    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb: parent[ra] = rb

    n_merges = 0
    track_id_list = list(fused.keys())
    for i, ta in enumerate(track_id_list):
        for tb in track_id_list[i + 1:]:
            if _find(ta) == _find(tb):
                continue
            cos = float(np.dot(fused[ta], fused[tb]))
            if cos < args.merge_cos_thresh:
                continue
            iou3d = _voxel_iou(voxels[ta], voxels[tb], args.voxel_size)
            if iou3d < args.merge_voxel_iou_thresh:
                continue
            _union(ta, tb)
            n_merges += 1
    print(f"[merge] {n_merges} pairwise merges", flush=True)

    # ── Compact local-track-id → global-id (dense [0, G)) ──────────────────
    root_to_global: Dict[int, int] = {}
    for tid in fused:
        r = _find(tid)
        if r not in root_to_global:
            root_to_global[r] = len(root_to_global)
    G = len(root_to_global)
    print(f"[final] {G} global tracks (after merge)", flush=True)

    # Per-instance global_track_id.
    global_track_id = np.full(K_total, -1, dtype=np.int64)
    for inst in range(K_total):
        tid = int(track_ids_per_inst[inst])
        if tid in fused:
            global_track_id[inst] = root_to_global[_find(tid)]

    # Drop tracks below min-kf-count.
    counts = np.zeros(G, dtype=np.int64)
    for inst in range(K_total):
        gid = int(global_track_id[inst])
        if gid >= 0:
            # de-dupe per KF (an instance is on one KF; counts unique-KF below)
            pass
    # Compute per-track unique-KF count.
    kf_seen_per_track: Dict[int, set] = {gid: set() for gid in range(G)}
    for inst in range(K_total):
        gid = int(global_track_id[inst])
        if gid >= 0:
            kf_seen_per_track[gid].add(int(kf_global_idx[inst]))
    track_kf_count = np.array([len(kf_seen_per_track[g]) for g in range(G)], dtype=np.int64)
    keep_track = track_kf_count >= args.min_track_kf_count
    print(f"[filter] tracks with ≥{args.min_track_kf_count} KFs: {int(keep_track.sum())}/{G}",
          flush=True)
    # Re-index after filter.
    new_idx = -np.ones(G, dtype=np.int64)
    new_idx[keep_track] = np.arange(int(keep_track.sum()))
    G_final = int(keep_track.sum())
    final_global_id = np.full(K_total, -1, dtype=np.int64)
    for inst in range(K_total):
        gid = int(global_track_id[inst])
        if gid >= 0 and keep_track[gid]:
            final_global_id[inst] = new_idx[gid]

    # Per-final-track: visibility-weighted fused embedding + total vis + centroid.
    track_emb_final = np.zeros((G_final, lang_emb.shape[1]), dtype=np.float32)
    track_total_vis = np.zeros(G_final, dtype=np.int64)
    track_centroid_final = np.zeros((G_final, 3), dtype=np.float32)
    centroid_count = np.zeros(G_final, dtype=np.int64)
    track_n_inst = np.zeros(G_final, dtype=np.int64)

    for inst in range(K_total):
        gid = int(final_global_id[inst])
        if gid < 0:
            continue
        a = float(areas[inst])
        track_emb_final[gid] += a * np.asarray(lang_emb[inst], dtype=np.float32)
        track_total_vis[gid] += int(areas[inst])
        # Centroid: re-pull from cache. Faster: keep the dict above.
        # We didn't keep a per-instance centroid; quick re-walk:
        track_n_inst[gid] += 1

    # Re-walk for centroids (needs depth/pose cache again — small cost).
    cached_kf = -1
    for inst in range(K_total):
        gid = int(final_global_id[inst])
        if gid < 0:
            continue
        kf_local = int(np.searchsorted(per_kf_offsets[1:], inst, side="right"))
        if kf_local != cached_kf:
            _, _, depth_t, pose_t = stream[kf_local]
            cached_depth = depth_t.cpu().numpy() if torch.is_tensor(depth_t) else np.asarray(depth_t)
            cached_pose = pose_t.cpu().numpy() if torch.is_tensor(pose_t) else np.asarray(pose_t)
            cached_kf = kf_local
        m = masks[inst].astype(bool)
        c = _instance_centroid_world(m, cached_depth, cached_pose, fx, fy, cx, cy)
        if not np.isfinite(c).all():
            continue
        track_centroid_final[gid] += c
        centroid_count[gid] += 1

    for gid in range(G_final):
        if track_total_vis[gid] > 0:
            track_emb_final[gid] /= float(track_total_vis[gid])
            n = np.linalg.norm(track_emb_final[gid]) + 1e-8
            track_emb_final[gid] /= n
        if centroid_count[gid] > 0:
            track_centroid_final[gid] /= float(centroid_count[gid])

    track_kf_count_final = np.array(
        [track_kf_count[g] for g in range(G) if keep_track[g]], dtype=np.int64
    )

    np.savez_compressed(
        out_path,
        global_track_id=final_global_id,                           # (K_total,)
        track_emb_pca=track_emb_final.astype(np.float16),          # (G_final, 256)
        track_kf_count=track_kf_count_final,                       # (G_final,)
        track_total_vis=track_total_vis,                           # (G_final,)
        track_centroid_w=track_centroid_final,                     # (G_final, 3)
        track_n_instances=track_n_inst,                            # (G_final,)
        n_tracks_total=np.int64(G_final),
        scene_id_offset=np.int64(scene_id_offset),
        scene=np.array(str(args.config)),
        feature_dim=np.int64(lang_emb.shape[1]),
    )
    size_kb = out_path.stat().st_size / 1024
    print(f"\n[done] {G_final} global tracks, {int((final_global_id >= 0).sum())}/{K_total} "
          f"instances assigned. {size_kb:.1f} KB at {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
