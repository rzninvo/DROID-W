"""Plan-v2 §Step 4 / B.2 — cross-KF instance tracking via OVI-MAP §3.1.

Implements the exact recipe of arXiv:2603.26541 §3.1:

  For each incoming 2D mask M_{t,j}, back-project to a point cloud P_{t,j}
  in the metric world frame; for each existing instance label k, count

    Omega_{j,k} = |{ x in P_{t,j} : V(x).label == k }|

  Pick k* = argmax_k Omega_{j,k}.
  If Omega_{j,k*} > theta_assoc -> assign P_{t,j} -> S_{k*};
  else -> spawn new instance S_new.
  Vote: O_v(label_assigned) += 1; v.label = argmax_k O_v(k).

We deviate from the paper in:
  - depth source: monocular DROID-W droid_disps_up (metric via scale=...)
    versus OVI-MAP RGB-D sensor depth
  - voxel grid: simple sparse dict on metric world (voxel_size=0.1m to
    match OVI-MAP §3); TSDF integration deferred to B.3 — for B.2 the
    label-vote field alone gives the cross-KF identity assignment.

Output: panoptic_v7_tracks.npz with the global track ID per mask (in the
same flat order as panoptic_v7_cropformer_refined.masks).

Usage (cvg, droid-w env):
  python scripts/build_panoptic_tracks_v7.py \
    --refined-npz Outputs/TUM_RGBD/freiburg3_walking_static/panoptic_v7_cropformer_refined.npz \
    --video-npz   Outputs/TUM_RGBD/freiburg3_walking_static/video.npz \
    --out         Outputs/TUM_RGBD/freiburg3_walking_static/panoptic_v7_tracks.npz \
    --voxel-size 0.1 --theta-assoc 0.3 --theta-merge 0.5 \
    --subsample 4 --disp-eps 0.01
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np


def backproject_kf(disp: np.ndarray, intr_full: np.ndarray, T_cw: np.ndarray,
                   scale: float, disp_eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel deprojection at 384x512.

    Returns (X_world[H,W,3], valid[H,W]).
    """
    fx, fy, cx, cy = intr_full
    H, W = disp.shape
    valid = disp > disp_eps
    depth = np.where(valid, scale / np.maximum(disp, disp_eps), 0.0)
    u = np.arange(W, dtype=np.float32)[None, :].repeat(H, axis=0)
    v = np.arange(H, dtype=np.float32)[:, None].repeat(W, axis=1)
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    P_cam = np.stack([X, Y, Z, np.ones_like(X)], axis=-1)  # (H,W,4)
    P_world = (T_cw @ P_cam.reshape(-1, 4).T).T.reshape(H, W, 4)
    return P_world[..., :3].astype(np.float32), valid


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--refined-npz", required=True, type=str)
    p.add_argument("--video-npz", required=True, type=str)
    p.add_argument("--out", required=True, type=str)
    p.add_argument("--voxel-size", default=0.1, type=float,
                   help="metric voxel size (OVI-MAP §3 default 0.1m)")
    p.add_argument("--theta-assoc", default=0.3, type=float,
                   help="Omega/|P_tj| threshold for assignment (plan-v2 default)")
    p.add_argument("--theta-merge", default=0.5, type=float,
                   help="merge-pass overlap threshold (plan-v2 default)")
    p.add_argument("--subsample", default=4, type=int,
                   help="pixel stride when back-projecting mask into world")
    p.add_argument("--disp-eps", default=0.01, type=float,
                   help="lower bound on droid_disps_up to drop sky / invalid")
    p.add_argument("--depth-clip", default=10.0, type=float,
                   help="upper clip on metric depth (m)")
    p.add_argument("--min-points", default=50, type=int,
                   help="drop a mask if its valid 3D point count is below this")
    args = p.parse_args()

    t0 = time.time()
    refined = np.load(args.refined_npz, allow_pickle=True)
    video = np.load(args.video_npz, allow_pickle=True)

    masks = refined["masks"]           # (N_total, H_mask, W_mask) bool
    offsets = refined["seg_kf_offsets"] # (n_kf+1,)
    n_kf = int(refined["n_keyframes"])
    H_mask, W_mask = masks.shape[1], masks.shape[2]
    print(f"[load] refined: {masks.shape[0]} masks, {n_kf} KFs, masks={H_mask}x{W_mask}",
          flush=True)

    poses = video["poses"]              # (n_kf, 4, 4)
    droid_up = video["droid_disps_up"]  # (n_kf, H, W) where H,W matches mask
    intr_ba = video["intrinsics"]       # (n_kf, 4) at BA grid 48x64
    scale = float(video["scale"])
    print(f"[load] video: poses={poses.shape}, droid_disps_up={droid_up.shape}, "
          f"scale={scale:.4f}", flush=True)

    # If the mask resolution does not match depth, we down/up-sample masks
    # on the fly to the depth resolution (nearest-neighbour, lossless for
    # boolean masks). The track is recorded in the original mask flat-list
    # order, so renderers can still use the input masks at native res.
    Hd, Wd = droid_up.shape[1], droid_up.shape[2]
    needs_mask_resize = (Hd, Wd) != (H_mask, W_mask)
    if needs_mask_resize:
        print(f"[setup] mask {H_mask}x{W_mask} != depth {Hd}x{Wd}; "
              f"will resize masks to {Hd}x{Wd} for back-projection",
              flush=True)
        import cv2 as _cv
        cv2_inter = _cv.INTER_NEAREST
    intr_factor = float(Hd / 48)
    intrinsics_full = intr_ba * intr_factor
    print(f"[setup] intrinsics x{intr_factor:.0f} -> "
          f"fx={intrinsics_full[0,0]:.2f} fy={intrinsics_full[0,1]:.2f} "
          f"cx={intrinsics_full[0,2]:.2f} cy={intrinsics_full[0,3]:.2f}",
          flush=True)

    s = args.subsample
    vox = args.voxel_size
    inv_vox = 1.0 / vox

    # Per-voxel state: dict{(i,j,k) -> Counter(label_votes)}
    voxel_votes: dict[tuple[int, int, int], Counter[int]] = {}
    # Per-voxel current argmax label (cached, recomputed when votes change)
    voxel_label: dict[tuple[int, int, int], int] = {}

    n_masks_total = int(masks.shape[0])
    global_track_ids = -np.ones(n_masks_total, dtype=np.int64)  # -1 = unassigned
    next_label = 1  # 0 reserved for "no label"
    n_new, n_assigned, n_skipped = 0, 0, 0
    omega_log: list[tuple[int, int, int, int]] = []  # (kf, mask_idx, omega_best, |P|)

    t_kf_total = 0.0
    for k in range(n_kf):
        t_kf = time.time()
        T_cw = poses[k].astype(np.float32)
        disp = droid_up[k].astype(np.float32)
        intr_full = intrinsics_full[k].astype(np.float32)

        # Back-project the full KF once.
        X_world, valid = backproject_kf(disp, intr_full, T_cw, scale,
                                        args.disp_eps)
        # Apply metric clip
        depth_z = (T_cw[:3, 3:4].T - X_world.reshape(-1, 3))  # not used; clip in cam
        # Cheaper: re-derive depth = z component of camera-frame point.
        # We computed P_world via T_cw @ P_cam; recompute Z explicitly is
        # cheaper than inverting. For depth clip we just re-use:
        z_cam = scale / np.maximum(disp, args.disp_eps)
        valid = valid & (z_cam < args.depth_clip)

        # Subsample the (H,W) grid by stride s
        X_world_sub = X_world[::s, ::s, :]            # (Hs, Ws, 3)
        valid_sub = valid[::s, ::s]
        H_sub, W_sub = valid_sub.shape

        # Pre-compute voxel indices for every subsampled valid pixel
        vox_ijk = np.floor(X_world_sub * inv_vox).astype(np.int32)

        s_off, e_off = int(offsets[k]), int(offsets[k + 1])
        for m_idx_in_kf, m_global in enumerate(range(s_off, e_off)):
            mask = masks[m_global]                    # (H_mask, W_mask) bool
            if needs_mask_resize:
                mask = _cv.resize(mask.astype(np.uint8), (Wd, Hd),
                                  interpolation=cv2_inter).astype(bool)
            mask_sub = mask[::s, ::s]                 # (Hs, Ws)
            keep = mask_sub & valid_sub
            if keep.sum() < args.min_points:
                n_skipped += 1
                continue
            ijk = vox_ijk[keep]                       # (P, 3) int32

            # Compute Omega_{j,k} = # of P_tj points landing in voxels with label k.
            # Walk through ijk once, look up voxel_label[(i,j,k_)] if present.
            omega = Counter()
            for row in ijk:
                key = (int(row[0]), int(row[1]), int(row[2]))
                lab = voxel_label.get(key, 0)
                if lab != 0:
                    omega[lab] += 1

            P_size = ijk.shape[0]
            if omega:
                k_star, omega_best = omega.most_common(1)[0]
                ratio = omega_best / max(P_size, 1)
            else:
                k_star, omega_best, ratio = 0, 0, 0.0

            if ratio > args.theta_assoc and k_star != 0:
                assigned_label = k_star
                n_assigned += 1
            else:
                assigned_label = next_label
                next_label += 1
                n_new += 1
            global_track_ids[m_global] = assigned_label
            omega_log.append((k, m_idx_in_kf, omega_best, P_size))

            # Vote and update voxel labels for THIS mask's points.
            for row in ijk:
                key = (int(row[0]), int(row[1]), int(row[2]))
                cnt = voxel_votes.setdefault(key, Counter())
                cnt[assigned_label] += 1
                # Update voxel argmax label
                top = cnt.most_common(1)[0][0]
                voxel_label[key] = top

        t_kf_total += time.time() - t_kf
        if (k + 1) % 10 == 0 or k == n_kf - 1:
            print(f"[kf {k+1:3d}/{n_kf}] tracks={next_label-1}  voxels={len(voxel_votes)}  "
                  f"assigned={n_assigned}  new={n_new}  skipped={n_skipped}  "
                  f"elapsed={time.time()-t0:.1f}s",
                  flush=True)

    # Final tally: track_kf_counts (how many KFs each global label appears in)
    track_kf_counts = np.zeros(next_label, dtype=np.int64)
    seen_per_kf: dict[int, set[int]] = {}
    for k in range(n_kf):
        s_off, e_off = int(offsets[k]), int(offsets[k + 1])
        labs = global_track_ids[s_off:e_off]
        for l in set(labs.tolist()):
            if l > 0:
                track_kf_counts[l] += 1

    n_global_tracks = int((np.bincount(global_track_ids[global_track_ids > 0],
                                       minlength=next_label) > 0).sum())
    print(f"\n[done] {next_label-1} raw labels -> {n_global_tracks} global tracks "
          f"with >=1 mask", flush=True)
    print(f"[done] assigned={n_assigned}  new={n_new}  skipped(<{args.min_points} pts)={n_skipped}",
          flush=True)
    print(f"[done] voxels (vote)={len(voxel_votes)}  voxels (labeled)={len(voxel_label)}",
          flush=True)
    print(f"[done] kf-loop wall={t_kf_total:.1f}s  total wall={time.time()-t0:.1f}s",
          flush=True)

    # Histogram of how many KFs each track persists.
    persistence = np.bincount(track_kf_counts[track_kf_counts > 0])
    for n_kf_appear in (1, 2, 3, 5, 10, 20, 50):
        n_tracks_at = (track_kf_counts >= n_kf_appear).sum()
        print(f"[hist] tracks visible in >={n_kf_appear:3d} KFs: {n_tracks_at}",
              flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             schema_version=np.int64(1),
             algorithm=np.array("OVI-MAP_3.1_label_vote", dtype="U64"),
             n_keyframes=np.int64(n_kf),
             n_masks_total=np.int64(n_masks_total),
             n_global_tracks=np.int64(n_global_tracks),
             global_track_ids=global_track_ids,
             track_kf_counts=track_kf_counts,
             voxel_size=np.float32(args.voxel_size),
             theta_assoc=np.float32(args.theta_assoc),
             theta_merge=np.float32(args.theta_merge),
             subsample_stride=np.int64(args.subsample),
             disp_eps=np.float32(args.disp_eps),
             depth_clip=np.float32(args.depth_clip),
             scale=np.float32(scale),
             walltime_seconds=np.float32(time.time() - t0))
    print(f"[save] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
