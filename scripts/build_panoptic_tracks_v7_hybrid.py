"""Plan-v2 §Step 4 / B.2 -- Fix B: hybrid association (voxel overlap +
DINO feature ReID) on top of OVI-MAP §3.1 label-vote.

This is the B.2 successor. The original build_panoptic_tracks_v7.py is
preserved unchanged. This file adds a second association cue --
mean-pooled DINO patch-feature cosine -- because OVI-MAP §3.1 pure
voxel coincidence cannot follow movers (a walker at ~1 m/s moves ~3
voxels per KF at 0.1 m voxel size).

References (cross-validated against primary sources + GitHub repos):

  ConceptGraphs (Gu et al., ICRA 2024, arXiv:2309.16650)
    aggregate_similarities() in slam/cfslam_pipeline_batch.py:
      sims = (1 + phys_bias) * spatial_sim + (1 - phys_bias) * visual_sim
      greedy argmax if row_max > sim_threshold
    Default phys_bias = 0  ->  equal weight, sims in [0, 2].
    Default sim_threshold = 1.1.

  DEVA (Cheng et al., ICCV 2023, arXiv:2309.03903)
    merge_by_iou() in deva/inference/segment_merging.py:
      iou = intersection / union ; matched = iou > 0.5
      greedy per (new -> existing) order
    DEVA is RGB-only via XMem propagation; we get the same effect
    using DROID-W's poses + droid_disps for flow-free Re-ID via DINO
    cosine, which is paper-faithful to ConceptGraphs and stays in our
    DROID-W feature ecosystem.

Algorithm per KF t, per refined B.1 mask M_tj:
  1. Spatial term (OVI-MAP §3.1 unchanged):
        Omega_jk = #points P_tj landing in voxels labeled k
        spatial_jk = Omega_jk / |P_tj|     (in [0, 1])
  2. Visual term (NEW):
        f_j = mean over (mask_pixels) of L2-normalized DINO patch feat
        visual_jk = (cos(f_j, mean_DINO[k]) + 1) / 2   (in [0, 1])
        (paper-faithful to ConceptGraphs ϕ_sem = f^T f' / 2 + 1/2.)
  3. Combined (ConceptGraphs sim_sum):
        sims_jk = (1 + phys_bias) * spatial_jk
                + (1 - phys_bias) * visual_jk
        k* = argmax_k sims_jk
        assign if sims_jk* > sim_threshold AND k* not used in this KF
        else spawn new label
  4. Vote per voxel (unchanged); update track's running-mean DINO
     feature with this mask's f_j.

Preserves Fix A: within-KF distinctness + HSV-friendly palette in the
downstream renderer.

Output: panoptic_v7_tracks_hybrid.npz schema v2 with the same flat-list
order as the input refined-npz mask stack.

Usage (cvg, droid-w env):
  python scripts/build_panoptic_tracks_v7_hybrid.py \
    --refined-npz Outputs/.../panoptic_v7_cropformer.npz \
    --video-npz   Outputs/.../video.npz \
    --out         Outputs/.../panoptic_v7_tracks_hybrid.npz \
    --voxel-size 0.1 --sim-threshold 1.1 --phys-bias 0.0 \
    --subsample 2 --disp-eps 0.01 --min-points 10 \
    --enforce-within-kf-distinct 1
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np


def backproject_kf(disp: np.ndarray, intr_full: np.ndarray, T_cw: np.ndarray,
                   scale: float, disp_eps: float) -> tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = intr_full
    H, W = disp.shape
    valid = disp > disp_eps
    depth = np.where(valid, scale / np.maximum(disp, disp_eps), 0.0)
    u = np.arange(W, dtype=np.float32)[None, :].repeat(H, axis=0)
    v = np.arange(H, dtype=np.float32)[:, None].repeat(W, axis=1)
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    P_cam = np.stack([X, Y, Z, np.ones_like(X)], axis=-1)
    P_world = (T_cw @ P_cam.reshape(-1, 4).T).T.reshape(H, W, 4)
    return P_world[..., :3].astype(np.float32), valid


def pool_dino_in_mask(dino_kf: np.ndarray, mask_full: np.ndarray) -> np.ndarray | None:
    """Mean-pool DINO patch features inside `mask_full` (at mask res),
    returning an L2-normalised 1-D feature, or None if mask is empty
    at DINO resolution.
    dino_kf: (H_d, W_d, C) float
    mask_full: (H_m, W_m) bool
    """
    import cv2 as _cv
    H_d, W_d, _ = dino_kf.shape
    # Resize the binary mask to DINO grid via INTER_AREA (effectively
    # gives fractional coverage; we then threshold > 0.5 to mark voted
    # DINO patches as "inside" the mask). This is paper-aligned with
    # ConceptGraphs's RoI pooling.
    m_small = _cv.resize(mask_full.astype(np.float32), (W_d, H_d),
                         interpolation=_cv.INTER_AREA)
    inside = m_small > 0.5
    if inside.sum() == 0:
        return None
    feats = dino_kf[inside]                   # (P, C)
    feat = feats.mean(axis=0)
    norm = np.linalg.norm(feat) + 1e-9
    return (feat / norm).astype(np.float32)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--refined-npz", required=True, type=str)
    p.add_argument("--video-npz", required=True, type=str)
    p.add_argument("--out", required=True, type=str)
    p.add_argument("--voxel-size", default=0.1, type=float)
    p.add_argument("--sim-threshold", default=1.1, type=float,
                   help="ConceptGraphs default 1.1; total sim is in [0, 2] "
                        "for phys_bias=0; tune for monocular DROID-W noise")
    p.add_argument("--phys-bias", default=0.0, type=float,
                   help="ConceptGraphs: sims = (1+bias)*spatial + (1-bias)*visual. "
                        "0 = equal weight, +1 = spatial only, -1 = visual only.")
    p.add_argument("--subsample", default=2, type=int)
    p.add_argument("--disp-eps", default=0.01, type=float)
    p.add_argument("--depth-clip", default=10.0, type=float)
    p.add_argument("--min-points", default=10, type=int)
    p.add_argument("--enforce-within-kf-distinct", default=1, type=int)
    p.add_argument("--voxel-decay-window", default=0, type=int)
    p.add_argument("--dino-momentum", default=0.5, type=float,
                   help="track's running-mean DINO feature update: "
                        "f_track <- m * f_track + (1-m) * f_new (L2-norm after)")
    args = p.parse_args()

    t0 = time.time()
    refined = np.load(args.refined_npz, allow_pickle=True)
    video = np.load(args.video_npz, allow_pickle=True)

    masks = refined["masks"]
    offsets = refined["seg_kf_offsets"]
    n_kf = int(refined["n_keyframes"])
    H_mask, W_mask = masks.shape[1], masks.shape[2]
    n_masks_total = int(masks.shape[0])
    print(f"[load] {n_masks_total} refined masks across {n_kf} KFs at {H_mask}x{W_mask}",
          flush=True)

    poses = video["poses"]
    droid_up = video["droid_disps_up"]
    intr_ba = video["intrinsics"]
    scale = float(video["scale"])
    dino = video["dino_feats"]                # (n_kf, H_d, W_d, C)
    H_d, W_d, C_d = int(dino.shape[1]), int(dino.shape[2]), int(dino.shape[3])
    print(f"[load] video: droid_disps_up={droid_up.shape}, dino_feats={dino.shape}, "
          f"scale={scale:.4f}", flush=True)

    Hd, Wd = int(droid_up.shape[1]), int(droid_up.shape[2])
    needs_mask_resize = (Hd, Wd) != (H_mask, W_mask)
    if needs_mask_resize:
        print(f"[setup] mask {H_mask}x{W_mask} != depth {Hd}x{Wd}; "
              f"will resize masks for back-projection", flush=True)
        import cv2 as _cv
    intr_factor = float(Hd / 48)
    intrinsics_full = intr_ba * intr_factor
    print(f"[setup] intrinsics x{intr_factor:.0f}; DINO grid {H_d}x{W_d}x{C_d}",
          flush=True)

    s = args.subsample
    vox = args.voxel_size
    inv_vox = 1.0 / vox

    voxel_votes: dict[tuple[int, int, int], Counter[int]] = {}
    voxel_label: dict[tuple[int, int, int], int] = {}
    voxel_last_kf: dict[tuple[int, int, int], int] = {}
    track_feat: dict[int, np.ndarray] = {}     # global_label -> L2-normed mean DINO feat
    track_kf_count: dict[int, int] = {}        # for momentum-style update
    track_last_kf: dict[int, int] = {}         # tracks may also be feature-decayed
    use_decay = args.voxel_decay_window > 0
    use_distinct = args.enforce_within_kf_distinct != 0

    global_track_ids = -np.ones(n_masks_total, dtype=np.int64)
    next_label = 1
    n_new, n_assigned, n_skipped, n_assoc_by_voxel, n_assoc_by_dino = 0, 0, 0, 0, 0

    for k in range(n_kf):
        T_cw = poses[k].astype(np.float32)
        disp = droid_up[k].astype(np.float32)
        intr_full = intrinsics_full[k].astype(np.float32)
        dino_kf = dino[k].astype(np.float32)   # (H_d, W_d, C)

        X_world, valid = backproject_kf(disp, intr_full, T_cw, scale, args.disp_eps)
        z_cam = scale / np.maximum(disp, args.disp_eps)
        valid = valid & (z_cam < args.depth_clip)
        X_world_sub = X_world[::s, ::s, :]
        valid_sub = valid[::s, ::s]
        vox_ijk = np.floor(X_world_sub * inv_vox).astype(np.int32)

        s_off, e_off = int(offsets[k]), int(offsets[k + 1])
        used_labels_this_kf: set[int] = set()
        decay_min_kf = k - args.voxel_decay_window

        for m_idx_in_kf, m_global in enumerate(range(s_off, e_off)):
            mask = masks[m_global]
            if needs_mask_resize:
                mask_for_depth = _cv.resize(mask.astype(np.uint8), (Wd, Hd),
                                            interpolation=_cv.INTER_NEAREST).astype(bool)
            else:
                mask_for_depth = mask
            mask_sub = mask_for_depth[::s, ::s]
            keep = mask_sub & valid_sub
            if keep.sum() < args.min_points:
                n_skipped += 1
                continue
            ijk = vox_ijk[keep]
            P_size = ijk.shape[0]

            # Spatial term: Omega per existing label.
            omega = Counter()
            for row in ijk:
                key = (int(row[0]), int(row[1]), int(row[2]))
                lab = voxel_label.get(key, 0)
                if lab == 0:
                    continue
                if use_decay and voxel_last_kf.get(key, -10**9) < decay_min_kf:
                    continue
                omega[lab] += 1

            # Visual term: pool DINO at mask res.
            f_j = pool_dino_in_mask(dino_kf, mask)   # (C,) or None

            # Combined sim: ConceptGraphs sim_sum.
            # spatial_jk = omega/|P|  in [0,1]
            # visual_jk  = (cos+1)/2  in [0,1]
            # sims       = (1+phys)*spatial + (1-phys)*visual  in [0, 2] at phys=0
            cand_labels = set(omega.keys())
            # Also include any *recent* track for which we have a feature -- a
            # walker who left voxel range can be re-id'd by feature alone.
            for lbl, lst_k in track_last_kf.items():
                if lbl in cand_labels:
                    continue
                if k - lst_k <= 5 and lbl in track_feat:
                    cand_labels.add(lbl)

            best_lab, best_sim, best_spatial, best_visual = 0, -1.0, 0.0, 0.0
            best_kind = ""
            for lbl in cand_labels:
                if use_distinct and lbl in used_labels_this_kf:
                    continue
                spatial = omega.get(lbl, 0) / max(P_size, 1)
                visual = 0.0
                if f_j is not None and lbl in track_feat:
                    cos = float(np.dot(f_j, track_feat[lbl]))
                    visual = 0.5 * (cos + 1.0)
                sim = (1.0 + args.phys_bias) * spatial + (1.0 - args.phys_bias) * visual
                if sim > best_sim:
                    best_sim = sim
                    best_lab = lbl
                    best_spatial, best_visual = spatial, visual
                    best_kind = "voxel" if spatial > visual else "dino"

            if best_sim > args.sim_threshold and best_lab != 0:
                assigned_label = best_lab
                n_assigned += 1
                if best_kind == "voxel":
                    n_assoc_by_voxel += 1
                else:
                    n_assoc_by_dino += 1
            else:
                assigned_label = next_label
                next_label += 1
                n_new += 1
            global_track_ids[m_global] = assigned_label
            used_labels_this_kf.add(assigned_label)

            # Vote + update voxel labels + update track DINO mean.
            for row in ijk:
                key = (int(row[0]), int(row[1]), int(row[2]))
                cnt = voxel_votes.setdefault(key, Counter())
                cnt[assigned_label] += 1
                top = cnt.most_common(1)[0][0]
                voxel_label[key] = top
                voxel_last_kf[key] = k

            if f_j is not None:
                if assigned_label not in track_feat:
                    track_feat[assigned_label] = f_j
                    track_kf_count[assigned_label] = 1
                else:
                    m_mom = args.dino_momentum
                    fused = m_mom * track_feat[assigned_label] + (1.0 - m_mom) * f_j
                    norm = np.linalg.norm(fused) + 1e-9
                    track_feat[assigned_label] = (fused / norm).astype(np.float32)
                    track_kf_count[assigned_label] += 1
            track_last_kf[assigned_label] = k

        if (k + 1) % 10 == 0 or k == n_kf - 1:
            print(f"[kf {k+1:3d}/{n_kf}] tracks={next_label-1}  "
                  f"voxels={len(voxel_votes)}  "
                  f"assigned={n_assigned} (vox={n_assoc_by_voxel} dino={n_assoc_by_dino})  "
                  f"new={n_new}  skipped={n_skipped}  "
                  f"elapsed={time.time()-t0:.1f}s",
                  flush=True)

    track_kf_counts = np.zeros(next_label, dtype=np.int64)
    for k in range(n_kf):
        s_off, e_off = int(offsets[k]), int(offsets[k + 1])
        labs = global_track_ids[s_off:e_off]
        for l in set(labs.tolist()):
            if l > 0:
                track_kf_counts[l] += 1
    n_global_tracks = int((track_kf_counts > 0).sum())

    print(f"\n[done] {next_label-1} raw labels -> {n_global_tracks} global tracks",
          flush=True)
    print(f"[done] assigned={n_assigned}  via_voxel={n_assoc_by_voxel}  "
          f"via_dino_only={n_assoc_by_dino}  new={n_new}  skipped={n_skipped}",
          flush=True)
    for n_at in (1, 2, 3, 5, 10, 20, 50):
        print(f"[hist] tracks visible in >={n_at:3d} KFs: "
              f"{int((track_kf_counts >= n_at).sum())}", flush=True)
    print(f"[done] wall={time.time()-t0:.1f}s", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             schema_version=np.int64(2),
             algorithm=np.array("OVI-MAP_3.1_hybrid_concept-graphs", dtype="U64"),
             n_keyframes=np.int64(n_kf),
             n_masks_total=np.int64(n_masks_total),
             n_global_tracks=np.int64(n_global_tracks),
             global_track_ids=global_track_ids,
             track_kf_counts=track_kf_counts,
             voxel_size=np.float32(args.voxel_size),
             sim_threshold=np.float32(args.sim_threshold),
             phys_bias=np.float32(args.phys_bias),
             dino_momentum=np.float32(args.dino_momentum),
             min_points=np.int64(args.min_points),
             subsample=np.int64(args.subsample),
             enforce_within_kf_distinct=np.int64(args.enforce_within_kf_distinct),
             voxel_decay_window=np.int64(args.voxel_decay_window),
             walltime_seconds=np.float32(time.time() - t0))
    print(f"[save] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
