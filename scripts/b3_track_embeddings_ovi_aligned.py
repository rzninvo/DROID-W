"""Plan-v2 §Step 4 / B.3 -- Fix C: OVI-MAP code-faithful alignment.

This file is a NEW companion to b3_track_embeddings.py (which is
paper-TEXT-faithful). After cloning the official OVI-MAP repo
(github.com/OVI-MAP/OVI-MAP, CVPR 2026) to /home/cvg/HERMES-SLAM/
ovi-map-ref and diffing line-by-line, we found 6 divergences between
the paper text and the actual published code. This script ports the
upstream behaviour so our B.3 matches their feature_aggregation.py
+ vl_models.py + test_view_selection.py exactly.

Upstream references (verbatim):
  ovi-map-ref/scripts/feature_aggregation.py
  ovi-map-ref/scripts/vl_models.py:62-106  (VLModel.encode_image_with_bbox)
  ovi-map-ref/scripts/test_view_selection.py:594  (final fusion = np.mean)

Six aligned behaviours vs paper-text b3_track_embeddings.py:

  (1) MULTI-SCALE CROPS: 3 padding levels {0, 0.1, 0.2} * (W, H) of bbox
      x {unmasked, masked-with-black-bg} = 6 crops per view, NOT 2.
      [vl_models.py:75-93]

  (2) PER-CROP L2-NORM BEFORE MEAN: each of the 6 crops L2-normed
      individually, THEN averaged -> single per-view feature.
      [vl_models.py:103-104]

  (3) VIS_AREA_THRES = 1000 (floor 500): skip mask if visible-pixel
      count < 1000 (or per-frame intersection < 500). Filters out
      tiny partial views that produce noisy SigLIP embeddings.
      [feature_aggregation.py:148, 209, 225]

  (4) TOP-10 VISIBILITY RATCHET AT STREAM TIME: even if novelty passes,
      reject if overlap_area <= min(top-10 already-seen vis_areas).
      [feature_aggregation.py:266-271]

  (5) VIEW-OVERLAP RATIO THRESHOLD = 0.9: select if overlap < 0.9
      (equivalently novelty > 0.1), NOT the 0.2 we had.
      [feature_aggregation.py:106-112]

  (6) END-OF-INSTANCE: top-10 by max vis_area, then per-instance
      feature = simple mean (axis=0). No visibility weighting, no
      incremental update.
      [feature_aggregation.py:320-322, test_view_selection.py:594]

Additional alignment:
  - PCA OBB centroid via open3d: voxel_down_sample(0.01),
    remove_statistical_outlier(nb=10, std=2.0), then OBB center.
    [feature_aggregation.py:28-64]

Usage (cvg, droid-w env):
  python scripts/b3_track_embeddings_ovi_aligned.py \\
    --refined-npz Outputs/.../panoptic_v7_cropformer.npz \\
    --tracks-npz  Outputs/.../panoptic_v7_tracks_deva.npz \\
    --video-npz   Outputs/.../video.npz \\
    --rgb-dir     datasets/.../rgb \\
    --out         Outputs/.../panoptic_v7_track_embeddings_ovi.npz \\
    --vis-area-thres 1000 --view-overlap-ratio-thres 0.9 \\
    --max-top-vis 10 --k-expand 0.1 \\
    --bin-phi 30 --bin-theta 60 \\
    --siglip-model google/siglip2-large-patch16-384
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


# --- helpers -------------------------------------------------------------- #

def back_project_pixels(disp: np.ndarray, mask_at_depth: np.ndarray,
                        T_cw: np.ndarray, intr_full: np.ndarray,
                        scale: float, disp_eps: float,
                        subsample: int) -> np.ndarray:
    """Return (P,3) metric world coords for mask pixels with valid depth.

    Uses the same convention as Fix A/B/the old b3 script:
      depth_metric = scale / max(disp, disp_eps)
      P_world = T_cw @ unproject(u, v, depth_metric)
    """
    fx, fy, cx, cy = intr_full
    H, W = disp.shape
    valid = disp > disp_eps
    keep = (mask_at_depth & valid)[::subsample, ::subsample]
    if keep.sum() == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z_full = scale / np.maximum(disp, disp_eps)
    z_sub = z_full[::subsample, ::subsample]
    u = np.arange(W, dtype=np.float32)[None, :].repeat(H, axis=0)[::subsample, ::subsample]
    v = np.arange(H, dtype=np.float32)[:, None].repeat(W, axis=1)[::subsample, ::subsample]
    Xc = (u - cx) * z_sub / fx
    Yc = (v - cy) * z_sub / fy
    Zc = z_sub
    P_cam = np.stack([Xc, Yc, Zc, np.ones_like(Xc)], axis=-1)[keep]
    P_world = (T_cw @ P_cam.T).T
    return P_world[:, :3].astype(np.float32)


def pca_obb_centroid(P_world: np.ndarray, voxel: float = 0.01,
                     nb_neighbors: int = 10, std_thres: float = 2.0
                     ) -> np.ndarray:
    """OVI-MAP feature_aggregation.py:28-64 — voxel-down + outlier-removal +
    OBB centre. Fallback to AABB centre on RuntimeError, then to mean.
    """
    import open3d as o3d
    if P_world.shape[0] < 4:
        return P_world.mean(axis=0) if P_world.shape[0] > 0 else np.zeros(3, np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P_world.astype(np.float64))
    pcd = pcd.voxel_down_sample(voxel)
    if len(pcd.points) >= nb_neighbors + 1:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors, std_thres)
    if len(pcd.points) < 4:
        return np.asarray(pcd.get_center(), dtype=np.float32)
    try:
        obb = pcd.get_oriented_bounding_box()
        return np.asarray(obb.center, dtype=np.float32)
    except RuntimeError:
        aabb = pcd.get_axis_aligned_bounding_box()
        center = 0.5 * (np.asarray(aabb.min_bound) + np.asarray(aabb.max_bound))
        return center.astype(np.float32)


def direction_bins_occ(P_world: np.ndarray, c_k: np.ndarray,
                       H_bin: int, W_bin: int) -> np.ndarray | None:
    """OVI-MAP update_view_cov_map style: spherical bin occupancy of
    viewing directions. Returns (H_bin, W_bin) bool occ, or None for empty."""
    if P_world.shape[0] == 0:
        return None
    d = P_world - c_k[None, :]
    n = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    d = d / n
    theta = np.arccos(np.clip(d[:, 2], -1.0, 1.0))                  # [0, pi]
    phi = np.arctan2(d[:, 1], d[:, 0]) + np.pi                       # [0, 2pi]
    i_theta = np.clip((theta / np.pi * H_bin).astype(np.int32), 0, H_bin - 1)
    i_phi = np.clip((phi / (2 * np.pi) * W_bin).astype(np.int32), 0, W_bin - 1)
    occ = np.zeros((H_bin, W_bin), dtype=bool)
    occ[i_theta, i_phi] = True
    return occ


def encode_view_six_crops(rgb: np.ndarray, mask: np.ndarray,
                           bbox: tuple[int, int, int, int],
                           model, processor, device: str,
                           k_expand: float = 0.1) -> np.ndarray:
    """Reimplements vl_models.py:62-106 verbatim.

    bbox = (x1, y1, x2, y2). For padding layer in {0,1,2}:
      - x_pad = k_expand * layer * (x2 - x1)
      - generate unmasked crop AND masked-black-bg crop at this padding
    Then 6 SigLIP-encoded vectors, per-crop L2-norm, mean -> 1 feature.
    """
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    inst_img = rgb                              # unmasked
    inst_img_black_bg = rgb.copy()
    inst_img_black_bg[~mask] = 0
    crops = []
    for layer in range(3):
        x_pad = int(k_expand * layer * (x2 - x1))
        y_pad = int(k_expand * layer * (y2 - y1))
        x_min = max(0, x1 - x_pad)
        y_min = max(0, y1 - y_pad)
        x_max = min(W - 1, x2 + x_pad)
        y_max = min(H - 1, y2 + y_pad)
        crops.append(Image.fromarray(inst_img[y_min:y_max, x_min:x_max]))
        crops.append(Image.fromarray(inst_img_black_bg[y_min:y_max, x_min:x_max]))

    with torch.no_grad():
        inp = processor(images=crops, return_tensors="pt").to(device)
        out = model.get_image_features(**inp)
        # Normalise transformers' polymorphic return.
        if not isinstance(out, torch.Tensor):
            for attr in ("pooler_output", "image_embeds", "last_hidden_state"):
                if hasattr(out, attr) and getattr(out, attr) is not None:
                    out = getattr(out, attr)
                    break
        if out.ndim == 3:
            out = out.mean(dim=1)
        # OVI-MAP per-crop L2-norm THEN mean (vl_models.py:103-104):
        out = out / (out.norm(dim=-1, keepdim=True) + 1e-9)
        feat = out.mean(dim=0)
    return feat.detach().cpu().float().numpy()


# --- main ----------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--refined-npz", required=True, type=str)
    p.add_argument("--tracks-npz", required=True, type=str)
    p.add_argument("--video-npz", required=True, type=str)
    p.add_argument("--rgb-dir", required=True, type=str)
    p.add_argument("--out", required=True, type=str)

    # OVI-MAP defaults from feature_aggregation.py
    p.add_argument("--vis-area-thres", default=1000, type=int)
    p.add_argument("--view-overlap-ratio-thres", default=0.9, type=float)
    p.add_argument("--max-top-vis", default=10, type=int)
    p.add_argument("--k-expand", default=0.1, type=float)
    p.add_argument("--bin-phi", default=30, type=int,
                   help="spherical theta-grid; paper 180, indoor 30 ok")
    p.add_argument("--bin-theta", default=60, type=int,
                   help="spherical phi-grid; paper 240, indoor 60 ok")

    p.add_argument("--disp-eps", default=0.01, type=float)
    p.add_argument("--depth-clip", default=10.0, type=float)
    p.add_argument("--subsample", default=4, type=int)
    p.add_argument("--siglip-model", default="google/siglip2-large-patch16-384",
                   type=str)
    args = p.parse_args()

    FORMAT = '%(asctime)s.%(msecs)06d %(levelname)-8s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%H:%M:%S')

    t0 = time.time()
    refined = np.load(args.refined_npz, allow_pickle=True)
    tracks = np.load(args.tracks_npz, allow_pickle=True)
    video = np.load(args.video_npz, allow_pickle=True)

    masks = refined["masks"]
    offsets = refined["seg_kf_offsets"]
    kf_gi = refined["kf_global_indices"]
    n_kf = int(refined["n_keyframes"])
    H_m, W_m = int(masks.shape[1]), int(masks.shape[2])
    gids = tracks["global_track_ids"]
    n_global_tracks = int(tracks["n_global_tracks"])
    poses = video["poses"]
    droid_up = video["droid_disps_up"]
    intr_ba = video["intrinsics"]
    scale = float(video["scale"])
    Hd, Wd = int(droid_up.shape[1]), int(droid_up.shape[2])
    intrinsics_full = intr_ba * (Hd / 48.0)

    rgb_dir = Path(args.rgb_dir)
    rgb_files = sorted(f for f in rgb_dir.iterdir() if f.suffix == ".png")

    logging.info(f"refined: {masks.shape[0]} masks, {n_kf} KFs, {H_m}x{W_m}")
    logging.info(f"tracks: {n_global_tracks} global IDs")

    # Group masks by global track.
    track_views: dict[int, list[tuple[int, int]]] = {}
    for k_kf in range(n_kf):
        s_off, e_off = int(offsets[k_kf]), int(offsets[k_kf + 1])
        for m_global in range(s_off, e_off):
            gid = int(gids[m_global])
            if gid <= 0:
                continue
            track_views.setdefault(gid, []).append((k_kf, m_global))
    logging.info(f"{len(track_views)} tracks with >=1 (KF, mask)")

    # PASS 1: per-track 3D centroid via PCA OBB.
    logging.info("pass1: PCA OBB centroids...")
    t1 = time.time()
    track_centroid: dict[int, np.ndarray] = {}
    for gid, views in track_views.items():
        P_all = []
        for (k_kf, m_global) in views:
            mask = masks[m_global]
            disp = droid_up[k_kf]
            if (Hd, Wd) != (H_m, W_m):
                m_at_depth = cv2.resize(mask.astype(np.uint8), (Wd, Hd),
                                         interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                m_at_depth = mask
            T_cw = poses[k_kf]
            intr = intrinsics_full[k_kf]
            P = back_project_pixels(disp, m_at_depth, T_cw, intr,
                                     scale, args.disp_eps, args.subsample)
            if P.shape[0] > 0:
                # depth clip filter
                depth_cam = np.linalg.norm(P - T_cw[:3, 3], axis=1)
                P = P[depth_cam < args.depth_clip * 3.0]  # generous
                if P.shape[0] > 0:
                    P_all.append(P)
        if not P_all:
            track_centroid[gid] = np.zeros(3, np.float32)
            continue
        P_all = np.concatenate(P_all, axis=0)
        track_centroid[gid] = pca_obb_centroid(P_all)
    logging.info(f"pass1 done: {len(track_centroid)} centroids in {time.time()-t1:.1f}s")

    # PASS 2: view selection with vis_area + view-overlap + top-10 ratchet.
    logging.info("pass2: view selection (3-stage gate)...")
    t2 = time.time()
    # selected: gid -> list of dicts {kf, m_global, vis_area, occ}
    selected: dict[int, list[dict]] = {gid: [] for gid in track_views}
    cov_per_track: dict[int, np.ndarray] = {}

    for gid, views in track_views.items():
        c_k = track_centroid[gid]
        cov = np.zeros((args.bin_phi, args.bin_theta), dtype=bool)
        for (k_kf, m_global) in views:
            mask = masks[m_global]
            vis_area = int(mask.sum())

            # OVI-MAP feature_aggregation.py:209,225: vis_area_thres filter.
            if vis_area < args.vis_area_thres:
                continue

            # Back-project for novelty computation.
            disp = droid_up[k_kf]
            if (Hd, Wd) != (H_m, W_m):
                m_at_depth = cv2.resize(mask.astype(np.uint8), (Wd, Hd),
                                         interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                m_at_depth = mask
            T_cw = poses[k_kf]
            intr = intrinsics_full[k_kf]
            P = back_project_pixels(disp, m_at_depth, T_cw, intr,
                                     scale, args.disp_eps, args.subsample)
            view_occ = direction_bins_occ(P, c_k, args.bin_phi, args.bin_theta)
            if view_occ is None or view_occ.sum() == 0:
                continue

            # OVI-MAP update_view_cov_map: select if overlap_ratio <= thres.
            overlap = int((view_occ & cov).sum())
            view_area = int(view_occ.sum())
            overlap_ratio = overlap / max(view_area, 1)
            if overlap_ratio > args.view_overlap_ratio_thres:
                continue

            # OVI-MAP feature_aggregation.py:266-271:
            # top-10 visibility ratchet at stream time.
            past_vis = [s["vis_area"] for s in selected[gid]]
            if len(past_vis) >= args.max_top_vis:
                top_vis = np.sort(np.array(past_vis))[-args.max_top_vis:]
                if vis_area <= int(top_vis.min()):
                    continue

            # accept
            cov |= view_occ
            selected[gid].append(dict(kf=k_kf, m=m_global,
                                       vis_area=vis_area))
        cov_per_track[gid] = cov

    n_total = sum(len(v) for v in selected.values())
    avg_per = n_total / max(len(selected), 1)
    logging.info(f"pass2 done: {n_total} selected (avg {avg_per:.1f}/track)  "
                 f"in {time.time()-t2:.1f}s")
    counts = np.array([len(v) for v in selected.values()])
    for at in (1, 2, 5, 10, 20):
        logging.info(f"hist  >= {at:2d} sel views: {int((counts >= at).sum())} tracks")

    # PASS 3: SigLIP-2-large 6-crop encoding per view.
    logging.info(f"pass3: loading {args.siglip_model}...")
    from transformers import AutoModel, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.siglip_model).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.siglip_model)
    # Probe dim.
    with torch.no_grad():
        dummy = Image.new("RGB", (32, 32), (128, 128, 128))
        inp = processor(images=dummy, return_tensors="pt").to(device)
        probe = model.get_image_features(**inp)
        if not isinstance(probe, torch.Tensor):
            for attr in ("pooler_output", "image_embeds", "last_hidden_state"):
                if hasattr(probe, attr) and getattr(probe, attr) is not None:
                    probe = getattr(probe, attr)
                    break
        if probe.ndim == 3:
            probe = probe.mean(dim=1)
        emb_dim = int(probe.shape[-1])
    logging.info(f"pass3: dim={emb_dim}, device={device}")

    # Encode every (gid, selected_view) -> 1 feat via 6-crop mean.
    track_feats: dict[int, list[np.ndarray]] = {gid: [] for gid in track_views}
    track_vis: dict[int, list[int]] = {gid: [] for gid in track_views}

    rgb_cache: dict[int, np.ndarray] = {}
    n_encoded = 0
    t3 = time.time()
    for gid, views_list in selected.items():
        for s in views_list:
            k_kf, m_global, vis_area = s["kf"], s["m"], s["vis_area"]
            mask = masks[m_global]
            if k_kf not in rgb_cache:
                gi = int(kf_gi[k_kf])
                bgr = cv2.imread(str(rgb_files[gi]))
                if bgr is None or bgr.shape[:2] != (H_m, W_m):
                    if bgr is None:
                        logging.warning(f"kf {k_kf}: no RGB at {rgb_files[gi]}")
                        continue
                    bgr = cv2.resize(bgr, (W_m, H_m), interpolation=cv2.INTER_AREA)
                rgb_cache[k_kf] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = rgb_cache[k_kf]
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            feat = encode_view_six_crops(rgb, mask, (x1, y1, x2, y2),
                                          model, processor, device,
                                          args.k_expand)
            track_feats[gid].append(feat)
            track_vis[gid].append(vis_area)
            n_encoded += 1
            if n_encoded % 100 == 0:
                logging.info(f"encode {n_encoded}  elapsed={time.time()-t3:.1f}s")
    logging.info(f"pass3 done: {n_encoded} views encoded "
                 f"({n_encoded * 6} crops) in {time.time()-t3:.1f}s")

    # PASS 4: final per-instance feature = mean of top-K by vis_area
    #         [feature_aggregation.py:320-322, test_view_selection.py:594].
    logging.info("pass4: top-K-vis selection + mean fusion...")
    n_views_total = np.zeros(n_global_tracks + 1, dtype=np.int64)
    n_views_selected = np.zeros(n_global_tracks + 1, dtype=np.int64)
    n_views_final = np.zeros(n_global_tracks + 1, dtype=np.int64)
    track_emb = np.zeros((n_global_tracks + 1, emb_dim), dtype=np.float32)
    for gid in track_views:
        n_views_total[gid] = len(track_views[gid])
        n_views_selected[gid] = len(selected[gid])
        feats = track_feats[gid]
        vis = track_vis[gid]
        if len(feats) == 0:
            continue
        feats_arr = np.array(feats)              # (k, D)
        vis_arr = np.array(vis)
        if feats_arr.shape[0] > args.max_top_vis:
            top_idx = np.argsort(vis_arr)[-args.max_top_vis:]
            feats_arr = feats_arr[top_idx]
        n_views_final[gid] = feats_arr.shape[0]
        # Simple mean across top-k views (OVI-MAP test_view_selection.py:594).
        track_emb[gid] = feats_arr.mean(axis=0)

    # Also L2-norm at the very end so cosine queries are stable.
    norms = np.linalg.norm(track_emb, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    track_emb_l2 = track_emb / norms

    n_nonzero = int((np.linalg.norm(track_emb, axis=1) > 0).sum())
    logging.info(f"done: {n_nonzero}/{n_global_tracks} tracks with non-zero embedding")
    logging.info(f"total wall = {time.time()-t0:.1f}s")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             schema_version=np.int64(3),
             algorithm=np.array("OVI-MAP_3.2_aligned_siglip2L_6crop_top10",
                                dtype="U64"),
             n_global_tracks=np.int64(n_global_tracks),
             embedding_dim=np.int64(emb_dim),
             embeddings=track_emb,                # un-L2-normed
             embeddings_l2=track_emb_l2,
             n_views_total=n_views_total,
             n_views_selected=n_views_selected,
             n_views_final=n_views_final,
             vis_area_thres=np.int64(args.vis_area_thres),
             view_overlap_ratio_thres=np.float32(args.view_overlap_ratio_thres),
             max_top_vis=np.int64(args.max_top_vis),
             k_expand=np.float32(args.k_expand),
             bin_phi=np.int64(args.bin_phi),
             bin_theta=np.int64(args.bin_theta),
             siglip_model=np.array(args.siglip_model, dtype="U64"),
             walltime_seconds=np.float32(time.time() - t0))
    logging.info(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
