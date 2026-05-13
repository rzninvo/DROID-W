"""Plan-v2 §Step 4 / B.3 -- Path C: slab-filter + OVI-MAP pano∪glo union.

Layered on top of Fix C (b3_track_embeddings_ovi_aligned.py). Two
upstream-grounded modifications motivated by Agent C's diagnostic on
real data + Agent A's deep read of OVI-MAP's actual pipeline:

  Change 1 -- SLAB-MASK REJECTION at encoding time:
    Drop any (track, view) where the obj_mask area >= 20000 px AND its
    bbox spans >= 480 px in width. This excludes CropFormer slab masks
    that cover monitor+wall+desk together (confirmed wrong tracks 20,
    27, 30, 38, 50, 83 in Report 33 figures). SigLIP cannot label a
    heterogeneous slab; the right move is to keep it out of the map.

  Change 2 -- OVI-MAP PANO ∪ GLO UNION mask for SigLIP encoding:
    Upstream OVI-MAP feature_aggregation.py:293 uses
        obj_mask = np.logical_or(pano_mask, glo_inst_mask)
    where pano_mask is the per-frame CropFormer entity mask and
    glo_inst_mask is the 3D-derived global-instance mask. For HERMES-
    SLAM, the glo_inst_mask analogue is the DEVA-propagated track region
    in this KF (where the per-pixel track ID in DEVA's output PNG equals
    our DEVA track ID). The union fills in pixels CropFormer missed
    (textureless regions, occluded boundaries) that the temporal
    propagator still believes belong to the object.

The rest of the script is identical to b3_track_embeddings_ovi_aligned
(LERF-style 6-crop encoding, top-10-by-vis fusion, PCA-OBB centroid,
spherical view-novelty selection). Both the OVI-MAP-faithful and the
Path-C variants exist side-by-side for ablation.

Upstream code references (verified file:line):
  feature_aggregation.py:213-219  (pano_mask via argmax pano_id)
  feature_aggregation.py:293     (obj_mask = pano OR glo_inst_mask)
  feature_aggregation.py:106-112 (view-overlap-ratio gate)
  feature_aggregation.py:266-271 (top-10 visibility ratchet at stream time)
  feature_aggregation.py:320-322 + test_view_selection.py:594 (final mean)
  vl_models.py:62-106            (6-crop multi-scale encoder)

Usage (cvg, droid-w env):
  python scripts/b3_track_embeddings_pathC.py \\
    --refined-npz   Outputs/.../panoptic_v7_cropformer.npz \\
    --tracks-npz    Outputs/.../panoptic_v7_tracks_deva.npz \\
    --video-npz     Outputs/.../video.npz \\
    --rgb-dir       datasets/.../rgb \\
    --deva-png-dir  DEVA_runs/walking_static/output/Annotations/walking_static \\
    --out           Outputs/.../panoptic_v7_track_embeddings_pathC.npz \\
    --slab-area 20000 --slab-bbox-w 480
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


def back_project_pixels(disp, mask_at_depth, T_cw, intr_full,
                        scale, disp_eps, subsample):
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


def pca_obb_centroid(P_world, voxel=0.01, nb=10, std=2.0):
    import open3d as o3d
    if P_world.shape[0] < 4:
        return (P_world.mean(axis=0).astype(np.float32) if P_world.shape[0] > 0
                else np.zeros(3, np.float32))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P_world.astype(np.float64))
    pcd = pcd.voxel_down_sample(voxel)
    if len(pcd.points) >= nb + 1:
        pcd, _ = pcd.remove_statistical_outlier(nb, std)
    if len(pcd.points) < 4:
        return np.asarray(pcd.get_center(), dtype=np.float32)
    try:
        return np.asarray(pcd.get_oriented_bounding_box().center, np.float32)
    except RuntimeError:
        a = pcd.get_axis_aligned_bounding_box()
        return ((np.asarray(a.min_bound) + np.asarray(a.max_bound)) / 2).astype(np.float32)


def direction_bins(P_world, c_k, H_bin, W_bin):
    if P_world.shape[0] == 0:
        return None
    d = P_world - c_k[None, :]
    n = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    d = d / n
    theta = np.arccos(np.clip(d[:, 2], -1.0, 1.0))
    phi = np.arctan2(d[:, 1], d[:, 0]) + np.pi
    i_t = np.clip((theta / np.pi * H_bin).astype(np.int32), 0, H_bin - 1)
    i_p = np.clip((phi / (2 * np.pi) * W_bin).astype(np.int32), 0, W_bin - 1)
    occ = np.zeros((H_bin, W_bin), dtype=bool)
    occ[i_t, i_p] = True
    return occ


def encode_six_crops(rgb, mask, bbox, model, processor, device, k_expand):
    """vl_models.py:62-106 — 3 padding layers × {unmasked, masked-black-bg}."""
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    inst_img = rgb
    inst_img_blackbg = rgb.copy()
    inst_img_blackbg[~mask] = 0
    crops = []
    for layer in range(3):
        x_pad = int(k_expand * layer * (x2 - x1))
        y_pad = int(k_expand * layer * (y2 - y1))
        xmn, ymn = max(0, x1 - x_pad), max(0, y1 - y_pad)
        xmx, ymx = min(W - 1, x2 + x_pad), min(H - 1, y2 + y_pad)
        crops.append(Image.fromarray(inst_img[ymn:ymx, xmn:xmx]))
        crops.append(Image.fromarray(inst_img_blackbg[ymn:ymx, xmn:xmx]))
    with torch.no_grad():
        inp = processor(images=crops, return_tensors="pt").to(device)
        out = model.get_image_features(**inp)
        if not isinstance(out, torch.Tensor):
            for a in ("pooler_output", "image_embeds", "last_hidden_state"):
                if hasattr(out, a) and getattr(out, a) is not None:
                    out = getattr(out, a)
                    break
        if out.ndim == 3:
            out = out.mean(dim=1)
        out = out / (out.norm(dim=-1, keepdim=True) + 1e-9)
        feat = out.mean(dim=0)
    return feat.detach().cpu().float().numpy()


def decode_deva_png(p: Path) -> np.ndarray:
    """RGB PNG -> int64 track-id map: id = R + G*256 + B*65536."""
    im = np.array(Image.open(p))
    if im.ndim == 2:
        return im.astype(np.int64)
    return (im[..., 0].astype(np.int64)
            + im[..., 1].astype(np.int64) * 256
            + im[..., 2].astype(np.int64) * 65536)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--refined-npz", required=True, type=str)
    p.add_argument("--tracks-npz", required=True, type=str)
    p.add_argument("--video-npz", required=True, type=str)
    p.add_argument("--rgb-dir", required=True, type=str)
    p.add_argument("--deva-png-dir", required=True, type=str,
                   help="DEVA output Annotations/<seq>/ (per-KF RGB-packed track IDs)")
    p.add_argument("--out", required=True, type=str)

    # OVI-MAP / Fix C defaults
    p.add_argument("--vis-area-thres", default=1000, type=int)
    p.add_argument("--view-overlap-ratio-thres", default=0.9, type=float)
    p.add_argument("--max-top-vis", default=10, type=int)
    p.add_argument("--k-expand", default=0.1, type=float)
    p.add_argument("--bin-phi", default=30, type=int)
    p.add_argument("--bin-theta", default=60, type=int)
    p.add_argument("--disp-eps", default=0.01, type=float)
    p.add_argument("--depth-clip", default=10.0, type=float)
    p.add_argument("--subsample", default=4, type=int)
    p.add_argument("--siglip-model", default="google/siglip2-large-patch16-384",
                   type=str)

    # Path-C additions
    p.add_argument("--slab-area", default=20000, type=int,
                   help="Path-C slab filter: reject obj_mask with area >= this (px)")
    p.add_argument("--slab-bbox-w", default=480, type=int,
                   help="AND bbox width >= this (out of image width 640) to be a slab")
    p.add_argument("--use-pano-glo-union", default=1, type=int,
                   help="1: obj_mask = pano OR glo (OVI-MAP feature_aggregation.py:293); "
                        "0: obj_mask = pano only (Fix C behaviour)")
    args = p.parse_args()

    FORMAT = '%(asctime)s.%(msecs)06d %(levelname)-8s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%H:%M:%S')

    t0 = time.time()
    refined = np.load(args.refined_npz, allow_pickle=True)
    tracks = np.load(args.tracks_npz, allow_pickle=True)
    video = np.load(args.video_npz, allow_pickle=True)

    masks = refined["masks"]                                  # (N, H, W) bool
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
    deva_dir = Path(args.deva_png_dir)

    logging.info(f"refined: {masks.shape[0]} masks, {n_kf} KFs, {H_m}x{W_m}")
    logging.info(f"tracks: {n_global_tracks} global IDs")
    logging.info(f"slab filter: area>={args.slab_area} AND bbox_w>={args.slab_bbox_w}")
    logging.info(f"pano-glo union: {bool(args.use_pano_glo_union)}")

    # ---- DEVA-ID -> our compact track ID mapping ----
    # We need the inverse of deva_to_panoptic_v7_tracks.py's argmax.
    # For each KF, decode the DEVA PNG and build {raw_deva_id: our_track_id}.
    # We do this by argmax: for each raw_deva_id in this KF, which of our
    # tracks has the most overlap with that DEVA region? Take argmax.
    logging.info("building DEVA-PNG-id -> our-track-id map per KF...")
    t1 = time.time()
    # First: for each KF, get the set of (track, mask) we know about.
    # Then: load DEVA PNG, for each raw_id in PNG, find which CropFormer mask
    # in this KF has maximum overlap with the raw region, look up its track id.
    # This is exactly the inverse of deva_to_panoptic_v7_tracks.py.
    raw_to_track_per_kf: list[dict[int, int]] = [{} for _ in range(n_kf)]
    for k_kf in range(n_kf):
        gi = int(kf_gi[k_kf])
        stem = rgb_files[gi].name[:-4]
        deva_png = deva_dir / f"{stem}.png"
        if not deva_png.exists():
            continue
        T = decode_deva_png(deva_png)                            # (H, W)
        raw_ids = np.unique(T)
        raw_ids = raw_ids[raw_ids > 0]
        # Get all CropFormer masks for this KF + their track IDs.
        s_off, e_off = int(offsets[k_kf]), int(offsets[k_kf + 1])
        for raw_id in raw_ids:
            region = (T == int(raw_id))
            # which of OUR tracks has max overlap with this raw_id's region?
            best_overlap, best_track = 0, 0
            for m_global in range(s_off, e_off):
                track_id = int(gids[m_global])
                if track_id <= 0:
                    continue
                overlap = int((masks[m_global] & region).sum())
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_track = track_id
            if best_track > 0 and best_overlap >= 5:
                raw_to_track_per_kf[k_kf][int(raw_id)] = best_track
    logging.info(f"DEVA-map built in {time.time()-t1:.1f}s")

    # ---- Group masks by global track ----
    track_views: dict[int, list[tuple[int, int]]] = {}
    for k_kf in range(n_kf):
        s_off, e_off = int(offsets[k_kf]), int(offsets[k_kf + 1])
        for m_global in range(s_off, e_off):
            gid = int(gids[m_global])
            if gid <= 0:
                continue
            track_views.setdefault(gid, []).append((k_kf, m_global))

    # ---- PASS 1: PCA-OBB centroids per track ----
    logging.info("pass1: PCA-OBB centroids...")
    track_centroid = {}
    for gid, views in track_views.items():
        P_all = []
        for (k_kf, m_global) in views:
            mask = masks[m_global]
            disp = droid_up[k_kf]
            m_at_d = (cv2.resize(mask.astype(np.uint8), (Wd, Hd),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
                      if (Hd, Wd) != (H_m, W_m) else mask)
            P = back_project_pixels(disp, m_at_d, poses[k_kf],
                                     intrinsics_full[k_kf], scale,
                                     args.disp_eps, args.subsample)
            if P.shape[0] > 0:
                P_all.append(P)
        track_centroid[gid] = (pca_obb_centroid(np.concatenate(P_all, axis=0))
                                if P_all else np.zeros(3, np.float32))

    # ---- PASS 2: view selection (3-stage gate) + SLAB FILTER (Path C) ----
    logging.info("pass2: view selection + slab filter...")
    selected: dict[int, list[dict]] = {gid: [] for gid in track_views}
    n_slab_skipped = 0
    for gid, views in track_views.items():
        c_k = track_centroid[gid]
        cov = np.zeros((args.bin_phi, args.bin_theta), dtype=bool)
        for (k_kf, m_global) in views:
            pano_mask = masks[m_global]
            vis_area = int(pano_mask.sum())
            if vis_area < args.vis_area_thres:
                continue

            # Construct obj_mask per OVI-MAP feature_aggregation.py:293
            if args.use_pano_glo_union:
                # Find DEVA's region for this track in this KF.
                gi = int(kf_gi[k_kf])
                stem = rgb_files[gi].name[:-4]
                deva_png = deva_dir / f"{stem}.png"
                if deva_png.exists():
                    T_png = decode_deva_png(deva_png)
                    # Find raw_id(s) that map to this track in this KF
                    glo_mask = np.zeros((H_m, W_m), dtype=bool)
                    for raw_id, mapped in raw_to_track_per_kf[k_kf].items():
                        if mapped == gid:
                            glo_mask |= (T_png == raw_id)
                    obj_mask = pano_mask | glo_mask
                else:
                    obj_mask = pano_mask
            else:
                obj_mask = pano_mask

            # PATH-C SLAB FILTER: skip slab-shaped obj_masks.
            ys, xs = np.where(obj_mask)
            if len(xs) == 0:
                continue
            bbox_w = int(xs.max() - xs.min() + 1)
            obj_area = int(obj_mask.sum())
            if obj_area >= args.slab_area and bbox_w >= args.slab_bbox_w:
                n_slab_skipped += 1
                logging.info(f"[slab-filter] track {gid:3d} KF{k_kf:03d} "
                              f"area={obj_area} bbox_w={bbox_w} -> SKIP")
                continue

            # 3D back-project for novelty
            disp = droid_up[k_kf]
            m_at_d = (cv2.resize(obj_mask.astype(np.uint8), (Wd, Hd),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
                      if (Hd, Wd) != (H_m, W_m) else obj_mask)
            P = back_project_pixels(disp, m_at_d, poses[k_kf],
                                     intrinsics_full[k_kf], scale,
                                     args.disp_eps, args.subsample)
            view_occ = direction_bins(P, c_k, args.bin_phi, args.bin_theta)
            if view_occ is None or view_occ.sum() == 0:
                continue
            overlap_ratio = int((view_occ & cov).sum()) / max(int(view_occ.sum()), 1)
            if overlap_ratio > args.view_overlap_ratio_thres:
                continue
            # top-10 visibility ratchet (stream-time)
            past = [s["vis_area"] for s in selected[gid]]
            if len(past) >= args.max_top_vis:
                if vis_area <= int(np.sort(np.array(past))[-args.max_top_vis:].min()):
                    continue
            cov |= view_occ
            selected[gid].append(dict(kf=k_kf, m=m_global, vis_area=vis_area,
                                      obj_mask=obj_mask))
    n_total = sum(len(v) for v in selected.values())
    logging.info(f"pass2: selected {n_total} views; slab-filter skipped {n_slab_skipped}")

    # ---- PASS 3: 6-crop encode each selected view ----
    logging.info(f"pass3: loading {args.siglip_model}...")
    from transformers import AutoModel, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.siglip_model).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.siglip_model)
    # Probe dim
    with torch.no_grad():
        dummy = Image.new("RGB", (32, 32), (128, 128, 128))
        inp = processor(images=dummy, return_tensors="pt").to(device)
        probe = model.get_image_features(**inp)
        if not isinstance(probe, torch.Tensor):
            for a in ("pooler_output", "image_embeds", "last_hidden_state"):
                if hasattr(probe, a) and getattr(probe, a) is not None:
                    probe = getattr(probe, a)
                    break
        if probe.ndim == 3:
            probe = probe.mean(dim=1)
        emb_dim = int(probe.shape[-1])
    logging.info(f"pass3: dim={emb_dim}")

    track_feats: dict[int, list[np.ndarray]] = {gid: [] for gid in track_views}
    track_vis: dict[int, list[int]] = {gid: [] for gid in track_views}
    rgb_cache: dict[int, np.ndarray] = {}
    n_enc = 0
    t3 = time.time()
    for gid, views_list in selected.items():
        for s in views_list:
            k_kf = s["kf"]
            obj_mask = s["obj_mask"]
            if k_kf not in rgb_cache:
                gi = int(kf_gi[k_kf])
                bgr = cv2.imread(str(rgb_files[gi]))
                if bgr is None or bgr.shape[:2] != (H_m, W_m):
                    continue
                rgb_cache[k_kf] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = rgb_cache[k_kf]
            ys, xs = np.where(obj_mask)
            if len(ys) == 0:
                continue
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            feat = encode_six_crops(rgb, obj_mask, (x1, y1, x2, y2),
                                     model, processor, device, args.k_expand)
            track_feats[gid].append(feat)
            track_vis[gid].append(s["vis_area"])
            n_enc += 1
            if n_enc % 100 == 0:
                logging.info(f"encode {n_enc} ... elapsed={time.time()-t3:.1f}s")
    logging.info(f"pass3 done: {n_enc} views encoded in {time.time()-t3:.1f}s")

    # ---- PASS 4: top-K visibility + mean fusion (test_view_selection.py:594) ----
    n_views_total = np.zeros(n_global_tracks + 1, dtype=np.int64)
    n_views_selected = np.zeros(n_global_tracks + 1, dtype=np.int64)
    n_views_final = np.zeros(n_global_tracks + 1, dtype=np.int64)
    track_emb = np.zeros((n_global_tracks + 1, emb_dim), dtype=np.float32)
    for gid in track_views:
        n_views_total[gid] = len(track_views[gid])
        n_views_selected[gid] = len(selected[gid])
        feats = track_feats[gid]
        vis = track_vis[gid]
        if not feats:
            continue
        feats_arr = np.array(feats)
        vis_arr = np.array(vis)
        if feats_arr.shape[0] > args.max_top_vis:
            top = np.argsort(vis_arr)[-args.max_top_vis:]
            feats_arr = feats_arr[top]
        n_views_final[gid] = feats_arr.shape[0]
        track_emb[gid] = feats_arr.mean(axis=0)

    norms = np.linalg.norm(track_emb, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    track_emb_l2 = track_emb / norms
    n_nonzero = int((np.linalg.norm(track_emb, axis=1) > 0).sum())
    logging.info(f"DONE: {n_nonzero}/{n_global_tracks} tracks have non-zero embedding "
                  f"({n_slab_skipped} slab views skipped)")
    logging.info(f"total wall = {time.time()-t0:.1f}s")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             schema_version=np.int64(4),
             algorithm=np.array("PathC_slabfilter_panoglo_siglip2L_6crop_top10",
                                dtype="U64"),
             n_global_tracks=np.int64(n_global_tracks),
             embedding_dim=np.int64(emb_dim),
             embeddings=track_emb,
             embeddings_l2=track_emb_l2,
             n_views_total=n_views_total,
             n_views_selected=n_views_selected,
             n_views_final=n_views_final,
             slab_area=np.int64(args.slab_area),
             slab_bbox_w=np.int64(args.slab_bbox_w),
             use_pano_glo_union=np.int64(args.use_pano_glo_union),
             n_slab_skipped=np.int64(n_slab_skipped),
             vis_area_thres=np.int64(args.vis_area_thres),
             view_overlap_ratio_thres=np.float32(args.view_overlap_ratio_thres),
             max_top_vis=np.int64(args.max_top_vis),
             k_expand=np.float32(args.k_expand),
             siglip_model=np.array(args.siglip_model, dtype="U64"),
             walltime_seconds=np.float32(time.time() - t0))
    logging.info(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
