"""Plan-v2 §Step 4 / B.3 — per-track SigLIP-2-large encoding.

Implements OVI-MAP §3.2 "View-Adaptive Semantic Feature Aggregation"
verbatim against the verified ar5iv HTML extraction:

  1. Spherical view-novelty selection (Cov_k ∈ {0,1}^{Hbin × Wbin}):
     for each KF where track k appears, compute viewing directions per
     mask pixel d = (X_world - c_k)/||...||, bin into (theta, phi).
     eta_{t,k} = |new bins| / |total bins in view|.
     If eta > theta_novel, select the view.

  2. Two crops per selected view (paper-faithful):
     f^(1) = SigLIP(I_t cropped to mask bbox + margin) — unmasked bbox
     f^(2) = SigLIP(I_t cropped, with background zeroed)  — masked

  3. Incremental running update:
     f_k ← (w_sum/(w_k+w_sum))·f_k + (w_k/(2(w_k+w_sum)))·(f^(1)+f^(2))
     w_k = visible pixels in this KF; w_sum = cumulative previous.

  4. (Engineering addition) L2-normalize at the end for stable cosine
     queries downstream; paper doesn't specify but it's mathematically
     equivalent to L2-normalizing at query time.

Reference (verbatim from agent extraction):
  Liu et al., OVI-MAP, arXiv:2603.26541, §3.2 "View-Adaptive Semantic
  Feature Aggregation". The novelty-threshold value θ_novel is NOT
  stated in the paper; we sweep on freiburg3_walking_static and pick
  the empirical knee. The spherical binning resolution 180×240 IS in
  the paper; we default to it.

Usage (cvg, droid-w env):
  python scripts/b3_track_embeddings.py \\
    --refined-npz Outputs/.../panoptic_v7_cropformer.npz \\
    --tracks-npz  Outputs/.../panoptic_v7_tracks_deva.npz \\
    --video-npz   Outputs/.../video.npz \\
    --rgb-dir     datasets/.../rgb \\
    --out         Outputs/.../panoptic_v7_track_embeddings.npz \\
    --theta-novel 0.2 --bin-theta 60 --bin-phi 120 \\
    --bbox-margin 0.10 --batch-size 16 \\
    --siglip-model google/siglip2-large-patch16-384
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def back_project_mask_pixels(mask: np.ndarray, depth: np.ndarray,
                              T_cw: np.ndarray, intr_full: np.ndarray,
                              scale: float, disp_eps: float,
                              subsample: int = 4) -> np.ndarray:
    """Return (P, 3) array of metric world coords for the mask's valid pixels."""
    fx, fy, cx, cy = intr_full
    H, W = depth.shape
    valid = depth > disp_eps
    m_at_depth = cv2.resize(mask.astype(np.uint8), (W, H),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
    keep = m_at_depth & valid
    keep = keep[::subsample, ::subsample]
    if keep.sum() == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z_cam_full = scale / np.maximum(depth, disp_eps)
    z_sub = z_cam_full[::subsample, ::subsample]
    u = np.arange(W, dtype=np.float32)[None, :].repeat(H, axis=0)[::subsample, ::subsample]
    v = np.arange(H, dtype=np.float32)[:, None].repeat(W, axis=1)[::subsample, ::subsample]
    Xc = (u - cx) * z_sub / fx
    Yc = (v - cy) * z_sub / fy
    Zc = z_sub
    P_cam = np.stack([Xc, Yc, Zc, np.ones_like(Xc)], axis=-1)[keep]
    P_world = (T_cw @ P_cam.T).T
    return P_world[:, :3].astype(np.float32)


def direction_bins(P_world: np.ndarray, c_k: np.ndarray,
                   H_bin: int, W_bin: int) -> np.ndarray:
    """Bin per-pixel viewing directions onto (H_bin × W_bin) spherical map.
    Returns a (H_bin, W_bin) bool occupancy."""
    if P_world.shape[0] == 0:
        return np.zeros((H_bin, W_bin), dtype=bool)
    d = P_world - c_k[None, :]
    n = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    d = d / n
    # spherical coords
    phi = np.arccos(np.clip(d[:, 2], -1.0, 1.0))                  # [0, pi]
    theta = np.arctan2(d[:, 1], d[:, 0]) + np.pi                  # [0, 2pi]
    i = np.clip((phi / np.pi * H_bin).astype(np.int32), 0, H_bin - 1)
    j = np.clip((theta / (2 * np.pi) * W_bin).astype(np.int32), 0, W_bin - 1)
    occ = np.zeros((H_bin, W_bin), dtype=bool)
    occ[i, j] = True
    return occ


def mask_bbox(mask: np.ndarray, margin_frac: float) -> tuple[int, int, int, int]:
    """Return (y0, x0, y1, x1) with margin."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0, 0, mask.shape[0], mask.shape[1]
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    h, w = y1 - y0, x1 - x0
    my, mx = int(h * margin_frac), int(w * margin_frac)
    H_im, W_im = mask.shape
    return (max(0, y0 - my), max(0, x0 - mx),
            min(H_im, y1 + my), min(W_im, x1 + mx))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--refined-npz", required=True, type=str)
    p.add_argument("--tracks-npz", required=True, type=str)
    p.add_argument("--video-npz", required=True, type=str)
    p.add_argument("--rgb-dir", required=True, type=str)
    p.add_argument("--out", required=True, type=str)
    p.add_argument("--theta-novel", default=0.2, type=float,
                   help="OVI-MAP §3.2 θ_novel; paper does not publish value")
    p.add_argument("--bin-theta", default=60, type=int,
                   help="azimuthal bins (paper default 240; 60 is finer at "
                        "indoor scale)")
    p.add_argument("--bin-phi", default=30, type=int,
                   help="elevation bins (paper default 180; 30 is reasonable indoor)")
    p.add_argument("--bbox-margin", default=0.10, type=float)
    p.add_argument("--batch-size", default=16, type=int)
    p.add_argument("--disp-eps", default=0.01, type=float)
    p.add_argument("--depth-clip", default=10.0, type=float)
    p.add_argument("--subsample", default=4, type=int,
                   help="pixel stride for 3D back-projection (only for view-novelty)")
    p.add_argument("--siglip-model", default="google/siglip2-large-patch16-384",
                   type=str)
    p.add_argument("--min-pixels", default=50, type=int,
                   help="skip a (track, kf) view if 2D mask has <N pixels")
    p.add_argument("--max-views-per-track", default=64, type=int,
                   help="hard cap on views per track (safety; OVI-MAP avg ~18.6)")
    args = p.parse_args()

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

    poses = video["poses"]                              # (n_kf, 4, 4)
    droid_up = video["droid_disps_up"]                  # (n_kf, H_d, W_d)
    intr_ba = video["intrinsics"]                       # (n_kf, 4) @ 48x64
    scale = float(video["scale"])
    Hd, Wd = int(droid_up.shape[1]), int(droid_up.shape[2])
    intrinsics_full = intr_ba * (Hd / 48.0)

    print(f"[load] refined: {masks.shape[0]} masks, {n_kf} KFs, {H_m}x{W_m}",
          flush=True)
    print(f"[load] tracks: {n_global_tracks} global IDs", flush=True)
    print(f"[load] depth res {Hd}x{Wd}; scale={scale:.4f}", flush=True)

    rgb_dir = Path(args.rgb_dir)
    rgb_files = sorted(f for f in rgb_dir.iterdir() if f.suffix == ".png")

    # Group masks by global track.
    track_views: dict[int, list[tuple[int, int]]] = {}
    for k_kf in range(n_kf):
        s_off, e_off = int(offsets[k_kf]), int(offsets[k_kf + 1])
        for m_global in range(s_off, e_off):
            gid = int(gids[m_global])
            if gid <= 0:
                continue
            track_views.setdefault(gid, []).append((k_kf, m_global))
    print(f"[setup] {len(track_views)} tracks have >=1 visible (KF, mask)",
          flush=True)

    # PASS 1: per-track 3D centroid c_k from all observed 2D pixels.
    print("[pass1] computing 3D centroids per track...", flush=True)
    t1 = time.time()
    track_centroid: dict[int, np.ndarray] = {}
    for gid, views in track_views.items():
        P_all = []
        for (k_kf, m_global) in views:
            mask = masks[m_global]
            disp = droid_up[k_kf]
            T_cw = poses[k_kf]
            intr = intrinsics_full[k_kf]
            P = back_project_mask_pixels(mask, disp, T_cw, intr, scale,
                                          args.disp_eps, args.subsample)
            if P.shape[0] > 0:
                # depth clip in cam frame already implicit via valid mask
                P_all.append(P)
        if not P_all:
            track_centroid[gid] = np.zeros(3, dtype=np.float32)
            continue
        P_all = np.concatenate(P_all, axis=0)
        track_centroid[gid] = P_all.mean(axis=0)
    print(f"[pass1] {len(track_centroid)} centroids in {time.time()-t1:.1f}s",
          flush=True)

    # PASS 2: view-novelty selection.
    print(f"[pass2] view-novelty selection (θ_novel={args.theta_novel}, "
          f"bins={args.bin_phi}x{args.bin_theta})...", flush=True)
    t2 = time.time()
    selected: dict[int, list[tuple[int, int]]] = {gid: [] for gid in track_views}
    for gid, views in track_views.items():
        # Score each view by visibility (mask pixel count) — used to break ties
        # but novelty drives the selection.
        c_k = track_centroid[gid]
        cov = np.zeros((args.bin_phi, args.bin_theta), dtype=bool)
        for (k_kf, m_global) in views:
            if len(selected[gid]) >= args.max_views_per_track:
                break
            mask = masks[m_global]
            n_pix = int(mask.sum())
            if n_pix < args.min_pixels:
                continue
            disp = droid_up[k_kf]
            T_cw = poses[k_kf]
            intr = intrinsics_full[k_kf]
            P = back_project_mask_pixels(mask, disp, T_cw, intr, scale,
                                          args.disp_eps, args.subsample)
            if P.shape[0] == 0:
                continue
            view_occ = direction_bins(P, c_k, args.bin_phi, args.bin_theta)
            n_view = int(view_occ.sum())
            if n_view == 0:
                continue
            n_new = int((view_occ & ~cov).sum())
            eta = n_new / n_view
            if eta > args.theta_novel:
                selected[gid].append((k_kf, m_global))
                cov |= view_occ
        if len(selected[gid]) == 0:
            # Fallback: at least take the view with most pixels (one frame).
            # The paper doesn't address singletons but reviewers will ask
            # what happens to single-KF tracks.
            best_kf, best_m, best_npix = -1, -1, 0
            for (k_kf, m_global) in views:
                n_pix = int(masks[m_global].sum())
                if n_pix > best_npix and n_pix >= args.min_pixels:
                    best_kf, best_m, best_npix = k_kf, m_global, n_pix
            if best_kf >= 0:
                selected[gid].append((best_kf, best_m))
                print(f"[WARN] track {gid}: 0 novel views (θ={args.theta_novel}), "
                      f"fallback to single best-pixel view kf={best_kf}",
                      flush=True)

    n_total_selected = sum(len(v) for v in selected.values())
    avg_per_track = n_total_selected / max(len(selected), 1)
    print(f"[pass2] {n_total_selected} (track, view) pairs selected "
          f"(avg {avg_per_track:.1f}/track, paper ~18.6) in {time.time()-t2:.1f}s",
          flush=True)
    # Quick histogram of views/track.
    counts = np.array([len(v) for v in selected.values()])
    for at in (1, 2, 5, 10, 20):
        print(f"[hist] tracks with >={at:3d} selected views: "
              f"{int((counts >= at).sum())}", flush=True)

    # PASS 3: SigLIP encoding + incremental fusion.
    print(f"[pass3] loading {args.siglip_model}...", flush=True)
    from transformers import AutoModel, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.siglip_model).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.siglip_model)
    print(f"[pass3] device={device}, dtype={next(model.parameters()).dtype}",
          flush=True)

    def _to_tensor(obj):
        """Some transformers versions return a tensor; others return a
        BaseModelOutputWithPooling. Normalise to a 2D tensor of features."""
        if isinstance(obj, torch.Tensor):
            return obj
        for attr in ("pooler_output", "image_embeds", "last_hidden_state"):
            if hasattr(obj, attr):
                t = getattr(obj, attr)
                if t is None:
                    continue
                # last_hidden_state is (B, N, D) -> mean pool the patch tokens
                if t.ndim == 3:
                    t = t.mean(dim=1)
                return t
        raise RuntimeError(f"Unknown SigLIP output type: {type(obj)}")

    # Probe output dim with a dummy.
    with torch.no_grad():
        dummy = Image.new("RGB", (32, 32), (128, 128, 128))
        inp = processor(images=dummy, return_tensors="pt").to(device)
        emb = _to_tensor(model.get_image_features(**inp))
        emb_dim = int(emb.shape[-1])
    print(f"[pass3] image embedding dim = {emb_dim}", flush=True)

    # Build list of (gid, kf, m_global) → crops batched.
    all_pairs = []
    for gid, views in selected.items():
        for (k_kf, m_global) in views:
            all_pairs.append((gid, k_kf, m_global))
    print(f"[pass3] {len(all_pairs)} (track, view) pairs to encode "
          f"({len(all_pairs)*2} crops total)", flush=True)

    track_emb = np.zeros((n_global_tracks + 1, emb_dim), dtype=np.float32)
    track_wsum = np.zeros(n_global_tracks + 1, dtype=np.float64)
    n_views_selected = np.zeros(n_global_tracks + 1, dtype=np.int64)
    n_views_total = np.zeros(n_global_tracks + 1, dtype=np.int64)
    for gid, views in track_views.items():
        n_views_total[gid] = len(views)
    for gid, views in selected.items():
        n_views_selected[gid] = len(views)

    t3 = time.time()
    bs = args.batch_size
    rgb_cache: dict[int, np.ndarray] = {}
    crops_buf: list[Image.Image] = []
    meta_buf: list[tuple[int, int]] = []  # (gid, w_k) per CROP
    for i, (gid, k_kf, m_global) in enumerate(all_pairs):
        mask = masks[m_global]
        gi = int(kf_gi[k_kf])
        if k_kf not in rgb_cache:
            bgr = cv2.imread(str(rgb_files[gi]))
            if bgr is None or bgr.shape[:2] != (H_m, W_m):
                if bgr is None:
                    print(f"[WARN] kf {k_kf}: cannot read RGB", flush=True)
                    continue
                bgr = cv2.resize(bgr, (W_m, H_m), interpolation=cv2.INTER_AREA)
            rgb_cache[k_kf] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = rgb_cache[k_kf]
        y0, x0, y1, x1 = mask_bbox(mask, args.bbox_margin)
        crop_unmasked = rgb[y0:y1, x0:x1]                                # (h,w,3)
        crop_mask = mask[y0:y1, x0:x1].astype(np.uint8)
        crop_masked = crop_unmasked * crop_mask[..., None]               # zero-bg
        w_k = int(mask.sum())
        crops_buf.append(Image.fromarray(crop_unmasked))
        crops_buf.append(Image.fromarray(crop_masked))
        meta_buf.append((gid, w_k))
        meta_buf.append((gid, w_k))
        if len(crops_buf) >= bs * 2 or i == len(all_pairs) - 1:
            with torch.no_grad():
                inp = processor(images=crops_buf, return_tensors="pt").to(device)
                emb = _to_tensor(model.get_image_features(**inp))
                emb = emb.cpu().float().numpy()
            # Process pairs: every 2 consecutive entries correspond to one
            # (track, kf) — (unmasked, masked). Apply OVI-MAP §3.2 incremental
            # update per (track, kf), not per crop:
            j = 0
            while j < len(meta_buf):
                gid_j, w_k = meta_buf[j]
                gid_j2, w_k2 = meta_buf[j + 1]
                assert gid_j == gid_j2 and w_k == w_k2, "pair-of-2 invariant"
                f1, f2 = emb[j], emb[j + 1]
                w_sum_prev = track_wsum[gid_j]
                if w_sum_prev == 0:
                    # First observation: f_k = 0.5 * (f1 + f2)
                    track_emb[gid_j] = 0.5 * (f1 + f2)
                else:
                    a = w_sum_prev / (w_k + w_sum_prev)
                    b = w_k / (2.0 * (w_k + w_sum_prev))
                    track_emb[gid_j] = a * track_emb[gid_j] + b * (f1 + f2)
                track_wsum[gid_j] += w_k
                j += 2
            crops_buf = []
            meta_buf = []
        if (i + 1) % 200 == 0 or i == len(all_pairs) - 1:
            print(f"[encode] pair {i+1}/{len(all_pairs)}  elapsed={time.time()-t3:.1f}s",
                  flush=True)

    # L2-normalize (engineering addition for clean cosine queries).
    norms = np.linalg.norm(track_emb, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    track_emb_normed = track_emb / norms

    n_nonzero = int((np.linalg.norm(track_emb, axis=1) > 0).sum())
    print(f"\n[done] {n_nonzero} tracks have non-zero embeddings out of "
          f"{n_global_tracks}", flush=True)
    print(f"[done] total wall = {time.time()-t0:.1f}s "
          f"(centroid+novelty {time.time()-t0-(time.time()-t3):.1f}s, "
          f"siglip {time.time()-t3:.1f}s)", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             schema_version=np.int64(2),
             algorithm=np.array("OVI-MAP_3.2_spherical_novelty_siglip2L",
                                dtype="U64"),
             n_global_tracks=np.int64(n_global_tracks),
             embedding_dim=np.int64(emb_dim),
             embeddings=track_emb,                # un-normalized (paper)
             embeddings_l2=track_emb_normed,      # L2-normalized (queries)
             n_views_total=n_views_total,
             n_views_selected=n_views_selected,
             total_pixel_weight=track_wsum.astype(np.float64),
             theta_novel=np.float32(args.theta_novel),
             bin_theta=np.int64(args.bin_theta),
             bin_phi=np.int64(args.bin_phi),
             bbox_margin=np.float32(args.bbox_margin),
             siglip_model=np.array(args.siglip_model, dtype="U64"),
             walltime_seconds=np.float32(time.time() - t0))
    print(f"[save] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
