"""
B.1 v5 — ViPE-exact open-vocab panoptic precompute.

Mirrors RADIO-ViPE arXiv:2604.26067 exactly (per their default config in
`thirdparty/RADIO-ViPE/configs/pipeline/default.yaml`):
  - Proposer: SAM-1 vit_b with `SamAutomaticMaskGenerator` (class-agnostic, no text)
  - Pooling:  ViPE-style mean of features inside the binarized downsampled mask
              (`vipe/priors/embedding/__init__.py:335-380`)
  - PCA:      target_dim=256 over c-radio_v3-b lang-aligned features

Inputs:
    --features  radseg_features_v3b.npz  (from B.1.a, c-radio_v3-b + siglip2)
    --pca-basis weights/pca_basis_v3b.pt (from B.1.b)
    --scene     Outputs/TUM_RGBD/<scene>/   (uses video.npz for RGB stream)
    --sam-ckpt  weights/sam_vit_b_01ec64.pth

Output:
    panoptic_amg.npz  with the v1 schema:
        masks            (K, H, W)  uint8
        lang_emb_pca     (K, 256)   fp16
        score            (K,)       fp16    — predicted_iou
        area             (K,)       int64   — pixel count
        kf_global_idx    (K,)       int64   — per-instance, dataset frame idx
        per_kf_offsets   (N+1,)     int64
        backend          str                — "sam1_vitb_amg_vipe"
        patch_size       int                — derived from radio version
        feature_dim      int                — 1536
        target_dim       int                — 256
        scene            str                — config / scene path
        amg_params       dict-as-json       — recorded for reproducibility
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

REPO_ROOT = Path("/home/cvg/HERMES-SLAM/DROID-W")
sys.path.insert(0, str(REPO_ROOT))


def _containment_merge(proposals: list, containment_thresh: float = 0.70) -> list:
    """Drop part-fragments that are ≥containment_thresh contained in a
    larger parent mask. Sorts by area desc; for each candidate, check
    intersection with already-kept parents — if (inter / cand_area) ≥ thr,
    drop. Mirrors `src/utils/mono_priors/mask_classifier.py:311-345`
    (ConceptGraphs/OVO-SLAM-style consolidation).

    Empirically this collapses the SAM-1 AMG part-fragments (head, torso,
    leg masks of one person) into the largest covering whole-object mask.
    """
    if not proposals:
        return proposals
    sorted_idx = sorted(range(len(proposals)), key=lambda i: -int(proposals[i]["area"]))
    kept = []
    kept_masks = []
    kept_areas = []
    for i in sorted_idx:
        m = proposals[i]["segmentation"].astype(bool)
        a = int(proposals[i]["area"])
        absorbed = False
        for pm, pa in zip(kept_masks, kept_areas):
            if pa <= a:                          # parents must be strictly larger
                continue
            inter = int(np.logical_and(m, pm).sum())
            if inter / max(1, a) >= containment_thresh:
                absorbed = True
                break
        if not absorbed:
            kept.append(proposals[i])
            kept_masks.append(m)
            kept_areas.append(a)
    return kept


def _vipe_pool(features_dhw: torch.Tensor, mask_hw: np.ndarray) -> np.ndarray:
    """ViPE-exact pooling: downsample (H, W) mask to feature grid via
    adaptive_avg_pool2d, threshold > 0.5, then mean of features over the
    thresholded cells. Mirrors `vipe/priors/embedding/__init__.py:_pool_embeddings_by_mask`.

    Args:
        features_dhw: (D, h_f, w_f) torch.Tensor (any dtype)
        mask_hw:      (H, W) bool/uint8 numpy

    Returns:
        (D,) float32 numpy. Returns zero vector (no L2 norm) if mask
        downsamples to all-zeros — caller must handle.
    """
    D, h_f, w_f = features_dhw.shape
    H, W = mask_hw.shape
    # Downsample mask via adaptive_avg_pool2d → fractional coverage per cell.
    m = torch.from_numpy(mask_hw.astype(np.float32))[None, None, :, :]
    m_down = torch.nn.functional.adaptive_avg_pool2d(m, (h_f, w_f))[0, 0]   # (h_f, w_f)
    # ViPE: threshold > 0.5 to get binary cell membership.
    cells_in = (m_down > 0.5)
    n_cells = int(cells_in.sum().item())
    if n_cells == 0:
        return np.zeros(D, dtype=np.float32)
    # Mean over selected cells.
    F = features_dhw.to(torch.float32)                      # (D, h_f, w_f)
    feat_flat = F.reshape(D, -1)                            # (D, h_f*w_f)
    sel = cells_in.reshape(-1)                              # (h_f*w_f,)
    pooled = feat_flat[:, sel].mean(dim=1)                  # (D,)
    return pooled.cpu().numpy().astype(np.float32)


def _encode_pca256(feature_d: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """e_256 = (e_d - mean) @ components.T."""
    return (feature_d.astype(np.float32) - mean) @ components.T


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, type=str,
                   help="radseg_features_v3b.npz from B.1.a")
    p.add_argument("--pca-basis", required=True, type=str,
                   help="pca_basis_v3b.pt from B.1.b")
    p.add_argument("--scene", required=True, type=str,
                   help="Scene dir containing video.npz")
    p.add_argument("--sam-ckpt", default="weights/sam_vit_b_01ec64.pth", type=str)
    p.add_argument("--sam-backbone", default="vit_b", choices=["vit_b", "vit_h"])
    # ViPE-exact AMG hyperparameters (from track_anything/__init__.py:46-69)
    p.add_argument("--amg-points-per-side", default=32, type=int)
    p.add_argument("--amg-pred-iou", default=0.88, type=float)
    p.add_argument("--amg-stability", default=0.95, type=float)
    p.add_argument("--amg-crop-layers", default=1, type=int)
    p.add_argument("--amg-crop-points-downscale", default=2, type=int)
    p.add_argument("--amg-min-area", default=200, type=int)
    p.add_argument("--amg-box-nms", default=0.7, type=float)
    p.add_argument("--containment-thresh", default=0.70, type=float,
                   help="Drop part-fragments that are ≥thresh contained in a "
                        "larger parent mask (ConceptGraphs/OVO-SLAM recipe). "
                        "Set to 1.01 to disable.")
    p.add_argument("--max-kfs", default=None, type=int)
    p.add_argument("--device", default="cuda:0", type=str)
    p.add_argument("--output", required=True, type=str)
    args = p.parse_args()

    feat_path = Path(args.features)
    if not feat_path.is_absolute(): feat_path = REPO_ROOT / feat_path
    basis_path = Path(args.pca_basis)
    if not basis_path.is_absolute(): basis_path = REPO_ROOT / basis_path
    scene_dir = Path(args.scene)
    if not scene_dir.is_absolute(): scene_dir = REPO_ROOT / scene_dir
    sam_ckpt = Path(args.sam_ckpt)
    if not sam_ckpt.is_absolute(): sam_ckpt = REPO_ROOT / sam_ckpt
    out_path = Path(args.output)
    if not out_path.is_absolute(): out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[setup] features={feat_path}\n[setup] basis={basis_path}", flush=True)
    print(f"[setup] scene={scene_dir}\n[setup] sam={sam_ckpt}", flush=True)
    print(f"[setup] out={out_path}", flush=True)

    # --- Load features (Phase B' v3-b) ---
    feat = np.load(feat_path)
    F = feat["lang_aligned_feats"]                          # (N_kf, D, h_f, w_f) fp16
    kf_indices = feat["kf_indices"]                         # (N_kf,) int  — global frame idx
    radio_version = str(feat["radio_version"]) if feat["radio_version"].dtype != object \
                    else str(feat["radio_version"].item())
    is_v4 = "v4" in radio_version.lower()
    patch_size = 14 if is_v4 else 16
    N_kf = F.shape[0]
    D = int(F.shape[1])
    h_f, w_f = int(F.shape[2]), int(F.shape[3])
    print(f"[features] N_kf={N_kf}  D={D}  feat_grid={h_f}x{w_f}  patch={patch_size}  radio={radio_version}",
          flush=True)
    if args.max_kfs is not None:
        N_kf = min(N_kf, args.max_kfs)
        print(f"[smoke] limited to first {N_kf} KFs", flush=True)

    # --- Load PCA basis ---
    state = torch.load(str(basis_path), map_location="cpu", weights_only=False)
    pca_mean = state["mean"].numpy().astype(np.float32)
    pca_components = state["components"].numpy().astype(np.float32)
    target_dim = int(state["target_dim"])
    if int(state["feature_dim"]) != D:
        print(f"[ERR] basis feature_dim {state['feature_dim']} != features D {D}", flush=True)
        return 1
    print(f"[basis] D={D} -> target_dim={target_dim}", flush=True)

    # --- Load images from video.npz ---
    video_path = scene_dir / "video.npz"
    if not video_path.exists():
        print(f"[ERR] {video_path} missing", flush=True)
        return 1
    v = np.load(video_path)
    images = v["images"]                                    # (N, 3, H, W) uint8 OR float
    if images.shape[0] < N_kf:
        print(f"[ERR] video.npz has {images.shape[0]} frames < {N_kf} KFs", flush=True)
        return 1
    sample = images[0]
    if sample.dtype != np.uint8:
        sample = (sample * 255.0).clip(0, 255).astype(np.uint8)
    rgb0 = sample.transpose(1, 2, 0)
    H, W = int(rgb0.shape[0]), int(rgb0.shape[1])
    print(f"[images] {H}x{W}, {images.shape[0]} total frames", flush=True)

    # --- Load SAM-1 + AMG ---
    print(f"[sam] loading {args.sam_backbone} from {sam_ckpt} ...", flush=True)
    t0 = time.time()
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry[args.sam_backbone](checkpoint=str(sam_ckpt))
    sam = sam.to(args.device).eval()
    amg = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.amg_points_per_side,
        pred_iou_thresh=args.amg_pred_iou,
        stability_score_thresh=args.amg_stability,
        crop_n_layers=args.amg_crop_layers,
        crop_n_points_downscale_factor=args.amg_crop_points_downscale,
        min_mask_region_area=args.amg_min_area,
        box_nms_thresh=args.amg_box_nms,
    )
    print(f"[sam] loaded in {time.time()-t0:.1f}s", flush=True)

    # --- Per-KF AMG + pool + PCA ---
    all_masks_kf: list[list[np.ndarray]] = []
    all_lang_pca_kf: list[list[np.ndarray]] = []
    all_score_kf: list[list[float]] = []
    all_area_kf: list[list[int]] = []
    n_total = 0
    t_pipeline = time.time()
    for kf_local in range(N_kf):
        img = images[kf_local]
        if img.dtype != np.uint8:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        rgb = img.transpose(1, 2, 0)
        F_kf = torch.from_numpy(F[kf_local].astype(np.float32))      # (D, h_f, w_f)

        # AMG produces a list of dicts: segmentation, area, bbox, predicted_iou, ...
        proposals = amg.generate(rgb)
        n_amg = len(proposals)
        # Containment dedup — drop part-fragments inside a larger parent.
        if args.containment_thresh < 1.0:
            proposals = _containment_merge(proposals, args.containment_thresh)

        masks_kept = []
        scores_kept = []
        areas_kept = []
        pca_kept = []
        for prop in proposals:
            mask = prop["segmentation"]                              # (H, W) bool
            if mask.shape[0] != H or mask.shape[1] != W:
                # Resize defensively (shouldn't happen with AMG defaults)
                continue
            area = int(mask.sum())
            if area < args.amg_min_area:
                continue
            score = float(prop["predicted_iou"])
            # ViPE-exact pool + PCA encode
            e_d = _vipe_pool(F_kf, mask)
            if not np.any(e_d):
                continue
            e_256 = _encode_pca256(e_d, pca_mean, pca_components).astype(np.float16)
            masks_kept.append(mask.astype(np.uint8))
            scores_kept.append(score)
            areas_kept.append(area)
            pca_kept.append(e_256)

        all_masks_kf.append(masks_kept)
        all_score_kf.append(scores_kept)
        all_area_kf.append(areas_kept)
        all_lang_pca_kf.append(pca_kept)
        n_total += len(masks_kept)

        if (kf_local + 1) % 5 == 0 or kf_local == N_kf - 1:
            el = time.time() - t_pipeline
            eta = el / (kf_local + 1) * (N_kf - kf_local - 1)
            print(f"  [{kf_local+1:3d}/{N_kf}]  AMG={n_amg:3d} -> kept={len(masks_kept):2d} inst  "
                  f"({el:5.1f}s, ~{eta:5.1f}s ETA)", flush=True)

    print(f"\n[done] {n_total} total instances across {N_kf} KFs "
          f"(mean {n_total/max(1,N_kf):.1f}/KF)", flush=True)

    # --- Pack flat arrays + per_kf_offsets + per-instance kf_global_idx ---
    masks_flat = np.zeros((n_total, H, W), dtype=np.uint8)
    lang_emb_pca = np.zeros((n_total, target_dim), dtype=np.float16)
    score_flat = np.zeros((n_total,), dtype=np.float16)
    area_flat = np.zeros((n_total,), dtype=np.int64)
    kf_global_idx_per_inst = np.zeros((n_total,), dtype=np.int64)
    per_kf_offsets = np.zeros((N_kf + 1,), dtype=np.int64)
    cursor = 0
    for k in range(N_kf):
        ms = all_masks_kf[k]
        per_kf_offsets[k + 1] = per_kf_offsets[k] + len(ms)
        gframe = int(kf_indices[k])
        for j, m in enumerate(ms):
            masks_flat[cursor] = m
            lang_emb_pca[cursor] = all_lang_pca_kf[k][j]
            score_flat[cursor] = all_score_kf[k][j]
            area_flat[cursor] = all_area_kf[k][j]
            kf_global_idx_per_inst[cursor] = gframe
            cursor += 1

    amg_params = dict(
        backbone=args.sam_backbone,
        points_per_side=args.amg_points_per_side,
        pred_iou_thresh=args.amg_pred_iou,
        stability_score_thresh=args.amg_stability,
        crop_n_layers=args.amg_crop_layers,
        crop_n_points_downscale_factor=args.amg_crop_points_downscale,
        min_mask_region_area=args.amg_min_area,
        box_nms_thresh=args.amg_box_nms,
    )

    np.savez_compressed(
        out_path,
        masks=masks_flat,
        lang_emb_pca=lang_emb_pca,
        score=score_flat,
        area=area_flat,
        kf_global_idx=kf_global_idx_per_inst,
        per_kf_offsets=per_kf_offsets,
        backend=np.array("sam1_vitb_amg_vipe", dtype=object),
        patch_size=np.int64(patch_size),
        feature_dim=np.int64(D),
        target_dim=np.int64(target_dim),
        scene=np.array(str(scene_dir), dtype=object),
        amg_params=np.array(json.dumps(amg_params), dtype=object),
        radio_version=np.array(radio_version, dtype=object),
    )
    sz = out_path.stat().st_size / 1024 / 1024
    print(f"[save] {out_path}  ({sz:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
