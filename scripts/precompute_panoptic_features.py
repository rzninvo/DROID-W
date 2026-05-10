"""
B.1 — per-keyframe panoptic precompute (FastSAM → SAM-3 refine).

For every Phase B' keyframe of a scene, produce a list of per-instance
tuples: (mask, lang_emb_pca256, score, area, kf_global_idx). Every mask
is SAM-3-sharp (FastSAM is only used as a class-agnostic *bbox proposer*;
SAM-3 always refines via box-prompt to give the final silhouette).

Pipeline per KF:
  1. Load image (from video.npz) + Phase B' feature F_i (from radseg_features.npz).
  2. FastSAM `segment_everything` → list of bboxes (class-agnostic).
  3. For each bbox passing area / score gates: SAM-3 with box-prompt → sharp mask.
  4. Dedupe: drop pairs with mask_iou > 0.92 (keeps highest-score).
  5. For each surviving mask:
       e_1536 = pool_lang_features_in_mask(F_i, mask, patch_size)        # (1536,)
       e_256  = (e_1536 - pca_mean) @ pca_components.T                   # (256,)
  6. Save per-KF tuple to a single `panoptic.npz` for the scene.

Output schema (.npz):
  masks:           (K_total, H, W)  uint8   — instance silhouettes.
  lang_emb_pca:    (K_total, 256)   fp16    — PCA-compressed pooled lang feature.
  score:           (K_total,)       fp16    — SAM-3 score.
  area:            (K_total,)       int64   — pixel count of mask.
  kf_global_idx:   (K_total,)       int64   — which Phase B' KF this came from.
  per_kf_offsets:  (N_kf+1,)        int64   — (start, end] indices per KF.
  backend:         str                       — "fastsam_sam3_refine".
  patch_size:      int                       — RADIO patch size used.
  feature_dim:     int                       — D = 1536 (pre-PCA).
  target_dim:      int                       — 256 (post-PCA).
  scene:           str                       — config scene name.

Usage:
    python scripts/precompute_panoptic_features.py \\
        --config configs/RGBD/Replica/room0.yaml \\
        --features Outputs/Replica/room0/radseg_features.npz \\
        --pca-basis weights/pca_basis.pt \\
        --output Outputs/Replica/room0/panoptic.npz \\
        --max-kfs 5      # smoke-test mode

(For full Plan B run: omit --max-kfs.)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from PIL import Image

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.mono_priors.panoptic_pooler import (
    PoolResult, encode_pca256, pool_lang_features_in_mask,
)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Bool-mask IoU."""
    if not a.any() or not b.any():
        return 0.0
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def _dedupe(masks: list[np.ndarray], scores: list[float],
            iou_thresh: float = 0.92) -> list[int]:
    """Greedy dedupe: keep highest-score; drop any later mask with IoU ≥ thresh.
    Returns indices to keep."""
    order = np.argsort(scores)[::-1]                            # high score first
    kept: list[int] = []
    for idx in order:
        keep = True
        for k in kept:
            if _mask_iou(masks[idx], masks[k]) >= iou_thresh:
                keep = False
                break
        if keep:
            kept.append(int(idx))
    return sorted(kept)


def _load_pca_basis(path: Path):
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    mean = state["mean"].numpy().astype(np.float32)             # (D,)
    components = state["components"].numpy().astype(np.float32)  # (target_dim, D)
    return mean, components, int(state["feature_dim"]), int(state["target_dim"])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str,
                   help="Scene config (e.g. configs/RGBD/Replica/room0.yaml)")
    p.add_argument("--features", required=True, type=str,
                   help="Path to radseg_features.npz for this scene.")
    p.add_argument("--pca-basis", default="weights/pca_basis.pt", type=str)
    p.add_argument("--output", default=None, type=str,
                   help="Default: <features.parent>/panoptic.npz")
    p.add_argument("--max-kfs", default=None, type=int,
                   help="Cap on KFs (debug / smoke-test).")
    p.add_argument("--device", default="cuda:0", type=str)
    p.add_argument("--fastsam-conf", default=0.4, type=float)
    p.add_argument("--fastsam-iou", default=0.9, type=float)
    p.add_argument("--fastsam-imgsz", default=1024, type=int)
    p.add_argument("--sam3-score-thresh", default=0.5, type=float)
    p.add_argument("--min-area", default=200, type=int)
    p.add_argument("--max-instances-per-kf", default=80, type=int,
                   help="Hard cap; FastSAM occasionally produces >100 tiny masks.")
    p.add_argument("--dedupe-iou", default=0.92, type=float)
    args = p.parse_args()

    feat_path = Path(args.features)
    if not feat_path.is_absolute():
        feat_path = REPO_ROOT / feat_path
    out_path = Path(args.output) if args.output else feat_path.parent / "panoptic.npz"
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pca_path = Path(args.pca_basis)
    if not pca_path.is_absolute():
        pca_path = REPO_ROOT / pca_path

    print(f"[setup] config={args.config}", flush=True)
    print(f"[setup] features={feat_path}", flush=True)
    print(f"[setup] pca_basis={pca_path}", flush=True)
    print(f"[setup] output={out_path}", flush=True)

    # ── Load features ──────────────────────────────────────────────────────
    m = np.load(feat_path)
    feats = m["lang_aligned_feats"]                            # (N, D, h, w) fp16
    N_kf, D, h, w = feats.shape
    image_hw = tuple(int(x) for x in m["image_hw"])
    H_img, W_img = image_hw
    radio_version = str(m["radio_version"])
    is_v4 = "v4" in radio_version.lower()
    patch_size = 14 if is_v4 else 16
    H_expect = h * patch_size
    W_expect = w * patch_size
    if abs(H_expect - H_img) > patch_size or abs(W_expect - W_img) > patch_size:
        print(f"[WARN] panoptic: feat-grid {(h, w)} × patch {patch_size} = {(H_expect, W_expect)} "
              f"vs image {(H_img, W_img)} — within tolerance", flush=True)
    print(f"[features] N_kf={N_kf}  D={D}  feat_grid=({h},{w})  image=({H_img},{W_img})  "
          f"patch_size={patch_size}  radio={radio_version}", flush=True)

    n_run = N_kf if args.max_kfs is None else min(args.max_kfs, N_kf)
    print(f"[features] running B.1 on {n_run} KFs (max_kfs={args.max_kfs})", flush=True)

    # ── Load images ────────────────────────────────────────────────────────
    video_path = feat_path.parent / "video.npz"
    if not video_path.exists():
        # Some scenes (Replica, ScanNet) don't ship a video.npz — load from
        # dataset stream instead.
        from src import config as droid_config
        from src.utils.datasets import get_dataset
        cfg = droid_config.load_config(args.config)
        stream = get_dataset(cfg)
        kf_global_indices = m["kf_indices"] if "kf_indices" in m.files else np.arange(N_kf)
        def _get_image(kf_idx: int) -> np.ndarray:
            global_idx = int(kf_global_indices[kf_idx])
            _, color, _, _ = stream[global_idx]
            if torch.is_tensor(color):
                color = color.cpu().numpy()
            # color is (3, H, W) float [0,1] — convert to (H, W, 3) uint8.
            if color.ndim == 4:
                color = color[0]
            color = np.transpose(color, (1, 2, 0))
            color = (color * 255.0).clip(0, 255).astype(np.uint8)
            return color
        print(f"[features] using dataset stream (video.npz missing)", flush=True)
    else:
        v = np.load(video_path)
        images = v["images"]                                   # (N, 3, H, W) uint8 or float
        if images.dtype != np.uint8:
            images = (images * 255.0).clip(0, 255).astype(np.uint8)
        kf_global_indices = m["kf_indices"] if "kf_indices" in m.files else np.arange(N_kf)
        def _get_image(kf_idx: int) -> np.ndarray:
            return images[kf_idx].transpose(1, 2, 0)
        print(f"[features] using video.npz", flush=True)

    # ── Load FastSAM + SAM-3 ───────────────────────────────────────────────
    print(f"[setup] loading FastSAM ...", flush=True)
    from src.utils.mono_priors.fastsam_segmentor import (
        get_fastsam_model, segment_everything,
    )
    fast_model = get_fastsam_model(model_name="FastSAM-x.pt", device=args.device)

    print(f"[setup] loading SAM-3 ...", flush=True)
    from transformers import Sam3Model, Sam3Processor
    sam3_proc = Sam3Processor.from_pretrained("facebook/sam3")
    sam3_model = Sam3Model.from_pretrained("facebook/sam3").to(args.device).eval()

    # ── Load PCA basis ─────────────────────────────────────────────────────
    pca_mean, pca_components, D_basis, target_dim = _load_pca_basis(pca_path)
    if D_basis != D:
        raise RuntimeError(f"PCA basis D={D_basis} != feature D={D}")
    print(f"[setup] PCA basis D={D_basis} → target_dim={target_dim}", flush=True)

    # ── Loop ───────────────────────────────────────────────────────────────
    all_masks: list[np.ndarray] = []
    all_lang_emb: list[np.ndarray] = []
    all_scores: list[float] = []
    all_areas: list[int] = []
    all_kf: list[int] = []
    per_kf_offsets = [0]

    t_loop_start = time.time()
    for kf in range(n_run):
        rgb = _get_image(kf)
        Hk, Wk = rgb.shape[:2]
        F_kf_np = feats[kf]                                    # (D, h, w) fp16
        F_kf = torch.from_numpy(F_kf_np).to(args.device)

        # FastSAM proposals (class-agnostic everything-mode).
        det = segment_everything(
            fast_model, rgb, device=args.device,
            imgsz=args.fastsam_imgsz,
            conf=args.fastsam_conf,
            iou=args.fastsam_iou,
        )
        # Filter by min_area on FastSAM mask.
        boxes = []
        for d in det:
            if d.get("area", 0) < args.min_area:
                continue
            box = d["box"]
            # Clip + sanity check.
            x1, y1, x2, y2 = (max(0, float(box[0])), max(0, float(box[1])),
                              min(Wk - 1, float(box[2])), min(Hk - 1, float(box[3])))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            boxes.append([x1, y1, x2, y2])
        # Cap to max-instances per KF.
        boxes = boxes[: args.max_instances_per_kf]

        # SAM-3 box-prompted refinement, BATCHED (all boxes in one forward).
        # ~0.4s/box × 60 boxes = 24s/KF unbatched. Batched: ~3s/KF (8x speedup).
        # Sam3Processor accepts `input_boxes=[[box1, box2, ...]]` as a list of
        # boxes for one image — single forward returns K masks.
        kf_masks: list[np.ndarray] = []
        kf_scores: list[float] = []
        if boxes:
            pil = Image.fromarray(rgb)
            inputs = sam3_proc(
                images=pil,
                input_boxes=[boxes],     # one image, K boxes
                return_tensors="pt",
            ).to(args.device)
            with torch.no_grad():
                out = sam3_model(**inputs)
            res = sam3_proc.post_process_instance_segmentation(
                out, threshold=args.sam3_score_thresh,
                target_sizes=[(Hk, Wk)],
            )[0]
            seg = res.get("masks") if "masks" in res else res.get("segmentation")
            scores = res.get("scores")
            if seg is not None:
                if torch.is_tensor(seg):
                    seg = seg.cpu().numpy()
                if scores is not None and torch.is_tensor(scores):
                    scores = scores.cpu().numpy()
                # Expect (K, H, W) one-mask-per-box. Some HF versions return
                # a labeled map (H, W) instead — handle both.
                if seg.ndim == 3:
                    n_out = seg.shape[0]
                    for k in range(n_out):
                        m_arr = (seg[k] > 0.5).astype(bool)
                        if int(m_arr.sum()) < args.min_area:
                            continue
                        sc = float(scores[k]) if scores is not None and len(scores) > k else float(args.sam3_score_thresh)
                        kf_masks.append(m_arr)
                        kf_scores.append(sc)
                elif seg.ndim == 2:
                    for lab in np.unique(seg):
                        if lab <= 0:
                            continue
                        m_arr = (seg == lab).astype(bool)
                        if int(m_arr.sum()) < args.min_area:
                            continue
                        kf_masks.append(m_arr)
                        kf_scores.append(float(args.sam3_score_thresh))

        # Dedupe.
        kept = _dedupe(kf_masks, kf_scores, iou_thresh=args.dedupe_iou)
        kf_masks = [kf_masks[i] for i in kept]
        kf_scores = [kf_scores[i] for i in kept]

        # Pool features per surviving mask + PCA-encode.
        kf_lang_emb_pca = []
        kf_areas = []
        kept_masks = []
        kept_scores = []
        for m_arr, sc in zip(kf_masks, kf_scores):
            res = pool_lang_features_in_mask(F_kf, m_arr, patch_size=patch_size)
            if res.is_empty:
                continue
            e_pca = encode_pca256(res.feature, pca_mean, pca_components)   # (256,)
            kept_masks.append(m_arr)
            kept_scores.append(sc)
            kf_lang_emb_pca.append(e_pca.astype(np.float16))
            kf_areas.append(int(m_arr.sum()))

        # Append to scene-wide buffers.
        for m_arr, sc, e_pca, a in zip(kept_masks, kept_scores, kf_lang_emb_pca, kf_areas):
            all_masks.append(m_arr.astype(np.uint8))
            all_lang_emb.append(e_pca)
            all_scores.append(np.float16(sc))
            all_areas.append(a)
            all_kf.append(int(kf))
        per_kf_offsets.append(len(all_masks))

        if (kf + 1) % 5 == 0 or kf == 0 or kf == n_run - 1:
            el = time.time() - t_loop_start
            eta = el / max(1, kf + 1) * (n_run - kf - 1)
            print(f"  [{kf+1:3d}/{n_run}]  K_kf={len(kept_masks):3d}  K_total={len(all_masks):4d}  "
                  f"({el:5.1f}s elapsed, ~{eta:5.1f}s ETA)", flush=True)

        del F_kf
        if (kf + 1) % 10 == 0:
            torch.cuda.empty_cache()

    # ── Save ───────────────────────────────────────────────────────────────
    if not all_masks:
        print(f"[FAIL] B.1: no instances produced over {n_run} KFs", flush=True)
        return 1

    # masks → fixed-shape (K, H, W) uint8.
    masks_arr = np.stack(all_masks, axis=0)
    lang_arr = np.stack(all_lang_emb, axis=0)
    np.savez(
        out_path,
        masks=masks_arr,
        lang_emb_pca=lang_arr,
        score=np.asarray(all_scores, dtype=np.float16),
        area=np.asarray(all_areas, dtype=np.int64),
        kf_global_idx=np.asarray(all_kf, dtype=np.int64),
        per_kf_offsets=np.asarray(per_kf_offsets, dtype=np.int64),
        backend=np.array("fastsam_sam3_refine"),
        patch_size=np.int64(patch_size),
        feature_dim=np.int64(D),
        target_dim=np.int64(target_dim),
        scene=np.array(str(args.config)),
        image_hw=np.asarray([H_img, W_img], dtype=np.int64),
    )
    size_mb = out_path.stat().st_size / 1024 / 1024
    n_kfs_actual = len(per_kf_offsets) - 1
    walltime = time.time() - t_loop_start
    print(f"\n[done] {len(all_masks)} instances over {n_kfs_actual} KFs in {walltime:.1f}s "
          f"(mean {len(all_masks)/n_kfs_actual:.1f} inst/KF)", flush=True)
    print(f"[save] {out_path}  ({size_mb:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
