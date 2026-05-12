"""
Plan-v2 §Step 4 / B.1 smoke test — Mask2Former (entity-style) replacement for
SAM-1 AMG on freiburg3_walking_static KF 9 (the visible-two-people KF from
Reports 19 and 0.b). Headline question: does Mask2Former produce <=2 person
masks (we want one per visible person) vs SAM-1 AMG's 5-8 fragments?

Uses HuggingFace `facebook/mask2former-swin-large-coco-panoptic` (Apache,
plan-v2 §Step 4 fallback path when CropFormer's Detectron2 install isn't
available).

Produces a triptych PNG:
    RGB | per-instance random-colour overlay | person-only red overlay

Usage (cvg, droid-w env):
    python tests/smoke_cropformer_kf9.py --kf-index 9
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")

import numpy as np
import torch
from PIL import Image


def _coco_id_to_label(model_cfg):
    # facebook/mask2former-* exposes id2label on the config.
    return {int(k): v for k, v in model_cfg.id2label.items()}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rgb-dir", default="datasets/TUM_RGBD/rgbd_dataset_freiburg3_walking_static/rgb",
                   type=str)
    p.add_argument("--kf-index", default=9, type=int,
                   help="dataset frame index (matches Step 0.b)")
    p.add_argument("--model", default="facebook/mask2former-swin-large-coco-panoptic",
                   type=str)
    p.add_argument("--out", default="Outputs/TUM_RGBD/freiburg3_walking_static/step4_b1_kf9_mask2former.png",
                   type=str)
    p.add_argument("--device", default="cuda:0", type=str)
    p.add_argument("--score-thresh", default=0.5, type=float,
                   help="discard masks with score < threshold")
    args = p.parse_args()

    rgb_dir = Path(args.rgb_dir)
    if not rgb_dir.is_absolute():
        rgb_dir = Path.cwd() / rgb_dir
    rgb_files = sorted(f for f in rgb_dir.iterdir() if f.suffix == ".png")
    rgb_path = rgb_files[args.kf_index]
    print(f"[setup] KF {args.kf_index} -> {rgb_path.name}", flush=True)
    image = Image.open(rgb_path).convert("RGB")
    H_img, W_img = image.height, image.width
    print(f"[setup] image {W_img}x{H_img}", flush=True)

    from transformers import (
        Mask2FormerForUniversalSegmentation,
        Mask2FormerImageProcessor,
    )
    print(f"[load] {args.model}", flush=True)
    processor = Mask2FormerImageProcessor.from_pretrained(args.model)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model).to(args.device).eval()

    id2label = _coco_id_to_label(model.config)
    print(f"[load] {len(id2label)} COCO panoptic classes", flush=True)

    inputs = processor(images=image, return_tensors="pt").to(args.device)
    with torch.no_grad():
        outputs = model(**inputs)
    panoptic = processor.post_process_panoptic_segmentation(
        outputs,
        target_sizes=[(H_img, W_img)],
        threshold=args.score_thresh,
        mask_threshold=0.5,
    )[0]
    seg_map = panoptic["segmentation"].cpu().numpy()           # (H, W) int per-pixel segment id
    seg_info = panoptic["segments_info"]                       # list of {id, label_id, score, ...}

    n_segments = len(seg_info)
    print(f"[predict] {n_segments} segments (score >= {args.score_thresh})", flush=True)
    n_person = 0
    person_segments = []
    for s in seg_info:
        label = id2label.get(s["label_id"], "?")
        marker = ""
        if "person" in label.lower():
            n_person += 1
            person_segments.append(s["id"])
            marker = "  <-- PERSON"
        print(f"  segment id={s['id']:3d}  label={label!r:25s}  score={s['score']:.3f}{marker}", flush=True)

    print(f"\n[predict] n_person_masks = {n_person}  (target: <=2)", flush=True)

    # ── Visualisation: triptych RGB | all-instance colour | person red overlay ──
    rgb_np = np.asarray(image)                                  # (H, W, 3) uint8

    # Random-but-deterministic colour per segment id.
    rng = np.random.default_rng(seed=0)
    color_palette = (rng.integers(64, 256, size=(max(1, n_segments + 2), 3))).astype(np.uint8)
    color_palette[0] = (0, 0, 0)  # background

    all_inst = np.zeros_like(rgb_np)
    for s in seg_info:
        m = (seg_map == s["id"])
        # remap segment id -> palette index (1-based)
        idx = (s["id"] % (len(color_palette) - 1)) + 1
        all_inst[m] = color_palette[idx]
    all_blend = (rgb_np.astype(np.float32) * 0.5 + all_inst.astype(np.float32) * 0.5).astype(np.uint8)

    # Person-only overlay (red 220,40,40 at alpha 0.55)
    person_mask = np.zeros((H_img, W_img), dtype=bool)
    for sid in person_segments:
        person_mask |= (seg_map == sid)
    person_overlay = rgb_np.copy()
    if person_mask.any():
        person_overlay[person_mask] = (
            0.45 * rgb_np[person_mask].astype(np.float32)
            + 0.55 * np.array([220, 40, 40], dtype=np.float32)
        ).astype(np.uint8)

    sep = np.full((H_img, 8, 3), 255, dtype=np.uint8)
    triptych = np.concatenate([rgb_np, sep, all_blend, sep, person_overlay], axis=1)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(triptych).save(out_path)
    print(f"[save] triptych {triptych.shape} -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
