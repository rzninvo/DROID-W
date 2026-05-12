"""
Plan-v2 §Step 4 / B.1 — TRUE CropFormer smoke (replaces the earlier
Mask2Former-Swin-L probe). Uses the qqlu/Entity (EntityV2) class-agnostic
entity segmenter, matching OVI-MAP §4 ("we adopt CropFormer for 2D entity
segmentation"). Tested on KF 9 of freiburg3_walking_static.

CropFormer is class-agnostic: it returns instance masks per "entity"
without category labels (one mask per coherent object boundary). This is
the right input for Plan B v7's per-instance SigLIP labeling stage (B.3),
since open-vocab labels are assigned LATER, not at proposer time.

Compare metrics to be reported:
  - Mask2Former-Swin-L on the same KF: 15 panoptic segments, 2 person
  - CropFormer: <n_entities>, person ambiguity (no class labels — will
    cross-check against the bbox covering the visible humans).

Usage (cvg, droid-w env):
    python tests/smoke_cropformer_actual_kf9.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the qqlu/Entity CropFormer dir importable.
CROPFORMER_DIR = Path("/home/cvg/HERMES-SLAM/DROID-W/thirdparty/Entity/Entityv2/CropFormer")
sys.path.insert(0, str(CROPFORMER_DIR))
sys.path.insert(0, str(CROPFORMER_DIR / "demo_cropformer"))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import cv2
from PIL import Image

# Detectron2 config + predictor.
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from predictor import CropFormerPredictor  # from demo_cropformer/predictor.py


def setup_cfg(config_file: str, weights_path: str, device: str = "cuda:0",
              confidence_threshold: float = 0.5):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = device
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    # confidence threshold — CropFormer EntityV2 may use higher scores
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.freeze()
    return cfg


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rgb-dir", default="datasets/TUM_RGBD/rgbd_dataset_freiburg3_walking_static/rgb",
                   type=str)
    p.add_argument("--kf-index", default=9, type=int)
    p.add_argument("--config", default=str(CROPFORMER_DIR /
                   "configs/entityv2/entity_segmentation/cropformer_swin_large_3x.yaml"),
                   type=str)
    p.add_argument("--weights", default="/home/cvg/HERMES-SLAM/DROID-W/weights/"
                   "CropFormer_model/Entity_Segmentation/CropFormer_swin_large_w7_3x/"
                   "CropFormer_swin_large_w7_3x_6843ef.pth", type=str)
    p.add_argument("--out", default="Outputs/TUM_RGBD/freiburg3_walking_static/"
                   "step4_b1_kf9_cropformer.png", type=str)
    p.add_argument("--score-thresh", default=0.5, type=float)
    p.add_argument("--device", default="cuda:0", type=str)
    args = p.parse_args()

    rgb_dir = Path(args.rgb_dir)
    if not rgb_dir.is_absolute():
        rgb_dir = Path.cwd() / rgb_dir
    rgb_files = sorted(f for f in rgb_dir.iterdir() if f.suffix == ".png")
    rgb_path = rgb_files[args.kf_index]
    print(f"[setup] KF {args.kf_index} -> {rgb_path.name}", flush=True)
    image_bgr = cv2.imread(str(rgb_path))
    H_img, W_img = image_bgr.shape[:2]
    print(f"[setup] image {W_img}x{H_img}", flush=True)

    print(f"[load] config={args.config}", flush=True)
    print(f"[load] weights={args.weights}", flush=True)
    cfg = setup_cfg(args.config, args.weights, args.device, args.score_thresh)
    predictor = CropFormerPredictor(cfg)
    print(f"[load] CropFormerPredictor ready", flush=True)

    # CropFormer expects BGR uint8.
    predictions = predictor(image_bgr)
    print(f"[predict] keys: {list(predictions.keys()) if isinstance(predictions, dict) else type(predictions)}",
          flush=True)

    # CropFormer returns an "instances" object similar to detectron2 Instances.
    instances = predictions["instances"]
    n = len(instances)
    print(f"[predict] {n} raw entity proposals", flush=True)

    if hasattr(instances, "pred_masks"):
        masks = instances.pred_masks.to("cpu").numpy().astype(bool)
        print(f"[predict] pred_masks shape: {masks.shape}, dtype: {masks.dtype}", flush=True)
    elif hasattr(instances, "pred_mask"):
        masks = instances.pred_mask.to("cpu").numpy().astype(bool)
    else:
        print(f"[ERR] no pred_masks field; fields: {instances.get_fields().keys()}", flush=True)
        return 2

    scores = instances.scores.to("cpu").numpy() if hasattr(instances, "scores") else np.ones(n)
    print(f"[predict] scores: min={scores.min():.3f}  max={scores.max():.3f}  "
          f"median={np.median(scores):.3f}", flush=True)

    # Keep masks above threshold.
    keep = scores >= args.score_thresh
    masks = masks[keep]
    scores = scores[keep]
    print(f"[predict] {len(masks)} entities after score >= {args.score_thresh}", flush=True)

    # Class-agnostic: no labels. The interesting metric is "n entities total"
    # + "are the two people each ONE mask (good) or fragmented (bad)?"
    # We do a quick post-hoc check by computing the bounding boxes and
    # checking how many entity masks fall in the rough human regions.
    # Rough heuristic: humans tend to be tall (h/w > 1.5) and span a large
    # vertical extent. Skip — just report total.

    # ─── Visualisation ───
    rgb_np = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rng = np.random.default_rng(0)
    n_kept = len(masks)
    palette = (rng.integers(64, 256, size=(max(1, n_kept + 1), 3))).astype(np.uint8)

    inst = np.zeros_like(rgb_np)
    for i, m in enumerate(masks):
        inst[m] = palette[(i % (len(palette) - 1)) + 1]
    blend = (rgb_np.astype(np.float32) * 0.5 + inst.astype(np.float32) * 0.5).astype(np.uint8)

    # Mark each entity centroid + score
    sep = np.full((H_img, 8, 3), 255, dtype=np.uint8)
    triptych = np.concatenate([rgb_np, sep, blend, sep, inst], axis=1)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(triptych).save(out_path)
    print(f"[save] triptych shape={triptych.shape} -> {out_path}", flush=True)
    print(f"\n[summary] CropFormer Swin-L on KF 9 walking_static:", flush=True)
    print(f"  n_entities (score >= {args.score_thresh}): {n_kept}", flush=True)
    print(f"  score range: [{scores.min():.3f}, {scores.max():.3f}]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
