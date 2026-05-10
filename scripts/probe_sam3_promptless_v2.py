"""
B.0.0 v2 — probe SAM-3 modes (v1 found pipeline mask-generation broken on
facebook/sam3 due to missing AMG decoder weights; abstract vocab failed too).

This v2 tries:
  M1. Concrete indoor-object vocab via SAM-3 text prompt (LOW threshold).
  M2. FastSAM-bbox seeding → SAM-3 box-prompt refinement (planned fallback).
  M3. (sanity) Same broad-vocab queries the user already saw working in
      `radio_seg_novel_sam3` (e.g. "a hand, keyboard, ceiling, ...").
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import cv2
import torch
from PIL import Image

torch.backends.cudnn.deterministic = True

REPO_ROOT = Path("/home/cvg/HERMES-SLAM/DROID-W")
sys.path.insert(0, str(REPO_ROOT))


def _overlay_masks(rgb: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    out = rgb.copy()
    rng = np.random.default_rng(42)
    for m in masks:
        if m.dtype != bool:
            m = m > 0.5
        if not m.any():
            continue
        c = rng.integers(60, 255, size=3, dtype=np.uint8)
        out[m] = (0.55 * out[m] + 0.45 * c).astype(np.uint8)
    return out


def _coverage(masks: list[np.ndarray], H: int, W: int) -> float:
    if not masks:
        return 0.0
    union = np.zeros((H, W), dtype=bool)
    for m in masks:
        if m.dtype != bool:
            m = m > 0.5
        union |= m
    return float(union.sum()) / (H * W) * 100.0


def _to_mask_list(seg, H, W):
    out = []
    if seg is None:
        return out
    if torch.is_tensor(seg):
        seg = seg.cpu().numpy()
    seg = np.asarray(seg)
    if seg.ndim == 3:
        for k in range(seg.shape[0]):
            m = seg[k]
            if m.shape != (H, W):
                continue
            out.append((m > 0.5).astype(bool))
    elif seg.ndim == 2:
        for lab in np.unique(seg):
            if lab <= 0:
                continue
            out.append(seg == lab)
    return out


def main() -> int:
    out_dir = Path("/tmp/probe_sam3_results_v2")
    out_dir.mkdir(exist_ok=True, parents=True)

    v = np.load(REPO_ROOT / "Outputs/TUM_RGBD/freiburg3_walking_static/video.npz")
    img = v["images"][0]
    if img.dtype != np.uint8:
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    rgb = img.transpose(1, 2, 0)
    H, W = rgb.shape[:2]
    pil = Image.fromarray(rgb)
    print(f"[setup] KF 0 shape: {rgb.shape}", flush=True)
    cv2.imwrite(str(out_dir / "0_input.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # Load SAM-3 once
    from transformers import Sam3Model, Sam3Processor
    print("[setup] loading SAM-3 ...", flush=True)
    sam3_proc = Sam3Processor.from_pretrained("facebook/sam3")
    sam3_model = Sam3Model.from_pretrained("facebook/sam3").to("cuda:0").eval()

    summary = {}

    # ── MODE 1: concrete indoor-objects vocab, low threshold ───────────────
    indoor = ["person", "chair", "monitor", "table", "keyboard", "plant",
              "door", "floor", "wall", "ceiling", "lamp", "book", "computer"]
    for thr in [0.5, 0.3, 0.2]:
        print(f"\n=== MODE 1 indoor vocab thr={thr} ===", flush=True)
        all_masks = []
        for q in indoor:
            inputs = sam3_proc(images=pil, text=q, return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                out = sam3_model(**inputs)
            res = sam3_proc.post_process_instance_segmentation(
                out, threshold=thr, target_sizes=[(H, W)],
            )[0]
            seg = res.get("masks") if "masks" in res else res.get("segmentation")
            new = _to_mask_list(seg, H, W)
            all_masks.extend(new)
        coverage = _coverage(all_masks, H, W)
        print(f"  thr={thr}: {len(all_masks)} masks, cov={coverage:.1f}%", flush=True)
        summary[f"indoor_thr{thr}"] = {"n_masks": len(all_masks), "coverage": coverage}
        if all_masks:
            ov = _overlay_masks(rgb, all_masks)
            cv2.imwrite(str(out_dir / f"1_indoor_thr{thr}.png"),
                        cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

    # ── MODE 2: FastSAM-bbox + SAM-3-refine (the planned fallback) ─────────
    print("\n=== MODE 2: FastSAM bbox -> SAM-3 box-prompt refine ===", flush=True)
    try:
        from src.utils.mono_priors.fastsam_segmentor import (
            get_fastsam_model, segment_everything,
        )
        fast_model = get_fastsam_model(model_name="FastSAM-x.pt", device="cuda:0")
        det = segment_everything(fast_model, rgb, device="cuda:0",
                                 imgsz=1024, conf=0.4, iou=0.9)
        print(f"  FastSAM produced {len(det)} proposals", flush=True)

        # Box-prompt SAM-3 with each FastSAM bbox
        boxes = [d["box"] for d in det if d.get("area", 0) >= 200]
        print(f"  {len(boxes)} boxes ≥200 area passed to SAM-3", flush=True)

        sam3_masks = []
        for box in boxes[:60]:  # cap at 60 to keep wall-time bounded
            # Sam3Processor accepts input_boxes as [[[x1,y1,x2,y2], ...]] per image
            inputs = sam3_proc(
                images=pil,
                input_boxes=[[list(map(float, box))]],
                return_tensors="pt",
            ).to("cuda:0")
            with torch.no_grad():
                out = sam3_model(**inputs)
            res = sam3_proc.post_process_instance_segmentation(
                out, threshold=0.3, target_sizes=[(H, W)],
            )[0]
            seg = res.get("masks") if "masks" in res else res.get("segmentation")
            ms = _to_mask_list(seg, H, W)
            if ms:
                # take the highest-score mask from each box prompt
                sam3_masks.append(ms[0])

        coverage = _coverage(sam3_masks, H, W)
        print(f"  FastSAM→SAM-3 refined: {len(sam3_masks)} masks, cov={coverage:.1f}%", flush=True)
        summary["fastsam_then_sam3"] = {
            "n_proposals": len(boxes),
            "n_refined": len(sam3_masks),
            "coverage": coverage,
        }
        if sam3_masks:
            ov = _overlay_masks(rgb, sam3_masks)
            cv2.imwrite(str(out_dir / "2_fastsam_then_sam3.png"),
                        cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
        summary["fastsam_then_sam3"] = {"error": f"{type(e).__name__}: {e}"}

    # ── MODE 3: novel vocab from earlier successful run (sanity) ───────────
    print("\n=== MODE 3: novel vocab (sanity vs prior radio_seg_novel_sam3) ===", flush=True)
    novel = ["a hand", "keyboard", "ceiling", "a coffee mug", "shoe", "plant",
             "person", "monitor", "office chair", "radiator", "door", "floor"]
    all_masks = []
    for q in novel:
        inputs = sam3_proc(images=pil, text=q, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = sam3_model(**inputs)
        res = sam3_proc.post_process_instance_segmentation(
            out, threshold=0.5, target_sizes=[(H, W)],
        )[0]
        seg = res.get("masks") if "masks" in res else res.get("segmentation")
        all_masks.extend(_to_mask_list(seg, H, W))
    coverage = _coverage(all_masks, H, W)
    print(f"  novel-vocab union: {len(all_masks)} masks, cov={coverage:.1f}%", flush=True)
    summary["novel_vocab_union"] = {"n_masks": len(all_masks), "coverage": coverage}
    if all_masks:
        ov = _overlay_masks(rgb, all_masks)
        cv2.imwrite(str(out_dir / "3_novel_vocab.png"),
                    cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

    # ── Verdict ────────────────────────────────────────────────────────────
    print("\n=== VERDICT ===", flush=True)
    import json
    print(json.dumps(summary, indent=2), flush=True)

    # Pick the highest-coverage mode that has ≥10 masks
    best = None
    for name, r in summary.items():
        if r.get("n_masks", r.get("n_refined", 0)) >= 10:
            cov = r.get("coverage", 0.0)
            if best is None or cov > best[1]:
                best = (name, cov)
    if best:
        print(f"\nPASS — chosen={best[0]} coverage={best[1]:.1f}%", flush=True)
    else:
        print("\nFAIL — no mode produced ≥10 masks; B.1 needs a different detector", flush=True)
    print(f"\noutputs in {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
