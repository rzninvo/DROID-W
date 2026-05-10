"""
B.0.0 — probe SAM-3 promptless / broad-vocab modes.

Determines which SAM-3 invocation produces the densest, sharpest entity coverage
on a single freiburg3 keyframe. Output drives B.1's PRIMARY detector choice.

Modes tested:
  1. pipeline("mask-generation") — the documented auto-mask-generator API.
  2. broad-vocab text union: ["a thing", "an object", "an item"].
  3. (control) text-prompted "person" — for visual reference.

Output: /tmp/probe_sam3_results/ — overlay PNGs for each mode + a one-line
PASS/FAIL verdict to stdout.
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
    """Random-colored mask overlay on rgb (H,W,3) uint8."""
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


def main() -> int:
    out_dir = Path("/tmp/probe_sam3_results")
    out_dir.mkdir(exist_ok=True, parents=True)

    # Load KF 0 from freiburg3
    v = np.load(REPO_ROOT / "Outputs/TUM_RGBD/freiburg3_walking_static/video.npz")
    img = v["images"][0]
    if img.dtype != np.uint8:
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    rgb = img.transpose(1, 2, 0)  # (H, W, 3)
    H, W = rgb.shape[:2]
    pil = Image.fromarray(rgb)
    print(f"[setup] KF 0 shape: {rgb.shape} dtype={rgb.dtype}", flush=True)
    cv2.imwrite(str(out_dir / "0_input.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    results_summary: dict[str, dict] = {}

    # ── MODE 1: pipeline("mask-generation") ────────────────────────────────
    print("\n=== MODE 1: pipeline('mask-generation') ===", flush=True)
    try:
        from transformers import pipeline
        t0 = time.time()
        generator = pipeline("mask-generation", model="facebook/sam3", device=0)
        load_s = time.time() - t0
        print(f"[mode1] pipeline loaded in {load_s:.1f}s", flush=True)

        t0 = time.time()
        out = generator(pil, points_per_batch=64)
        infer_s = time.time() - t0
        masks_list = out["masks"] if isinstance(out, dict) else out
        # Coerce to list of (H, W) bool numpy arrays
        m_arr = []
        for m in masks_list:
            if torch.is_tensor(m):
                m = m.cpu().numpy()
            m = np.squeeze(m)
            if m.ndim == 2:
                m_arr.append(m.astype(bool))
        coverage = _coverage(m_arr, H, W)
        print(f"[mode1] {len(m_arr)} masks generated, coverage={coverage:.2f}%, infer={infer_s:.1f}s", flush=True)

        overlay = _overlay_masks(rgb, m_arr)
        cv2.imwrite(str(out_dir / "1_pipeline_promptless.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        results_summary["mode1_pipeline_promptless"] = {
            "supported": True, "n_masks": len(m_arr),
            "coverage_pct": coverage, "infer_s": infer_s,
        }

        del generator
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[mode1] FAILED: {type(e).__name__}: {e}", flush=True)
        results_summary["mode1_pipeline_promptless"] = {
            "supported": False, "error": f"{type(e).__name__}: {e}",
        }

    # ── MODE 2: broad-vocab text prompt union ──────────────────────────────
    print("\n=== MODE 2: broad-vocab text prompt ===", flush=True)
    try:
        from transformers import Sam3Model, Sam3Processor
        t0 = time.time()
        sam3_proc = Sam3Processor.from_pretrained("facebook/sam3")
        sam3_model = Sam3Model.from_pretrained("facebook/sam3").to("cuda:0").eval()
        load_s = time.time() - t0
        print(f"[mode2] model loaded in {load_s:.1f}s", flush=True)

        all_masks: list[np.ndarray] = []
        all_scores: list[float] = []
        for q in ["a thing", "an object", "an item"]:
            inputs = sam3_proc(images=pil, text=q, return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                out = sam3_model(**inputs)
            res = sam3_proc.post_process_instance_segmentation(
                out, threshold=0.5, target_sizes=[(H, W)],
            )[0]
            seg = res.get("segmentation")
            if seg is None:
                continue
            if torch.is_tensor(seg):
                seg = seg.cpu().numpy()
            # `segmentation` may be a (K, H, W) stack OR (H, W) labeled map.
            if seg.ndim == 3:
                for k in range(seg.shape[0]):
                    all_masks.append(seg[k] > 0.5)
            elif seg.ndim == 2:
                # labeled map — break into per-instance masks
                for lab in np.unique(seg):
                    if lab <= 0:
                        continue
                    all_masks.append(seg == lab)
            scores = res.get("scores")
            if scores is not None:
                if torch.is_tensor(scores):
                    scores = scores.cpu().numpy()
                all_scores.extend(scores.tolist())
            print(f"  query='{q}': +{len(all_masks)} cumulative", flush=True)

        coverage = _coverage(all_masks, H, W)
        print(f"[mode2] {len(all_masks)} masks (broad-vocab union), coverage={coverage:.2f}%", flush=True)
        overlay = _overlay_masks(rgb, all_masks)
        cv2.imwrite(str(out_dir / "2_broad_vocab.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        results_summary["mode2_broad_vocab"] = {
            "supported": True, "n_masks": len(all_masks),
            "coverage_pct": coverage,
        }
        del sam3_model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[mode2] FAILED: {type(e).__name__}: {e}", flush=True)
        results_summary["mode2_broad_vocab"] = {
            "supported": False, "error": f"{type(e).__name__}: {e}",
        }

    # ── Verdict ────────────────────────────────────────────────────────────
    print("\n=== VERDICT ===", flush=True)
    m1 = results_summary.get("mode1_pipeline_promptless", {})
    m2 = results_summary.get("mode2_broad_vocab", {})
    if m1.get("supported") and m1.get("n_masks", 0) >= 5:
        chosen = "pipeline-promptless"
        print(f"PASS — chosen={chosen}  n_masks={m1['n_masks']}  coverage={m1['coverage_pct']:.1f}%", flush=True)
    elif m2.get("supported") and m2.get("n_masks", 0) >= 5:
        chosen = "broad-vocab-union"
        print(f"PASS — chosen={chosen}  n_masks={m2['n_masks']}  coverage={m2['coverage_pct']:.1f}%", flush=True)
    else:
        chosen = "fastsam-bbox-then-sam3-refine"
        print(f"FALLBACK — neither mode usable; B.1 PRIMARY = {chosen}", flush=True)
        print(f"  mode1: {m1}", flush=True)
        print(f"  mode2: {m2}", flush=True)

    print(f"\n[done] outputs in {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
