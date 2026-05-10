"""
B.1 sharpness verifier — automated mask-quality metrics + visual overlays.

We don't have a per-frame instance GT for freiburg3 (the user's reference
"sharp baseline" was the live SAM-3 `radio_seg_novel_sam3/combined.mp4`,
which we don't store at the per-mask level). Instead we apply two automated
proxies for SAM-3 sharpness, which together give a paper-defensible signal:

  1. **Compactness (perimeter² / (4π · area))** — a perfect circle scores 1.
     Jagged / fragmented masks score arbitrarily high (linear in perimeter).
     SAM-3 silhouettes typically score 1.5–4.0 for well-shaped objects.
     **Gate (paper-grade): median compactness ≤ 6.0** across all instances
     of the run. Above this signals fragmentation / aliasing.

  2. **Hole fraction (1 - mask_area / convex_hull_area)** — masks with many
     holes have high values. Sharp SAM-3 typically <0.10.
     **Gate: median hole-fraction ≤ 0.15.**

Plus visual: for K random KFs, save an RGB+mask-overlay PNG to inspect.
The user's eyeball remains the final arbiter (CLAUDE.md §0).

Usage:
    python scripts/verify_panoptic_sharpness.py \\
        --panoptic Outputs/TUM_RGBD/freiburg3_walking_static/panoptic.npz \\
        --video    Outputs/TUM_RGBD/freiburg3_walking_static/video.npz \\
        --n-vis-kfs 5 \\
        --vis-out  /tmp/panoptic_sharpness/freiburg3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def _compactness(mask: np.ndarray) -> float:
    """perimeter² / (4π · area). 1 for a circle; >1 for elongated/jagged."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.sum() < 1:
        return float("nan")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return float("nan")
    # Sum perimeters of all components (handles fragmented masks naturally).
    perim = sum(cv2.arcLength(c, True) for c in contours)
    area = float(mask.sum())
    if area <= 0 or perim <= 0:
        return float("nan")
    return float(perim * perim / (4 * np.pi * area))


def _hole_fraction(mask: np.ndarray) -> float:
    """1 - area / convex_hull_area. 0 for a convex blob; up to 1 for ring shapes."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.sum() < 1:
        return float("nan")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return float("nan")
    # Take largest contour for the hull.
    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)
    hull_area = float(cv2.contourArea(hull))
    area = float(mask.sum())
    if hull_area <= 0:
        return float("nan")
    return float(1.0 - area / hull_area)


def _save_overlay(rgb: np.ndarray, masks_kf: list[np.ndarray], out_path: Path,
                  title: str) -> None:
    """Render RGB + instance overlay (random colors per instance) to PNG."""
    out = rgb.copy()
    rng = np.random.default_rng(42)
    for m in masks_kf:
        if not m.any():
            continue
        c = rng.integers(60, 255, size=3, dtype=np.uint8)
        out[m] = (0.55 * out[m] + 0.45 * c).astype(np.uint8)
    # Add title bar.
    h, w = out.shape[:2]
    BANNER = 30
    canvas = np.zeros((h + BANNER, w, 3), dtype=np.uint8)
    cv2.putText(canvas[:BANNER], title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    canvas[BANNER:] = out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--panoptic", required=True, type=str)
    p.add_argument("--video", required=True, type=str)
    p.add_argument("--gate-median-compactness", default=6.0, type=float)
    p.add_argument("--gate-median-hole-fraction", default=0.15, type=float)
    p.add_argument("--n-vis-kfs", default=5, type=int)
    p.add_argument("--vis-out", default=None, type=str)
    p.add_argument("--seed", default=42, type=int)
    args = p.parse_args()

    pan_path = Path(args.panoptic)
    vid_path = Path(args.video)
    print(f"[setup] panoptic={pan_path}", flush=True)
    print(f"[setup] video={vid_path}", flush=True)

    m = np.load(pan_path)
    masks = m["masks"]                                 # (K, H, W) uint8
    per_kf = m["per_kf_offsets"]                       # (N+1,)
    K_total, H, W = masks.shape
    n_kfs = len(per_kf) - 1
    print(f"[panoptic] K_total={K_total}  N_kfs={n_kfs}  HxW={H}x{W}", flush=True)

    # ── Automated metrics ───────────────────────────────────────────────────
    print("\n=== mask-quality metrics ===", flush=True)
    compact = []
    hole = []
    n_skip = 0
    for k in range(K_total):
        mk = masks[k]
        c = _compactness(mk)
        h_ = _hole_fraction(mk)
        if not np.isfinite(c) or not np.isfinite(h_):
            n_skip += 1
            continue
        compact.append(c)
        hole.append(h_)
    if not compact:
        print(f"[FAIL] no valid masks to score", flush=True)
        return 1
    compact = np.array(compact)
    hole = np.array(hole)
    med_c = float(np.median(compact))
    p95_c = float(np.percentile(compact, 95))
    med_h = float(np.median(hole))
    p95_h = float(np.percentile(hole, 95))
    print(f"  compactness   median={med_c:.3f}  p95={p95_c:.3f}  (gate ≤ {args.gate_median_compactness})", flush=True)
    print(f"  hole_fraction median={med_h:.4f}  p95={p95_h:.4f}  (gate ≤ {args.gate_median_hole_fraction})", flush=True)
    print(f"  scored {len(compact)}/{K_total} masks ({n_skip} skipped, e.g. empty/degenerate)", flush=True)

    fail = False
    if med_c > args.gate_median_compactness:
        print(f"[FAIL] median compactness {med_c:.3f} > gate {args.gate_median_compactness}", flush=True)
        fail = True
    if med_h > args.gate_median_hole_fraction:
        print(f"[FAIL] median hole_fraction {med_h:.4f} > gate {args.gate_median_hole_fraction}", flush=True)
        fail = True

    # ── Visual overlays ─────────────────────────────────────────────────────
    if args.vis_out:
        v = np.load(vid_path)
        images = v["images"]
        if images.dtype != np.uint8:
            images = (images * 255.0).clip(0, 255).astype(np.uint8)
        rng = np.random.default_rng(args.seed)
        kf_pick = rng.choice(n_kfs, size=min(args.n_vis_kfs, n_kfs), replace=False)
        out_root = Path(args.vis_out)
        out_root.mkdir(parents=True, exist_ok=True)
        for kf in sorted(kf_pick.tolist()):
            rgb = images[kf].transpose(1, 2, 0)
            start, end = int(per_kf[kf]), int(per_kf[kf + 1])
            kf_masks = [masks[i].astype(bool) for i in range(start, end)]
            title = f"KF {kf:3d}  K={end-start}  panoptic_smoke"
            _save_overlay(rgb, kf_masks,
                          out_root / f"kf_{kf:04d}_overlay.png",
                          title)
        print(f"[vis] wrote {len(kf_pick)} overlays to {out_root}", flush=True)

    if fail:
        print(f"\n[FAIL] {pan_path}", flush=True)
        return 1
    print(f"\n[PASS] {pan_path}  (median compactness {med_c:.2f} ≤ {args.gate_median_compactness}, "
          f"median hole {med_h:.4f} ≤ {args.gate_median_hole_fraction})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
