"""
Render a single-class highlight video — vivid colour on the queried
class, dimmed background everywhere else.

Uses the SAME pipeline as Phase B (RADIO + SigLIP-2 + **SAM-3**) so masks
stay sharp. Class is `--query` (any text); nothing about "chair" or
"person" is hard-coded.

Two query modes (auto-selected):
1. **Cached** — if `<scene>/radseg_masks.npz` already contains the query,
   read masks directly. ~1 s for the full video (no model load).
2. **Live** — otherwise, run RadioGrounder with `set_queries([query])`
   over all keyframes. ~40 s on freiburg3_walking_static at v4-h+SAM-3
   (one query × 83 KFs × 0.5 s/KF SAM-3 forward).

In both modes the visualization is the same: detected pixels keep their
RGB and get a coloured overlay; everything else is desaturated to gray
to draw the eye to the queried object.

Usage (cvg):
    python scripts/highlight_class.py \\
        --scene Outputs/TUM_RGBD/freiburg3_walking_static \\
        --query "office chair" \\
        --threshold 0.50 \\
        --output Outputs/TUM_RGBD/freiburg3_walking_static/highlight_chair.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Determinism flags before torch import (CLAUDE.md §0).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.mono_priors.radio_grounding import RadioGrounder

_HIGHLIGHT_BGR = (60, 60, 230)   # vivid red — could be made a CLI arg
_BG_GRAY_BLEND = 0.6              # 0=full colour bg, 1=full gray bg


def _query_from_cached_masks(scene: Path, query: str, threshold: float):
    """If `<scene>/radseg_masks.npz` was made for a query list that includes
    `query`, return per-keyframe (best_query==idx, best_score) for that
    class. Otherwise return None."""
    masks_path = scene / "radseg_masks.npz"
    if not masks_path.exists():
        return None
    m = np.load(masks_path)
    queries = list(m["queries"])
    if query not in queries:
        return None
    qi = queries.index(query)
    bq = m["best_query"]                                       # (N, H, W) int16
    bs = m["best_score"]                                       # (N, H, W) float16
    saved_thr = float(m.get("threshold", 0.0))
    if abs(saved_thr - threshold) > 1e-6:
        print(f"[INFO] cached: file used threshold={saved_thr:.2f}, you "
              f"asked threshold={threshold:.2f}; re-thresholding from "
              f"saved best_score (which has the encoder's pred-thresh floor "
              f"baked in at {float(m.get('encoder_pred_thresh', 0.0)):.2f})", flush=True)
    foreground = (bq == qi) & (bs.astype(np.float32) >= threshold)
    return foreground, bs.astype(np.float32)


def _render_mp4(images_uint8: np.ndarray, foreground: np.ndarray, out_path: Path,
                query: str, threshold: float, fps: int = 5,
                source_tag: str = "live") -> None:
    """Write a side-by-side (gray-background + vivid foreground) video."""
    N, _, H, W = images_uint8.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    BANNER_H = 44
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H + BANNER_H))

    color = np.array(_HIGHLIGHT_BGR, dtype=np.uint8)
    n_pixels_total = 0
    for i in range(N):
        rgb = images_uint8[i].transpose(1, 2, 0)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # Background = blend of true RGB and grayscale (so masked region pops).
        out_img = ((1.0 - _BG_GRAY_BLEND) * bgr + _BG_GRAY_BLEND * gray3).astype(np.uint8)
        fg = foreground[i]
        if fg.any():
            # Foreground = original RGB + 45% colour overlay.
            blended = (0.55 * bgr[fg] + 0.45 * color).astype(np.uint8)
            out_img[fg] = blended
            n_pixels_total += int(fg.sum())

        banner = np.zeros((BANNER_H, W, 3), dtype=np.uint8)
        cv2.putText(banner, f"highlight: \"{query}\"  thr={threshold:.2f}  KF {i:3d}/{N}",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(banner, f"source: {source_tag}",
                    (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 220, 255), 1, cv2.LINE_AA)

        vw.write(np.vstack([banner, out_img]))
    vw.release()
    pct = 100.0 * n_pixels_total / max(1, N * H * W)
    print(f"[done] wrote {out_path}  ({pct:.2f}% of total pixels highlighted)",
          flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, type=Path)
    p.add_argument("--query", required=True, type=str,
                   help="Free-text class to highlight (any phrase).")
    p.add_argument("--threshold", default=0.50, type=float,
                   help="Softmax threshold; pixels with best_score >= thr and "
                        "argmax == query class are highlighted.")
    p.add_argument("--output", default=None, type=Path,
                   help="Default: <scene>/highlight_<query>.mp4")
    p.add_argument("--device", default="cuda:0", type=str)
    p.add_argument("--radio-version", default="c-radio_v4-h", type=str)
    p.add_argument("--no-sam-refinement", action="store_true",
                   help="Disable SAM-3 (faster but fuzzier — Phase B' quality).")
    p.add_argument("--no-cache", action="store_true",
                   help="Ignore <scene>/radseg_masks.npz; always run live.")
    p.add_argument("--fps", default=5, type=int)
    args = p.parse_args()

    scene = args.scene.resolve()
    if args.output is None:
        safe = args.query.replace(" ", "_").replace("/", "_")
        out_path = scene / f"highlight_{safe}.mp4"
    else:
        out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    video_path = scene / "video.npz"
    if not video_path.exists():
        print(f"[ERR] {video_path} not found", flush=True)
        return 1
    v = np.load(video_path)
    images = v["images"]
    if images.dtype != np.uint8:
        images = (images * 255.0).clip(0, 255).astype(np.uint8)

    # ── Try cached path first (CLAUDE.md §3 — simplicity first). ─────
    foreground = None
    source_tag = "live (RADIO + SAM-3, single query)"
    if not args.no_cache:
        cached = _query_from_cached_masks(scene, args.query, args.threshold)
        if cached is not None:
            foreground, _bs = cached
            source_tag = f"cached: {scene.name}/radseg_masks.npz (Phase B)"
            print(f"[cache] hit: '{args.query}' is in radseg_masks.npz; "
                  f"using saved Phase B masks (SAM-3-refined). No model load.",
                  flush=True)

    # ── Live path (Phase B-quality, single query). ───────────────────
    if foreground is None:
        if not args.no_cache:
            print(f"[cache] miss: '{args.query}' not in radseg_masks.npz; "
                  f"running grounder live for one query.", flush=True)
        t_load = time.time()
        grounder = RadioGrounder(
            device=args.device,
            radio_version=args.radio_version,
            sam_refinement=(not args.no_sam_refinement),
        )
        grounder.set_queries([args.query])
        print(f"[load] grounder ready in {time.time() - t_load:.1f}s "
              f"(SAM-3={not args.no_sam_refinement})", flush=True)

        N, _, H, W = images.shape
        foreground = np.zeros((N, H, W), dtype=bool)
        t_loop = time.time()
        for i in range(N):
            rgb = images[i].transpose(1, 2, 0)
            result = grounder.segment(rgb)
            bq = result["best_query"]                                # (H, W) ∈ {-1, 0}
            bs = result["best_score"]                                # (H, W) float
            foreground[i] = (bq == 0) & (bs >= args.threshold)
            del result
            if (i + 1) % 10 == 0 or i == 0 or i == N - 1:
                el = time.time() - t_loop
                eta = el / max(1, i + 1) * (N - i - 1)
                print(f"  [{i+1:3d}/{N}]  ({el:5.1f}s, ~{eta:5.1f}s ETA)", flush=True)
            torch.cuda.empty_cache() if (i + 1) % 25 == 0 else None
        print(f"[done] grounded {N} keyframes for '{args.query}' in "
              f"{time.time() - t_loop:.1f}s", flush=True)

    _render_mp4(images, foreground, out_path, args.query, args.threshold,
                fps=args.fps, source_tag=source_tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
