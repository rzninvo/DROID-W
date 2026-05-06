"""
Post-processing test driver for RADIO + SigLIP-2 per-keyframe text grounding.

Runs the new `RadioGrounder` against the keyframes saved in a DROID-W
`video.npz`, with an arbitrary list of text queries. Writes one mp4 per
query plus a `combined.mp4` that overlays every query in distinct colours.

Usage
-----
    python scripts/test_radio_segmentation.py \\
        --scene Outputs/TUM_RGBD/freiburg3_walking_static \\
        --queries person monitor "office chair" radiator door floor \\
        --threshold 0.20

This is a deliberately simple test loop — it does NOT touch DROID-W's SLAM
loop, scene-graph code, or any tracker. It only loads `images` from
`video.npz`, runs grounding per keyframe, and writes overlay videos.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

# Allow `python scripts/...` from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.mono_priors.radio_grounding import RadioGrounder


# Distinct BGR colours for per-query overlays. Cycles if Q > len(palette).
# Picked to read clearly on the typical TUM/Bonn indoor RGB.
_PALETTE_BGR = [
    ( 60,  60, 230),  # red
    ( 60, 230,  60),  # green
    (230,  60,  60),  # blue
    ( 60, 230, 230),  # yellow
    (230,  60, 230),  # magenta
    (230, 230,  60),  # cyan
    ( 60, 150, 230),  # orange
    (200, 100, 230),  # pink
]


def _load_keyframes(scene: Path) -> np.ndarray:
    """Load (N, 3, H, W) uint8 RGB keyframes from `<scene>/video.npz`.

    DROID-W stores `images` as float32 in [0, 1]. Convert to uint8 once.
    """
    npz_path = scene / "video.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"video.npz not found at {npz_path}")
    print(f"[load] {npz_path}", flush=True)
    data = np.load(npz_path)
    images = data["images"]
    if images.dtype != np.uint8:
        images = (images * 255.0).clip(0, 255).astype(np.uint8)
    print(f"[load] images: shape={images.shape} dtype={images.dtype}", flush=True)
    return images


def _overlay_mask(rgb_bgr: np.ndarray, mask: np.ndarray, color_bgr) -> np.ndarray:
    """Blend a colour into the BGR image where `mask` is True. Non-destructive."""
    if mask.sum() == 0:
        return rgb_bgr
    out = rgb_bgr.copy()
    overlay = np.zeros_like(out)
    overlay[mask] = color_bgr
    out[mask] = (0.55 * out[mask] + 0.45 * overlay[mask]).astype(np.uint8)
    return out


def _draw_legend(
    frame_bgr: np.ndarray,
    queries: List[str],
    colours: List[tuple],
    kf_idx: int,
    n_total: int,
) -> np.ndarray:
    """Draw a small legend with query → colour swatches at the top-left."""
    out = frame_bgr.copy()
    y = 18
    for q, c in zip(queries, colours):
        cv2.rectangle(out, (10, y - 10), (28, y + 4), c, -1)
        cv2.putText(out, q, (34, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 18
    H = out.shape[0]
    cv2.putText(out, f"KF {kf_idx}/{n_total}", (10, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="RADIO + SigLIP-2 per-keyframe text grounding test.")
    p.add_argument("--scene", required=True, type=Path,
                   help="Scene directory containing video.npz, e.g. Outputs/TUM_RGBD/freiburg3_walking_static")
    p.add_argument("--queries", required=True, nargs="+",
                   help="One or more text queries, e.g. person monitor 'office chair'")
    p.add_argument("--threshold", type=float, default=0.50,
                   help="Per-pixel softmax-over-queries threshold for binary mask "
                        "(default 0.50). With temperature=100 (RADIO-ViPE default), "
                        "a confident pixel scores ~0.9+ for its winning query. "
                        "Threshold semantics: 'winning query beats runners-up' — "
                        "0.50 = 2× runner-up, 0.85 ~= 5× runner-up. "
                        "Pass --raw-cosine to threshold raw cosine instead.")
    p.add_argument("--raw-cosine", action="store_true",
                   help="Threshold on raw cosine similarity instead of softmax. "
                        "Useful for diagnostics; cosine values cluster 0.05-0.20.")
    p.add_argument("--output", type=Path, default=None,
                   help="Output dir (default: <scene>/radio_seg).")
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--radio-version", type=str, default="c-radio_v3-b")
    p.add_argument("--lang-adaptor", type=str, default="siglip2")
    p.add_argument("--max-keyframes", type=int, default=None,
                   help="Optional cap on keyframes (debug).")
    args = p.parse_args()

    scene = args.scene.resolve()
    out_dir = (args.output or (scene / "radio_seg")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] scene={scene}", flush=True)
    print(f"[setup] output={out_dir}", flush=True)
    print(f"[setup] queries={args.queries}", flush=True)
    print(f"[setup] threshold={args.threshold}", flush=True)

    images = _load_keyframes(scene)
    if args.max_keyframes is not None:
        images = images[: args.max_keyframes]

    N, _, H, W = images.shape
    if N == 0:
        print(
            f"[WARN] {scene}/video.npz contained zero keyframes; nothing to render. "
            f"expected=N>0, got=N=0, fallback=exit cleanly",
            flush=True,
        )
        return 1

    # Build grounder
    t0 = time.time()
    grounder = RadioGrounder(
        device=args.device,
        radio_version=args.radio_version,
        lang_adaptor=args.lang_adaptor,
    )
    grounder.set_queries(args.queries)
    print(f"[load] RadioGrounder ready in {time.time() - t0:.1f}s "
          f"(patch_size={grounder.patch_size}, lang_adaptor={grounder.lang_adaptor_name})",
          flush=True)

    Q = len(args.queries)
    colours = [_PALETTE_BGR[i % len(_PALETTE_BGR)] for i in range(Q)]

    # Open per-query writers + a combined writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    per_q_writers = []
    for q in args.queries:
        safe = q.replace(" ", "_").replace("/", "_")
        path = out_dir / f"{safe}.mp4"
        if path.exists():
            path.unlink()
        per_q_writers.append((path, cv2.VideoWriter(str(path), fourcc, args.fps, (W, H))))
    combined_path = out_dir / "combined.mp4"
    if combined_path.exists():
        combined_path.unlink()
    combined_writer = cv2.VideoWriter(str(combined_path), fourcc, args.fps, (W, H))

    # Stats — track per-query mask-pixel coverage for a sanity-check at the end.
    per_q_pixels = np.zeros(Q, dtype=np.int64)
    score_min = np.full(Q, np.inf, dtype=np.float64)
    score_max = np.full(Q, -np.inf, dtype=np.float64)

    score_field = "similarity" if args.raw_cosine else "softmax"
    print(f"[run] grounding {N} keyframes ; thresholding on '{score_field}' "
          f"@ {args.threshold}", flush=True)
    t_run = time.time()
    for i in range(N):
        rgb = images[i].transpose(1, 2, 0)  # (H, W, 3) uint8 RGB
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        result = grounder.segment(rgb)
        score = result[score_field]   # (Q, H, W)

        # Per-query writer with that query's binary mask + colour
        for qi in range(Q):
            mask_q = score[qi] >= args.threshold
            per_q_pixels[qi] += int(mask_q.sum())
            score_min[qi] = min(score_min[qi], float(score[qi].min()))
            score_max[qi] = max(score_max[qi], float(score[qi].max()))
            frame = _overlay_mask(bgr, mask_q, colours[qi])
            frame = _draw_legend(frame, [args.queries[qi]], [colours[qi]], i, N)
            per_q_writers[qi][1].write(frame)

        # Combined: each pixel takes the winning query (argmax over queries
        # in softmax space), but only if best_score (softmax probability of
        # the winning query) is above threshold. Pixels below threshold for
        # all queries stay un-coloured.
        combined = bgr.copy()
        best_q = result["best_query"]   # (H, W) int — argmax of softmax
        best_score = result["best_score"]  # (H, W) — softmax max
        valid = best_score >= args.threshold
        if valid.any():
            for qi in range(Q):
                m = valid & (best_q == qi)
                if m.any():
                    combined = _overlay_mask(combined, m, colours[qi])
        combined = _draw_legend(combined, args.queries, colours, i, N)
        combined_writer.write(combined)

        if i % 10 == 0:
            print(f"  KF {i:3d}/{N}", flush=True)

    print(f"[run] grounding loop done in {time.time() - t_run:.1f}s", flush=True)

    for _, w in per_q_writers:
        w.release()
    combined_writer.release()

    print(f"\n=== Per-query coverage stats (mask-pixel fraction, sim range) ===", flush=True)
    total_px = N * H * W
    for qi, q in enumerate(args.queries):
        frac = per_q_pixels[qi] / max(1, total_px)
        print(f"  {q:20s} : {frac*100:5.2f}% pixels above thr={args.threshold:.2f}, "
              f"sim range [{score_min[qi]:+.3f}, {score_max[qi]:+.3f}]", flush=True)

    print(f"\n[done] outputs:", flush=True)
    for path, _ in per_q_writers:
        print(f"  {path}", flush=True)
    print(f"  {combined_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
