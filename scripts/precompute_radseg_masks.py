"""
Phase B — persist per-keyframe radseg open-vocab masks alongside video.npz.

Runs `RadioGrounder.segment()` over every keyframe in `<scene>/video.npz`
and writes a SIBLING file `<scene>/radseg_masks.npz` with:

    best_query  (N, H, W)  int16  — argmax class index (-1 = no class above thr)
    best_score  (N, H, W)  float16 — max softmax probability
    queries     (Q,)       str    — text of each query, in argmax order
    threshold   float              — softmax threshold used at scoring time
    radio_version, lang_adaptor    — backbone identifiers
    sam_refinement                 — bool flag
    n_keyframes                    — int, matches video.npz['images'].shape[0]
    image_hw                       — (H, W) tuple

Why a sibling file (not video.npz mutation): keeps the SLAM artefact
immutable. Phase C consumers (BA reweighting, scene-graph fusion, paper
figures) load the sibling explicitly.

Storage: argmax + score is ~78 KB / keyframe at 384×512 (vs ~12 MB
per-keyframe for the full Q×H×W softmax). Lossy by design — to recover
softmax, re-run the script.

Usage (cvg):
    python scripts/precompute_radseg_masks.py \\
        --scene Outputs/TUM_RGBD/freiburg3_walking_static \\
        --queries person monitor "office chair" radiator door floor \\
        --threshold 0.50

Per CLAUDE.md §6 every fallback emits a [WARN] line.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.mono_priors.radio_grounding import RadioGrounder


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, type=Path,
                   help="Scene dir with video.npz (e.g. Outputs/TUM_RGBD/freiburg3_walking_static)")
    p.add_argument("--queries", required=True, nargs="+",
                   help="Open-vocab text queries (order is preserved as argmax indices).")
    p.add_argument("--threshold", type=float, default=0.50,
                   help="Softmax threshold below which best_query is set to -1. "
                        "Same semantics as test_radio_segmentation.py.")
    p.add_argument("--radio-version", default="c-radio_v4-h", type=str)
    p.add_argument("--lang-adaptor", default=None, type=str,
                   help="Auto-detected from --radio-version when omitted.")
    p.add_argument("--no-sam-refinement", action="store_true",
                   help="Skip SAM-3/SAM-1 mask refinement (faster, lower person IoU).")
    p.add_argument("--output", default=None, type=Path,
                   help="Output .npz path. Default: <scene>/radseg_masks.npz")
    p.add_argument("--device", default="cuda:0", type=str)
    p.add_argument("--max-keyframes", default=None, type=int,
                   help="Optional cap (debug).")
    args = p.parse_args()

    scene = args.scene.resolve()
    npz_path = scene / "video.npz"
    if not npz_path.exists():
        print(f"[ERR] {npz_path} not found", flush=True)
        return 1
    out_path = args.output.resolve() if args.output is not None else scene / "radseg_masks.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[setup] scene={scene}", flush=True)
    print(f"[setup] queries={args.queries}", flush=True)
    print(f"[setup] threshold={args.threshold}  radio={args.radio_version}  "
          f"sam_refinement={not args.no_sam_refinement}", flush=True)
    print(f"[setup] output={out_path}", flush=True)

    z = np.load(npz_path)
    if "images" not in z.files:
        print(f"[ERR] {npz_path} has no 'images' field (got {z.files})", flush=True)
        return 1
    images = z["images"]
    if images.dtype != np.uint8:
        images = (images * 255.0).clip(0, 255).astype(np.uint8)
    N_full, _, H, W = images.shape
    if N_full == 0:
        print(f"[ERR] {npz_path} has 0 keyframes — nothing to ground", flush=True)
        return 1
    if args.max_keyframes is not None:
        if args.max_keyframes >= N_full:
            print(f"[WARN] precompute: --max-keyframes={args.max_keyframes} >= "
                  f"N_full={N_full}; fallback=using all keyframes", flush=True)
            N = N_full
        else:
            N = args.max_keyframes
    else:
        N = N_full
    images = images[:N]
    print(f"[setup] N_keyframes={N} (of {N_full} in video.npz)  "
          f"image (H,W)=({H},{W})  queries Q={len(args.queries)}", flush=True)

    t_load = time.time()
    grounder = RadioGrounder(
        device=args.device,
        radio_version=args.radio_version,
        lang_adaptor=args.lang_adaptor,
        sam_refinement=(not args.no_sam_refinement),
    )
    grounder.set_queries(list(args.queries))
    print(f"[load] grounder ready in {time.time() - t_load:.1f}s "
          f"(patch={grounder.patch_size}, lang={grounder.lang_adaptor_name})",
          flush=True)

    # Allocate output buffers — int16 supports up to 32k classes (we never
    # have that many) and -1 sentinel; float16 is enough precision for a
    # softmax probability.
    best_query = np.full((N, H, W), -1, dtype=np.int16)
    best_score = np.zeros((N, H, W), dtype=np.float16)
    n_dyn_per_kf: list[int] = []

    t_loop = time.time()
    for i in range(N):
        rgb = images[i].transpose(1, 2, 0)  # (H, W, 3) uint8
        result = grounder.segment(rgb)
        bq = result["best_query"]               # (H, W) int32 ∈ [-1, Q-1]
        bs = result["best_score"]               # (H, W) float32
        # Apply explicit threshold here too (encoder already threshold-clamped via
        # prediction_thresh, but we re-apply at args.threshold for downstream
        # users picking different thresholds at query time).
        below_thr = bs < args.threshold
        bq = np.where(below_thr, -1, bq)
        best_query[i] = bq.astype(np.int16)
        best_score[i] = bs.astype(np.float16)
        # Cheap stat: #pixels assigned to ANY class (not -1).
        n_dyn_per_kf.append(int((bq >= 0).sum()))
        del result
        if (i + 1) % 10 == 0 or i == 0 or i == N - 1:
            elapsed = time.time() - t_loop
            eta = elapsed / max(1, i + 1) * (N - i - 1)
            print(f"  [{i+1:4d}/{N}]  any_class={100*n_dyn_per_kf[-1]/(H*W):5.2f}%  "
                  f"({elapsed:5.1f}s, ~{eta:5.1f}s ETA)", flush=True)

    walltime = time.time() - t_loop
    print(f"[done] grounded {N} keyframes in {walltime:.1f}s "
          f"(mean {1000*walltime/N:.1f} ms/kf)", flush=True)

    # Save sibling .npz. Separate fields rather than a pickled dict so it's
    # easy to inspect with `np.load(...).files` from any consumer.
    # Provenance fields (CLAUDE.md §1, §6): record the EFFECTIVE thresholds
    # that the encoder applied internally, not just the CLI flag — so a
    # downstream re-thresholder cannot accidentally undo a hidden floor.
    # - encoder.prediction_thresh: 0.0 by default; 0.05 when sam3=True
    #   (radseg_encoder.py:410-411)
    # - sam3_score_threshold: 0.7 (sam3_refinement.py:43, our tightened value)
    sam3_active = (not args.no_sam_refinement) and ("v4" in args.radio_version)
    encoder_pred_thresh = 0.05 if sam3_active else 0.0
    sam3_score_thresh = 0.7 if sam3_active else float("nan")

    np.savez(
        out_path,
        best_query=best_query,
        best_score=best_score,
        queries=np.array(list(args.queries)),
        threshold=np.float32(args.threshold),
        encoder_pred_thresh=np.float32(encoder_pred_thresh),
        sam3_score_thresh=np.float32(sam3_score_thresh),
        radio_version=np.array(args.radio_version),
        lang_adaptor=np.array(grounder.lang_adaptor_name),
        sam_refinement=np.bool_(not args.no_sam_refinement),
        n_keyframes=np.int64(N),
        n_keyframes_in_video=np.int64(N_full),
        image_hw=np.asarray([H, W], dtype=np.int64),
        kf_indices=np.arange(N, dtype=np.int64),  # alignment-safe row map to video.npz
        n_assigned_per_kf=np.asarray(n_dyn_per_kf, dtype=np.int64),
        walltime_seconds=np.float32(walltime),
    )
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[save] {out_path}  ({size_mb:.1f} MB)", flush=True)

    # Sanity: per-query coverage table.
    print(f"\n=== per-query assignment coverage (averaged over {N} keyframes) ===",
          flush=True)
    for qi, q in enumerate(args.queries):
        frac = float((best_query == qi).mean()) * 100.0
        print(f"  [{qi}] {q:<20} : {frac:5.2f}% pixels assigned", flush=True)
    frac_unassigned = float((best_query == -1).mean()) * 100.0
    print(f"  [-1] (below threshold) : {frac_unassigned:5.2f}% pixels", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
