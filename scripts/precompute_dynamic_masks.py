"""
Goal C — precompute per-frame dynamic-object masks for DROID-W.

Iterates the SAME dataset stream the tracker would (so frame indexing
matches `tstamp == i` from `tracker.py:60` + `datasets.py:230`), runs
RADIO + SigLIP-2 + SAM-3 grounding, and writes a single
`<output>/dynamic_masks.npz` with key `mask` shape `(N_frames, H, W)`
boolean (True = STATIC, False = DYNAMIC).

The convention is the MegaSaM (Li et al., CVPR 2025, Eq. 2) one — the
mask gets multiplied INTO the BA-cost weight at SLAM time
(`src/depth_video.py:ba()`), so 1 ⇒ keep, 0 ⇒ down-weight to ε. Inverted
relative to "this is a person" because the BA wants `m * w` where m
attenuates dynamic regions.

Usage:
    python scripts/precompute_dynamic_masks.py \\
        --config configs/TUM_RGBD/freiburg3_walking_static.yaml \\
        --queries person \\
        --threshold 0.40 \\
        --radio-version c-radio_v4-h

Outputs to `<data.input_folder>/dynamic_masks.npz` so the SLAM-side
loader (`slam.py:_maybe_load_semantic_masks`) finds it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

# CLAUDE.md §0 — pin determinism so re-runs produce bit-identical masks.
# Required by Reviewer A audit (Report 17): mirrors the radseg-features pipeline
# (precompute_radseg_features.py:45-56). CUBLAS_WORKSPACE_CONFIG must be set
# BEFORE torch is imported so cuBLAS bmm in SCRA/SCGA becomes deterministic.
# See https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass

# Allow `python scripts/...` from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src import config as droid_config
from src.utils.datasets import get_dataset
from src.utils.mono_priors.radio_grounding import RadioGrounder


def main() -> int:
    p = argparse.ArgumentParser(description="Goal C precompute: per-frame dynamic mask.")
    p.add_argument("--config", required=True, type=Path,
                   help="Scene config (e.g. configs/TUM_RGBD/freiburg3_walking_static.yaml)")
    p.add_argument("--queries", nargs="+", default=None,
                   help="Text queries marking DYNAMIC pixels. Default: cfg.tracking.semantic_mask.queries")
    p.add_argument("--threshold", type=float, default=None,
                   help="Per-pixel softmax threshold for marking dynamic. "
                        "Default: cfg.tracking.semantic_mask.sim_threshold")
    p.add_argument("--score-field", choices=["softmax", "similarity"], default="softmax",
                   help="Use raw cosine ('similarity') or softmax-over-queries ('softmax'). "
                        "softmax is more stable across scenes.")
    p.add_argument("--dilate-px", type=int, default=2,
                   help="Dilation radius applied to the dynamic mask before saving. "
                        "Helps tolerate SAM-3 silhouette error (per DynaSLAM/NGD-SLAM).")
    p.add_argument("--radio-version", type=str, default="c-radio_v4-h",
                   help="RADIO checkpoint version.")
    p.add_argument("--lang-adaptor", type=str, default=None,
                   help="Language adaptor name. Auto-detected from --radio-version when omitted.")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output", type=Path, default=None,
                   help="Output .npz path. Default: <data.input_folder>/dynamic_masks.npz")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Optional cap on dataset length (debug).")
    args = p.parse_args()

    # Use DROID-W's canonical config loader so `inherit_from` chains and
    # `ROOT_FOLDER_PLACEHOLDER` substitution match SLAM-time exactly.
    cfg = droid_config.load_config(str(args.config))

    # Resolve queries + threshold from CLI args or config defaults.
    sm_cfg = cfg.get("tracking", {}).get("semantic_mask", {})
    queries: List[str] = args.queries or sm_cfg.get("queries", ["person"])
    threshold: float = args.threshold if args.threshold is not None \
        else float(sm_cfg.get("sim_threshold", 0.40))

    # Output location — defaults next to the dataset so SLAM finds it.
    # Mirror BaseDataset's ROOT_FOLDER_PLACEHOLDER substitution
    # (datasets.py:110-112) — otherwise we'd save to a literal
    # "ROOT_FOLDER_PLACEHOLDER" subdir.
    if args.output is None:
        input_folder = cfg["data"]["input_folder"]
        if "ROOT_FOLDER_PLACEHOLDER" in input_folder:
            input_folder = input_folder.replace(
                "ROOT_FOLDER_PLACEHOLDER", cfg["data"]["root_folder"]
            )
        out_path = Path(input_folder) / sm_cfg.get("path", "dynamic_masks.npz")
    else:
        out_path = args.output
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[setup] config={args.config}", flush=True)
    print(f"[setup] queries={queries}  threshold={threshold}  field={args.score_field}", flush=True)
    print(f"[setup] dilate_px={args.dilate_px}  radio={args.radio_version}", flush=True)
    print(f"[setup] output={out_path}", flush=True)

    # Build the dataset stream EXACTLY like SLAM does (run.py:55).
    stream = get_dataset(cfg, device=args.device)
    n_frames = len(stream) if args.max_frames is None else min(args.max_frames, len(stream))

    # Grounder load (heavy — ~3 GB GPU for c-radio_v4-h).
    t0 = time.time()
    grounder = RadioGrounder(
        device=args.device,
        radio_version=args.radio_version,
        lang_adaptor=args.lang_adaptor,
    )
    grounder.set_queries(queries)
    print(f"[load] RadioGrounder ready in {time.time() - t0:.1f}s "
          f"(patch={grounder.patch_size}, lang_adaptor={grounder.lang_adaptor_name})",
          flush=True)

    # Probe shape from frame 0. DROID-W's BaseDataset returns color as
    # (1, 3, H, W) — leading batch dim from cv2 imread → torch conversion.
    _, color_0, _, _ = stream[0]
    if color_0.dim() == 4 and color_0.shape[0] == 1:
        color_0 = color_0[0]
    if color_0.dim() != 3 or color_0.shape[0] != 3:
        raise RuntimeError(f"Unexpected color_data shape {color_0.shape}")
    H, W = int(color_0.shape[1]), int(color_0.shape[2])
    print(f"[setup] dataset N_frames={n_frames}  image (H,W)=({H},{W})", flush=True)

    # Allocate output. Float16 saves 4× over uint8/bool wasn't viable because
    # SLAM-side does `m.clamp_min(eps)` * weight on float32 — bool/uint8 work
    # but we lose post-hoc thresholding flexibility. Keep bool for now; the
    # SLAM loader casts to float at load time.
    masks_static = np.ones((n_frames, H, W), dtype=bool)  # 1 = static (default)

    # Optional dilation kernel (binary) — done per-frame post-grounding.
    dil = args.dilate_px
    if dil > 0:
        import cv2  # localized import — only needed if dilating
        kernel = np.ones((2 * dil + 1, 2 * dil + 1), dtype=np.uint8)
    else:
        cv2 = None
        kernel = None

    n_dyn_total = 0
    n_dyn_per_frame = []
    print(f"[run] grounding {n_frames} frames against {len(queries)} quer{'y' if len(queries)==1 else 'ies'} ...",
          flush=True)
    t_loop = time.time()
    for i in range(n_frames):
        _, color_data, _, _ = stream[i]
        # color_data: torch float32 (1, 3, H, W) in [0, 1] post-edge-crop.
        # Drop the batch dim then permute to (H, W, 3) uint8 RGB for grounder.
        if color_data.dim() == 4 and color_data.shape[0] == 1:
            color_data = color_data[0]
        rgb = (color_data.permute(1, 2, 0).clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
        result = grounder.segment(rgb)
        score = result[args.score_field]      # (Q, H, W) float
        best_q = result["best_query"]         # (H, W) int
        best_score = result["best_score"]     # (H, W) float

        # A pixel is DYNAMIC iff `best_score >= threshold`. RadioGrounder's
        # `best_score` is the per-pixel softmax (after temperature-100 + prompt
        # denoising + SAM-3 score-thresholding) of the argmax query in our
        # `set_queries(queries)` list. Pixels below threshold or that fall to
        # the implicit ignore class have `best_score == 0` (radio_grounding.py
        # writes `sim_full[0]` = zeros for the ignore class), so the threshold
        # comparison alone is sufficient — no separate `best_query >= 0` check.
        dyn = best_score >= threshold

        if cv2 is not None and dyn.any():
            dyn = cv2.dilate(dyn.astype(np.uint8), kernel, iterations=1).astype(bool)

        masks_static[i] = ~dyn
        n_dyn_frame = int(dyn.sum())
        n_dyn_total += n_dyn_frame
        n_dyn_per_frame.append(n_dyn_frame)

        if i % 10 == 0 or i == n_frames - 1:
            dt = time.time() - t_loop
            eta = dt / max(1, i + 1) * (n_frames - i - 1)
            print(f"  frame {i:4d}/{n_frames}  dyn_pct={100.0*n_dyn_frame/(H*W):5.2f}%  "
                  f"({dt:5.1f}s elapsed, ~{eta:5.1f}s ETA)", flush=True)

    total_px = n_frames * H * W
    pct_dyn = 100.0 * n_dyn_total / max(1, total_px)
    print(f"[done] grounded {n_frames} frames in {time.time() - t_loop:.1f}s. "
          f"Aggregate dynamic coverage = {pct_dyn:.2f}% of pixels.", flush=True)

    np.savez(
        out_path,
        mask=masks_static,
        queries=np.array(queries),
        threshold=np.float32(threshold),
        score_field=np.array(args.score_field),
        dilate_px=np.int32(args.dilate_px),
        radio_version=np.array(args.radio_version),
        n_dyn_per_frame=np.asarray(n_dyn_per_frame, dtype=np.int64),
    )
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[save] {out_path}  ({size_mb:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
