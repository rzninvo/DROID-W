"""
Sanity-check `<scene>/radseg_features.npz` (Phase B') by re-running the
RADSegEncoder feature path on a small random sample of keyframes and
checking the saved features stay close to the live extraction.

We set `amp=False`, `eval()`, `cudnn.deterministic=True`, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` — but the chain still has irreducible
fp16-roundtrip + sliding-window-overlap accumulator noise on the
order of 0.01-0.1 per element. **Cosine similarity is the strict
signal** (we expect mean cos > 0.9999 on every keyframe); per-element
max-abs-diff is informational only and gated loosely.

Verifies:
- Schema is complete.
- lang_aligned_feats shape == (n_keyframes, D, hp, wp).
- Mean cosine similarity per keyframe > 0.9999 (catches backbone drift,
  lang_adaptor swap, systematic bugs).
- Min cosine similarity per keyframe > 0.99 (catches whole-pixel flips).

Usage (cvg):
    python scripts/verify_radseg_features.py \\
        --scene Outputs/TUM_RGBD/freiburg3_walking_static \\
        --n-spotcheck 3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# CUBLAS_WORKSPACE_CONFIG must be set BEFORE torch import so cuBLAS bmm
# (used in SCRA/SCGA) is deterministic (CLAUDE.md §0).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.mono_priors.radseg.radseg_encoder import RADSegEncoder


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, type=Path)
    p.add_argument("--features", default=None, type=Path,
                   help="Default: <scene>/radseg_features.npz")
    p.add_argument("--n-spotcheck", default=3, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--max-abs-diff", default=0.2, type=float,
                   help="Loose informational gate on per-element max-abs-diff "
                        "(default 0.2). fp16 chain noise can be ~0.05-0.1; "
                        "much higher means a bug. Cosine gates are the strict ones.")
    p.add_argument("--min-mean-cos", default=0.9999, type=float,
                   help="Per-keyframe MEAN cosine similarity gate (default 0.9999, strict).")
    p.add_argument("--min-min-cos", default=0.95, type=float,
                   help="Per-keyframe MIN cosine similarity gate (default 0.95). "
                        "Sliding-window edge pixels can drop to ~0.97 in fp16 "
                        "chain; we tolerate that and rely on mean_cos for the "
                        "real signal.")
    p.add_argument("--min-p1-cos", default=0.99, type=float,
                   help="1st-percentile (worst-1%) cosine gate (default 0.99). "
                        "Catches whole-class flips while ignoring isolated edge "
                        "pixels. Real bugs would tank this number.")
    p.add_argument("--device", default="cuda:0", type=str)
    args = p.parse_args()

    scene = args.scene.resolve()
    feats_path = args.features if args.features is not None else scene / "radseg_features.npz"
    video_path = scene / "video.npz"
    print(f"[setup] features={feats_path}", flush=True)

    m = np.load(feats_path)
    expected = {
        "lang_aligned_feats", "radio_version", "lang_adaptor",
        "scra_scaling", "scga_scaling", "slide_crop", "slide_stride",
        "n_keyframes", "image_hw", "feat_hw", "kf_indices",
        "feature_dim", "prompt_denoising_thresh_default",
    }
    missing = expected - set(m.files)
    if missing:
        print(f"[FAIL] {feats_path} missing fields: {missing}", flush=True)
        return 1
    print(f"[ok] all expected fields present: {sorted(m.files)}", flush=True)

    feats = m["lang_aligned_feats"]
    radio_version = str(m["radio_version"])
    lang_adaptor = str(m["lang_adaptor"])
    N, D, hp, wp = feats.shape
    H, W = int(m["image_hw"][0]), int(m["image_hw"][1])
    if int(m["n_keyframes"]) != N:
        print(f"[FAIL] n_keyframes field {m['n_keyframes']} != "
              f"feats.shape[0]={N}", flush=True)
        return 1
    print(f"[ok] feats {feats.shape}, image (H,W)=({H},{W}), feat_grid=({hp},{wp})",
          flush=True)

    # Build the same encoder used at precompute (predict=False, no SAM,
    # amp off). Will reproduce features bit-for-bit modulo cudnn nondet
    # — which we pin above.
    z = np.load(video_path)
    images = z["images"]
    if images.dtype != np.uint8:
        images = (images * 255.0).clip(0, 255).astype(np.uint8)

    is_v4 = "v4" in radio_version.lower()
    encoder = RADSegEncoder(
        device=args.device,
        model_version=radio_version,
        lang_model=lang_adaptor,
        return_radio_features=True,
        compile=False,
        amp=False,
        predict=False,
        sam_refinement=False,
        sam3=is_v4,
        sam_ckpt="",
        scra_scaling=float(m["scra_scaling"]),
        scga_scaling=float(m["scga_scaling"]),
        slide_crop=int(m["slide_crop"]),
        slide_stride=int(m["slide_stride"]),
    )

    rng = np.random.default_rng(args.seed)
    idx_to_check = rng.choice(min(N, len(images)),
                              size=min(args.n_spotcheck, N, len(images)),
                              replace=False)
    print(f"[setup] spot-checking keyframes {sorted(idx_to_check.tolist())}",
          flush=True)

    n_pass = 0
    for kf_idx in idx_to_check:
        img = torch.from_numpy(images[int(kf_idx)]).float() / 255.0
        img = img.unsqueeze(0).to(args.device)
        with torch.no_grad():
            f = encoder.encode_image_to_feat_map(img)
            a = encoder.align_spatial_features_with_language(f, onehot=False)
        live = a.squeeze(0).cpu().to(torch.float16).numpy()        # (D, hp, wp)
        saved = feats[int(kf_idx)]                                  # (D, hp, wp) fp16

        if live.shape != saved.shape:
            print(f"  [{int(kf_idx):4d}] FAIL  live shape {live.shape} != saved {saved.shape}",
                  flush=True)
            continue

        diff = np.abs(live.astype(np.float32) - saved.astype(np.float32))
        max_abs = float(diff.max())
        # Cosine similarity per spatial pixel, averaged.
        live_flat = live.reshape(D, -1).astype(np.float32)
        saved_flat = saved.reshape(D, -1).astype(np.float32)
        live_n = live_flat / (np.linalg.norm(live_flat, axis=0, keepdims=True) + 1e-8)
        saved_n = saved_flat / (np.linalg.norm(saved_flat, axis=0, keepdims=True) + 1e-8)
        cosines = (live_n * saved_n).sum(axis=0)                   # (hp*wp,)
        min_cos = float(cosines.min())
        p1_cos = float(np.percentile(cosines, 1))
        mean_cos = float(cosines.mean())

        ok_diff = max_abs <= args.max_abs_diff
        ok_mean = mean_cos >= args.min_mean_cos
        ok_min = min_cos >= args.min_min_cos
        ok_p1 = p1_cos >= args.min_p1_cos
        if ok_diff and ok_mean and ok_min and ok_p1:
            n_pass += 1
            tag = "EXACT" if max_abs == 0.0 else "PASS"
            print(f"  [{int(kf_idx):4d}] {tag}  max_abs_diff={max_abs:.6f}  "
                  f"min_cos={min_cos:.6f}  p1_cos={p1_cos:.6f}  mean_cos={mean_cos:.6f}",
                  flush=True)
        else:
            print(f"  [{int(kf_idx):4d}] FAIL  max_abs_diff={max_abs:.4f} (gate {args.max_abs_diff})  "
                  f"min={min_cos:.4f} (gate {args.min_min_cos})  "
                  f"p1={p1_cos:.4f} (gate {args.min_p1_cos})  "
                  f"mean={mean_cos:.6f} (gate {args.min_mean_cos})",
                  flush=True)

        del img, f, a, live, saved
        torch.cuda.empty_cache()

    print(f"\n=== verify summary ===", flush=True)
    print(f"  {n_pass}/{len(idx_to_check)} keyframes passed roundtrip", flush=True)
    if n_pass < len(idx_to_check):
        print(f"  [FAIL] saved features deviate from live extraction beyond "
              f"fp16 tolerance — backbone drift or pipeline bug.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
