"""
Phase B' — persist per-keyframe LANGUAGE-ALIGNED RADIO features (no SAM,
no class baking), so any text query can be answered later in milliseconds
without re-running RADIO.

Workflow this enables:
    1. Run THIS script ONCE per scene (~5 min on 5090 at v4-h, no SAM).
    2. Use scripts/query_radseg_features.py interactively with any text:
       it runs only the SigLIP-2 text encoder + cosine + softmax — no
       RADIO forward pass needed.

Compared to scripts/precompute_radseg_masks.py (Phase B):
    * Phase B  saves ARGMAX (62 MB / scene) — class set baked, fast to read,
      requires RADIO+SAM-3 re-run to swap classes.
    * Phase B' saves LANG-ALIGNED FEATURES (~150 MB / scene at v4-h) —
      class-agnostic, requires only text-encode+cosine to query.

Storage (fp16, native feat-grid resolution from sliding-window aggregation):
    freiburg3 (384x512) on v4-h+siglip2-g → grid 42x42 (sliding-window
        round-up at v4-h preprocessing), dim 1536 (siglip2-g lang head)
        ≈ 5.17 MB / KF × 83 KFs ≈ 429 MB raw / 450 MB on-disk.
    For v3-b+siglip2: grid 24x32, dim 1152 → ~1.77 MB / KF.
    PCA-256 (RADIO-ViPE's compression) brings v4-h to ~72 MB total —
    not implemented here; Phase C optimization if cosine quality holds.

Per CLAUDE.md §1 + §6 we record provenance (radio_version, lang_adaptor,
slide_crop / slide_stride, scra/scga scaling) so a future consumer can
reproduce the exact feature space.

Usage (cvg):
    python scripts/precompute_radseg_features.py \\
        --scene Outputs/TUM_RGBD/freiburg3_walking_static
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# CLAUDE.md §0 — pin determinism so a re-run produces bit-identical
# features. Required for the bit-exact verifier (verify_radseg_features.py).
# CUBLAS_WORKSPACE_CONFIG MUST be set before torch is imported so that
# cuBLAS bmm in SCRA/SCGA (radseg_encoder.py:130-133, 498-502) becomes
# deterministic. See https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

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
    p.add_argument("--radio-version", default="c-radio_v4-h", type=str)
    p.add_argument("--lang-adaptor", default=None, type=str,
                   help="Auto-detected from --radio-version when omitted "
                        "(siglip2-g for v4, siglip2 for v3).")
    p.add_argument("--scra-scaling", default=10.0, type=float)
    p.add_argument("--scga-scaling", default=10.0, type=float)
    p.add_argument("--slide-crop", default=336, type=int)
    p.add_argument("--slide-stride", default=224, type=int)
    p.add_argument("--device", default="cuda:0", type=str)
    p.add_argument("--output", default=None, type=Path,
                   help="Default: <scene>/radseg_features.npz")
    p.add_argument("--max-keyframes", default=None, type=int)
    p.add_argument("--prompt-denoising-thresh-default", default=0.5, type=float,
                   help="Default prompt-denoising threshold to RECORD in metadata "
                        "(query_radseg_features.py applies its own at query time).")
    args = p.parse_args()

    scene = args.scene.resolve()
    npz_path = scene / "video.npz"
    if not npz_path.exists():
        print(f"[ERR] {npz_path} not found", flush=True)
        return 1
    out_path = args.output.resolve() if args.output is not None else scene / "radseg_features.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    is_v4 = "v4" in args.radio_version.lower()
    if args.lang_adaptor is None:
        lang_adaptor = "siglip2-g" if is_v4 else "siglip2"
    else:
        lang_adaptor = args.lang_adaptor

    print(f"[setup] scene={scene}", flush=True)
    print(f"[setup] radio={args.radio_version}  lang_adaptor={lang_adaptor}", flush=True)
    print(f"[setup] output={out_path}", flush=True)

    z = np.load(npz_path)
    if "images" not in z.files:
        print(f"[ERR] {npz_path} has no 'images' field", flush=True)
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
            print(f"[WARN] features: --max-keyframes={args.max_keyframes} >= "
                  f"N_full={N_full}; fallback=using all keyframes", flush=True)
            N = N_full
        else:
            N = args.max_keyframes
    else:
        N = N_full
    images = images[:N]
    print(f"[setup] N_keyframes={N}/{N_full}  image (H,W)=({H},{W})", flush=True)

    # Build encoder in feature-extraction-only mode (no SAM, no class
    # baking, no prompt denoising).
    t_load = time.time()
    encoder = RADSegEncoder(
        device=args.device,
        model_version=args.radio_version,
        lang_model=lang_adaptor,
        return_radio_features=True,
        compile=False,
        amp=False,                # CLAUDE.md §0 — determinism
        predict=False,            # no class prediction
        sam_refinement=False,     # no SAM
        sam3=is_v4,
        sam_ckpt="",
        scra_scaling=args.scra_scaling,
        scga_scaling=args.scga_scaling,
        slide_crop=args.slide_crop,
        slide_stride=args.slide_stride,
    )
    print(f"[load] encoder ready in {time.time() - t_load:.1f}s "
          f"(patch={encoder.model.patch_size})", flush=True)

    # Probe one keyframe to learn the feature grid + lang-aligned dim.
    img0 = torch.from_numpy(images[0]).float() / 255.0  # (3, H, W)
    img0 = img0.unsqueeze(0).to(args.device)
    with torch.no_grad():
        feat0 = encoder.encode_image_to_feat_map(img0)             # (1, C_radio, h, w)
        aligned0 = encoder.align_spatial_features_with_language(feat0, onehot=False)
    _, D, hp, wp = aligned0.shape
    print(f"[setup] feat grid (h,w)=({hp},{wp})  lang-aligned dim D={D}", flush=True)

    # Allocate output buffer on CPU. fp16 is enough for cosine quality
    # (we'll re-normalize at query time).
    feats_cpu = np.zeros((N, D, hp, wp), dtype=np.float16)

    t_loop = time.time()
    for i in range(N):
        img = torch.from_numpy(images[i]).float() / 255.0
        img = img.unsqueeze(0).to(args.device)
        with torch.no_grad():
            f = encoder.encode_image_to_feat_map(img)              # (1, C_radio, h, w)
            a = encoder.align_spatial_features_with_language(f, onehot=False)
        feats_cpu[i] = a.squeeze(0).cpu().to(torch.float16).numpy()
        del img, f, a
        if (i + 1) % 10 == 0 or i == 0 or i == N - 1:
            elapsed = time.time() - t_loop
            eta = elapsed / max(1, i + 1) * (N - i - 1)
            print(f"  [{i+1:4d}/{N}]  ({elapsed:5.1f}s, ~{eta:5.1f}s ETA)", flush=True)
        torch.cuda.empty_cache() if (i + 1) % 50 == 0 else None

    walltime = time.time() - t_loop
    print(f"[done] encoded {N} keyframes in {walltime:.1f}s "
          f"(mean {1000*walltime/N:.1f} ms/kf)", flush=True)

    np.savez(
        out_path,
        lang_aligned_feats=feats_cpu,             # (N, D, hp, wp) fp16
        radio_version=np.array(args.radio_version),
        lang_adaptor=np.array(lang_adaptor),
        scra_scaling=np.float32(args.scra_scaling),
        scga_scaling=np.float32(args.scga_scaling),
        slide_crop=np.int64(args.slide_crop),
        slide_stride=np.int64(args.slide_stride),
        prompt_denoising_thresh_default=np.float32(args.prompt_denoising_thresh_default),
        feature_dim=np.int64(D),
        n_keyframes=np.int64(N),
        n_keyframes_in_video=np.int64(N_full),
        image_hw=np.asarray([H, W], dtype=np.int64),
        feat_hw=np.asarray([hp, wp], dtype=np.int64),
        kf_indices=np.arange(N, dtype=np.int64),
        walltime_seconds=np.float32(walltime),
    )
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[save] {out_path}  ({size_mb:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
