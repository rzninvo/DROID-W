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
    p.add_argument("--scene", default=None, type=Path,
                   help="Scene directory containing video.npz (legacy path; preferred for TUM/YouTube scenes).")
    p.add_argument("--config", default=None, type=str,
                   help="Scene config (NEW; for Replica/ScanNet scenes that "
                        "stream from BaseDataset rather than ship a video.npz).")
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

    if (args.scene is None) == (args.config is None):
        print(f"[ERR] specify exactly one of --scene <dir> or --config <yaml>", flush=True)
        return 1
    if args.scene is not None:
        scene = args.scene.resolve()
        npz_path = scene / "video.npz"
        if not npz_path.exists():
            print(f"[ERR] {npz_path} not found", flush=True)
            return 1
        out_path = args.output.resolve() if args.output is not None else scene / "radseg_features.npz"
        stream_mode = "video.npz"
        cfg = None
    else:
        # Config-driven path: stream from BaseDataset (Replica / ScanNet).
        from src import config as droid_config
        cfg = droid_config.load_config(args.config)
        # Output dir: <data.output>/<scene>/radseg_features.npz
        scene_name = cfg.get("scene", "scene")
        scene = Path(cfg["data"]["output"]) / scene_name
        scene.mkdir(parents=True, exist_ok=True)
        npz_path = None
        out_path = args.output.resolve() if args.output is not None else scene / "radseg_features.npz"
        stream_mode = f"BaseDataset({cfg['dataset']})"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    is_v4 = "v4" in args.radio_version.lower()
    if args.lang_adaptor is None:
        lang_adaptor = "siglip2-g" if is_v4 else "siglip2"
    else:
        lang_adaptor = args.lang_adaptor

    print(f"[setup] scene={scene}", flush=True)
    print(f"[setup] radio={args.radio_version}  lang_adaptor={lang_adaptor}", flush=True)
    print(f"[setup] output={out_path}", flush=True)

    if stream_mode == "video.npz":
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
        # Mapping local-KF → global-frame is identity in legacy-video.npz mode.
        kf_global_indices = np.arange(N_full, dtype=np.int64)
    else:
        # Stream from dataset.
        from src.utils.datasets import get_dataset
        stream = get_dataset(cfg)
        N_full = len(stream)
        if N_full == 0:
            print(f"[ERR] BaseDataset({cfg['dataset']}) returned 0 frames", flush=True)
            return 1
        # Probe first frame for H, W.
        _, color0, _, _ = stream[0]
        if torch.is_tensor(color0):
            color0 = color0.cpu().numpy()
        if color0.ndim == 4:
            color0 = color0[0]
        H, W = color0.shape[1], color0.shape[2]
        # Map local-KF idx → global dataset frame idx (after stride).
        cfg_stride = int(cfg.get("stride", 1)) if cfg else 1
        cfg_max_frames = int(cfg.get("max_frames", -1)) if cfg else -1
        # Build kf_global_indices that mirrors Replica.__init__'s
        #   color_paths[:max_frames][::stride]   (datasets.py:273-275)
        if cfg_max_frames < 0:
            kf_global_indices = np.arange(0, N_full * cfg_stride, cfg_stride, dtype=np.int64)
        else:
            kf_global_indices = np.arange(0, cfg_max_frames, cfg_stride, dtype=np.int64)[:N_full]
        images = None  # streamed lazily below
        print(f"[setup] streaming N_full={N_full} frames at H={H} W={W} from {stream_mode}",
              flush=True)
    if args.max_keyframes is not None:
        if args.max_keyframes >= N_full:
            print(f"[WARN] features: --max-keyframes={args.max_keyframes} >= "
                  f"N_full={N_full}; fallback=using all keyframes", flush=True)
            N = N_full
        else:
            N = args.max_keyframes
    else:
        N = N_full
    if stream_mode == "video.npz":
        images = images[:N]
    # else: streaming path, _get_image_uint8(i) handles range below.
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

    # Helper to fetch image i as (3, H, W) uint8 — works for both modes.
    def _get_image_uint8(i: int) -> np.ndarray:
        if stream_mode == "video.npz":
            return images[i]
        else:
            _, color, _, _ = stream[i]
            if torch.is_tensor(color):
                color = color.cpu().numpy()
            if color.ndim == 4:
                color = color[0]
            return (color * 255.0).clip(0, 255).astype(np.uint8)  # (3, H, W) uint8

    # Probe one keyframe to learn the feature grid + lang-aligned dim.
    img0_np = _get_image_uint8(0)
    img0 = torch.from_numpy(img0_np).float() / 255.0  # (3, H, W)
    img0 = img0.unsqueeze(0).to(args.device)
    with torch.no_grad():
        feat0 = encoder.encode_image_to_feat_map(img0)             # (1, C_radio, h, w)
        aligned0 = encoder.align_spatial_features_with_language(feat0, onehot=False)
    _, D, hp, wp = aligned0.shape
    print(f"[setup] feat grid (h,w)=({hp},{wp})  lang-aligned dim D={D}", flush=True)

    # Allocate output buffer on CPU. fp16 is enough for cosine quality.
    feats_cpu = np.zeros((N, D, hp, wp), dtype=np.float16)

    t_loop = time.time()
    for i in range(N):
        img_np = _get_image_uint8(i)
        img = torch.from_numpy(img_np).float() / 255.0
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
        # kf_indices stores the LOCAL KF position within this scene's
        # `radseg_features.npz` — useful for offsets. The mapping to the
        # original dataset frame index is `kf_global_indices` (added per
        # Reviewer-2 audit, B.0 plan #B0): identity for legacy video.npz
        # mode, true global indices for streaming-from-dataset mode.
        kf_indices=np.arange(N, dtype=np.int64),
        kf_global_indices=kf_global_indices[:N].astype(np.int64),
        walltime_seconds=np.float32(walltime),
    )
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[save] {out_path}  ({size_mb:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
