"""
Step 0.a verifier (plan-v2): feature preservation across BA-grid resampling.

Tests that round-tripping F_native (the encoder's stored grid, e.g. 42x42
for c-radio_v3-b at 336 px) through the DROID-W BA grid (H/8 x W/8) and
back preserves features to mean cosine >= 0.97. Catches resize regressions
before Steps 2/3 attempt to use these features inside BA.

Loads <scene>/radseg_features.npz from Phase B' / Plan B v5 precompute,
picks a held-out keyframe (default: middle index), resamples
    (D, h_native, w_native) -> (D, H//8, W//8) -> (D, h_native, w_native)
via bilinear interpolation, and computes per-pixel cosine on N random
samples.

Gate per plan-v2 Step 0.a: mean cosine >= 0.97 on 1000 random samples.
On failure, emits a [WARN] per CLAUDE.md section 6 (no silent fallback)
and exits non-zero.

Usage (cvg, droid-w conda env):
    python scripts/verify_step0a_feature_preservation.py \\
        --features Outputs/TUM_RGBD/freiburg3_walking_static/radseg_features.npz \\
        --kf-index -1 --n-samples 1000 --ba-stride 8
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_n = a / (a.norm(dim=-1, keepdim=True) + eps)
    b_n = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a_n * b_n).sum(dim=-1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--kf-index", default=-1, type=int,
                   help="Keyframe to test (default: -1 -> middle KF)")
    p.add_argument("--n-samples", default=1000, type=int)
    p.add_argument("--ba-stride", default=8, type=int,
                   help="DROID-W BA grid stride (image dim / stride = BA grid dim)")
    p.add_argument("--gate-mean", default=0.97, type=float)
    p.add_argument("--seed", default=0, type=int)
    args = p.parse_args()

    if not args.features.exists():
        print(f"[ERR] {args.features} not found", flush=True)
        return 2

    d = np.load(args.features, allow_pickle=False)
    if "lang_aligned_feats" not in d.files:
        print(f"[ERR] {args.features} missing 'lang_aligned_feats'", flush=True)
        return 2

    feats = d["lang_aligned_feats"]
    image_hw = tuple(int(x) for x in d["image_hw"])
    feat_hw = tuple(int(x) for x in d["feat_hw"])
    n_kf = int(d["n_keyframes"])

    h_native, w_native = feat_hw
    H_img, W_img = image_hw
    H_ba, W_ba = H_img // args.ba_stride, W_img // args.ba_stride

    print(f"[step0a] features: {args.features}")
    print(f"[step0a] image_hw={image_hw}, feat_hw={feat_hw}, "
          f"BA grid=({H_ba},{W_ba}) at stride={args.ba_stride}")
    print(f"[step0a] n_keyframes={n_kf}, D={feats.shape[1]}, dtype={feats.dtype}")

    kf = args.kf_index if args.kf_index >= 0 else n_kf // 2
    if not (0 <= kf < n_kf):
        print(f"[ERR] kf={kf} out of range [0,{n_kf})", flush=True)
        return 2

    F_native = torch.from_numpy(feats[kf]).float().unsqueeze(0)        # (1, D, h, w)
    print(f"[step0a] testing KF {kf}: shape={tuple(F_native.shape)}")

    F_ba = F.interpolate(F_native, size=(H_ba, W_ba), mode="bilinear",
                         align_corners=False)
    F_rt = F.interpolate(F_ba,     size=(h_native, w_native), mode="bilinear",
                         align_corners=False)

    rng = np.random.default_rng(args.seed)
    n_samples = min(args.n_samples, h_native * w_native)
    idx = rng.choice(h_native * w_native, size=n_samples, replace=False)
    py, px = np.unravel_index(idx, (h_native, w_native))

    a = F_native[0, :, py, px].T                                       # (n_samples, D)
    b = F_rt[0,     :, py, px].T

    cos = _cosine(a, b).cpu().numpy()
    mean_c = float(cos.mean())
    min_c = float(cos.min())
    p5_c = float(np.percentile(cos, 5))
    std_c = float(cos.std())

    print(f"[step0a] roundtrip cosine on n={n_samples} samples:")
    print(f"[step0a]   mean={mean_c:.6f}  min={min_c:.6f}  "
          f"p5={p5_c:.6f}  std={std_c:.6f}")
    print(f"[step0a] gate: mean >= {args.gate_mean}")

    passed = mean_c >= args.gate_mean
    if passed:
        print(f"[step0a] PASS  ({mean_c:.4f} >= {args.gate_mean})")
        return 0

    print(f"[WARN] step0a feature_preservation: expected mean_cos >= "
          f"{args.gate_mean}, got {mean_c:.4f}, fallback=FAIL "
          f"(do not proceed to Steps 2/3 with these features; see plan-v2 Step 0.a)",
          flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
