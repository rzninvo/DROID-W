"""
B.0.5 verifier — strict roundtrip cosine gate on the frozen PCA-256 basis.

Loads the basis fit by `fit_pca_basis.py` and tests that every Phase B'
feature, when projected to 256-d and reconstructed, has high cosine
similarity to the original. Failure means the 256-d subspace doesn't
capture the relevant variance for cosine-vs-text-embedding queries — the
downstream open-vocab queries (B.4) would silently degrade.

Strict gate: per-keyframe MEAN cosine similarity ≥ 0.985 across all
spatial pixels. Fails if any keyframe drops below.

This catches:
  • PCA fit on too few / unrepresentative samples.
  • PCA fit on a different feature distribution (wrong scenes pooled).
  • Numerical issues in encode/decode formula.

Encode:  e = (F - mean) @ components.T              # (D,) → (target_dim,)
Decode:  F_hat = e @ components + mean              # (target_dim,) → (D,)

NOT TESTED HERE: that the basis generalizes to UNSEEN scenes (Replica office_*,
ScanNet, KITTI). Plan B uses a basis fit only on freiburg3+tokyo+room0 and
must re-verify per scene at B.5 entry.

Usage (cvg):
    python scripts/verify_pca_roundtrip.py \\
        --basis weights/pca_basis.pt \\
        --features Outputs/TUM_RGBD/freiburg3_walking_static/radseg_features.npz
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _encode_decode(feats: np.ndarray, mean: np.ndarray,
                   components: np.ndarray) -> np.ndarray:
    """Project to PCA-256 then back. feats is (..., D), returns same shape."""
    flat_shape = feats.shape
    D = flat_shape[-1]
    flat = feats.reshape(-1, D).astype(np.float32)
    centered = flat - mean
    e = centered @ components.T                                        # (n, target_dim)
    decoded = e @ components + mean                                    # (n, D)
    return decoded.reshape(flat_shape).astype(np.float32)


def _per_kf_mean_cosine(feats_orig: np.ndarray, feats_decoded: np.ndarray) -> np.ndarray:
    """Per-KF mean cosine similarity over all spatial positions.

    feats_*: (N, D, h, w) float32. Returns (N,) mean cosines.
    """
    N, D, h, w = feats_orig.shape
    a = torch.from_numpy(feats_orig).reshape(N, D, h * w).permute(0, 2, 1)  # (N, h*w, D)
    b = torch.from_numpy(feats_decoded).reshape(N, D, h * w).permute(0, 2, 1)
    a = F.normalize(a, dim=-1, eps=1e-8)
    b = F.normalize(b, dim=-1, eps=1e-8)
    cos = (a * b).sum(dim=-1)                                          # (N, h*w)
    return cos.mean(dim=-1).numpy()                                    # (N,)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--basis", required=True, type=str,
                   help="Path to pca_basis.pt from fit_pca_basis.py.")
    p.add_argument("--features", required=True, type=str,
                   help="Path to radseg_features.npz to test.")
    p.add_argument("--gate-mean-cosine", default=0.985, type=float,
                   help="Per-KF mean cosine gate (default 0.985, paper-grade).")
    p.add_argument("--gate-p1-cosine", default=0.93, type=float,
                   help="Per-KF 1st-percentile cosine HARD gate (Reviewer A #4: "
                        "min-only is fragile to single bad pixels; p1 catches "
                        "systematic worst-1%% drift while ignoring isolated outliers).")
    p.add_argument("--gate-min-cosine", default=0.85, type=float,
                   help="Per-KF MIN cosine soft floor — informational warn.")
    args = p.parse_args()

    basis_path = REPO_ROOT / args.basis if not Path(args.basis).is_absolute() else Path(args.basis)
    feat_path = REPO_ROOT / args.features if not Path(args.features).is_absolute() else Path(args.features)

    print(f"[setup] basis={basis_path}", flush=True)
    print(f"[setup] features={feat_path}", flush=True)

    state = torch.load(str(basis_path), map_location="cpu", weights_only=False)
    mean = state["mean"].numpy().astype(np.float32)            # (D,)
    components = state["components"].numpy().astype(np.float32) # (target_dim, D)
    D_basis = int(state["feature_dim"])
    target_dim = int(state["target_dim"])
    print(f"[basis] D={D_basis}  target_dim={target_dim}  "
          f"variance_explained={state['fit_variance_explained']*100:.2f}%  "
          f"n_samples={state['n_samples_total']}", flush=True)

    m = np.load(feat_path)
    feats = m["lang_aligned_feats"]                            # (N, D, h, w) fp16
    N, D, h, w = feats.shape
    if D != D_basis:
        print(f"[FAIL] feature D={D} != basis D={D_basis}; basis incompatible", flush=True)
        return 1
    print(f"[setup] features shape={feats.shape}", flush=True)

    # Move (N, D, h, w) → (N, h, w, D) for encode_decode (which expects D last).
    feats_f32 = feats.astype(np.float32).transpose(0, 2, 3, 1)        # (N, h, w, D)
    decoded = _encode_decode(feats_f32, mean, components)             # (N, h, w, D)
    feats_back = feats_f32.transpose(0, 3, 1, 2)                      # (N, D, h, w)
    decoded_back = decoded.transpose(0, 3, 1, 2)                      # (N, D, h, w)

    cos_per_kf = _per_kf_mean_cosine(feats_back, decoded_back)         # (N,)
    # Also compute min cosine per KF (worst pixel).
    a = torch.from_numpy(feats_back).reshape(N, D, h * w).permute(0, 2, 1)
    b = torch.from_numpy(decoded_back).reshape(N, D, h * w).permute(0, 2, 1)
    a = F.normalize(a, dim=-1, eps=1e-8)
    b = F.normalize(b, dim=-1, eps=1e-8)
    cos_full = (a * b).sum(dim=-1)                                     # (N, h*w)
    min_cos_per_kf = cos_full.min(dim=-1).values.numpy()
    p1_cos_per_kf = np.percentile(cos_full.numpy(), 1, axis=-1)

    fail = False
    n_under_mean = int((cos_per_kf < args.gate_mean_cosine).sum())
    n_under_p1 = int((p1_cos_per_kf < args.gate_p1_cosine).sum())
    n_under_min = int((min_cos_per_kf < args.gate_min_cosine).sum())

    print(f"\n=== per-KF cosine summary (N={N} keyframes) ===", flush=True)
    print(f"  mean cosine   min={cos_per_kf.min():.5f}  median={np.median(cos_per_kf):.5f}  "
          f"max={cos_per_kf.max():.5f}", flush=True)
    print(f"  p1   cosine   min={p1_cos_per_kf.min():.5f}  median={np.median(p1_cos_per_kf):.5f}", flush=True)
    print(f"  min  cosine   min={min_cos_per_kf.min():.5f}  median={np.median(min_cos_per_kf):.5f}  "
          f"max={min_cos_per_kf.max():.5f}", flush=True)

    if n_under_mean > 0:
        worst = np.argsort(cos_per_kf)[:5]
        print(f"\n[FAIL] {n_under_mean}/{N} KFs have mean cosine < {args.gate_mean_cosine}", flush=True)
        for k in worst:
            print(f"    KF {k:4d}: mean={cos_per_kf[k]:.5f}  p1={p1_cos_per_kf[k]:.5f}  "
                  f"min={min_cos_per_kf[k]:.5f}", flush=True)
        fail = True

    if n_under_p1 > 0:
        # Reviewer A audit #4: p1 is now a HARD gate (was implicit warn).
        worst = np.argsort(p1_cos_per_kf)[:5]
        print(f"\n[FAIL] {n_under_p1}/{N} KFs have p1 cosine < {args.gate_p1_cosine} "
              f"(systematic worst-1% drift)", flush=True)
        for k in worst:
            print(f"    KF {k:4d}: p1={p1_cos_per_kf[k]:.5f}  mean={cos_per_kf[k]:.5f}", flush=True)
        fail = True

    if n_under_min > 0:
        print(f"\n[WARN] {n_under_min}/{N} KFs have MIN-pixel cosine < {args.gate_min_cosine} "
              f"(informational; isolated outlier pixels)", flush=True)

    if fail:
        print(f"\n[FAIL] {feat_path}", flush=True)
        return 1
    print(f"\n[PASS] {feat_path}  (mean cosine ≥ {args.gate_mean_cosine} on every KF)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
