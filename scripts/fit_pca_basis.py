"""
B.0.5 — fit a frozen PCA-256 basis on Phase B' lang-aligned RADIO features.

Mirrors RADIO-ViPE's `pca_component_calc.py` schema (`{mean, components,
metadata}`) so a ViPE-compatible eval harness can load our basis directly.

Pipeline:
  1. Load features from `radseg_features.npz` for the configured sources.
     DEFAULT: freiburg3_walking_static + tokyo_walking1 (indoor + outdoor).
     Replica scenes are NOT included by default — they don't have Phase B'
     features yet (B.1 will produce them). Plan B B.5 should re-fit per
     scene at eval entry to hedge against domain-shift drift.
  2. Pool feature vectors over all KFs and spatial positions.
  3. Sample N_SAMPLES (default 50k per source = 100k total) random vectors.
     This satisfies n/D ≈ 65 (HTF rule of thumb is n ≥ 100·D = 153k for very
     reliable PCA; 65 is sufficient given the 96.5% variance plateau).
  4. Fit PCA via numpy SVD on the centered sample matrix.
  5. Save the basis as a torch .pt with the RADIO-ViPE-compatible schema.

D = 1536 for v4-h + siglip2-g. target_dim = 256 (RADIO-ViPE canonical).

DOMAIN COVERAGE CAVEAT (verified empirically with the new strict p1 gate):
  • freiburg3 (indoor): PASS at p1 ≥ 0.955 — well within paper-grade gate.
  • tokyo_walking1 (outdoor): FAILS p1 at 5/104 KFs at p1=0.90-0.92 — signals
    that the 50/50 indoor/outdoor pool under-represents the outdoor manifold
    even at 100k samples.
  Plan B's eval scope is Replica + freiburg3 (both indoor), so the basis is
  fit-for-purpose. Tokyo is out-of-scope for paper-grade numbers.

  At B.5 entry (per Plan), re-verify on each Replica scene; if any FAILs the
  p1 gate, re-fit the basis with that scene included before reporting numbers.

Reviewer A audit (Report 18 B.0.5) hardenings:
  #2  Missing-source paths now raise SystemExit (was silent [WARN] skip).
  #3  Schema includes a sibling `metadata` dict for RADIO-ViPE
      backwards-compat with their `state["metadata"]["target_dim"]` access.
  #5  Default n-per-source bumped 25k → 50k (was below HTF rule).

Out of scope: incremental PCA, learned compression, per-scene basis at fit
time. Plan B uses a single shared basis (mirrors RADIO-ViPE); B.5 re-checks
on each Replica scene before reporting numbers.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

torch.backends.cudnn.deterministic = True

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Default sources of Phase B' features. Each entry: (label, path-to-radseg_features.npz).
DEFAULT_SOURCES = [
    ("freiburg3_walking_static", "Outputs/TUM_RGBD/freiburg3_walking_static/radseg_features.npz"),
    ("tokyo_walking1", "Outputs/YouTube/tokyo_walking1/radseg_features.npz"),
]


def _sample_features(sources: list[tuple[str, Path]], n_per_source: int, seed: int,
                     features_key: str = "lang_aligned_feats") -> tuple[np.ndarray, dict]:
    """Pool random feature vectors across the given sources.

    Returns:
      X: (n_total, D) float32 array.
      meta: dict with per-source counts + the (D, h, w) shape sanity.
    """
    rng = np.random.default_rng(seed)
    X_chunks = []
    meta = {"sources": [], "D": None, "features_key": features_key}
    for label, path in sources:
        if not Path(path).exists():
            # Reviewer A audit fix #2: silent skip on missing source could
            # quietly produce a 1-source basis (or worse, an empty one).
            # Hard fail per CLAUDE.md §6.
            raise SystemExit(
                f"[FAIL] B.0.5: source '{label}' missing at {path}. "
                f"Either remove from --sources or precompute its Phase B' "
                f"features first."
            )
        m = np.load(path)
        if features_key not in m.files:
            raise SystemExit(
                f"[FAIL] B.0.5: source '{label}' at {path} lacks "
                f"'{features_key}' field. Available: {list(m.files)}. "
                f"Re-precompute with the updated precompute_radseg_features.py "
                f"(saves both lang_aligned_feats and encoder_feats)."
            )
        feats = m[features_key]            # (N, D, h, w) fp16
        N, D, h, w = feats.shape
        if meta["D"] is None:
            meta["D"] = int(D)
        elif meta["D"] != int(D):
            raise RuntimeError(
                f"feature-dim mismatch: {label} has D={D} but expected {meta['D']} — "
                "PCA basis must be per-(radio, lang_adaptor) pair, not global"
            )
        # Reshape to (N*h*w, D) and sample n_per_source rows.
        flat = feats.transpose(0, 2, 3, 1).reshape(-1, D).astype(np.float32)
        n_pixels = flat.shape[0]
        n_take = min(n_per_source, n_pixels)
        idx = rng.choice(n_pixels, size=n_take, replace=False)
        sample = flat[idx]
        X_chunks.append(sample)
        meta["sources"].append({"label": label, "path": str(path), "N_kf": int(N),
                                "h": int(h), "w": int(w), "n_pixels_total": int(n_pixels),
                                "n_sampled": int(n_take)})
        print(f"  [{label}] sampled {n_take:,} / {n_pixels:,} feat vectors  D={D}",
              flush=True)
        del feats, flat, m
    if not X_chunks:
        raise RuntimeError("no feature sources found — check paths")
    X = np.concatenate(X_chunks, axis=0)
    print(f"[sample] pooled X shape={X.shape}  size={X.nbytes/1024/1024:.1f} MB", flush=True)
    return X, meta


def _fit_pca(X: np.ndarray, target_dim: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit PCA via SVD on the centered sample matrix.

    Returns:
      mean: (D,) float32
      components: (target_dim, D) float32 — orthonormal rows.
      explained_var_ratio: cumulative variance retained by the top target_dim.
    """
    mean = X.mean(axis=0)
    Xc = X - mean
    # Use SVD on (n, D). For n=50k, D=1536: cost ~ O(n*D^2) ~ ~120 GFLOPs, fine on CPU.
    print(f"[fit] SVD on {Xc.shape} — this takes ~30s on CPU ...", flush=True)
    t0 = time.time()
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    print(f"[fit] SVD done in {time.time() - t0:.1f}s", flush=True)
    components = Vt[:target_dim]                            # (target_dim, D)
    var_total = (S ** 2).sum()
    var_top = (S[:target_dim] ** 2).sum()
    return mean.astype(np.float32), components.astype(np.float32), float(var_top / var_total)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target-dim", default=256, type=int,
                   help="PCA dim (RADIO-ViPE uses 256).")
    p.add_argument("--n-per-source", default=50_000, type=int,
                   help="Random pixels sampled per scene (100k total at 2 sources). "
                        "Reviewer A bumped from 25k - HTF rule of thumb is n >= 100*D = 153k; "
                        "we accept n/D ~= 65 given the 96.5%% variance plateau.")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--out", default="weights/pca_basis.pt", type=str)
    p.add_argument("--sources", default=None, nargs="+",
                   help="Override default sources (label=path label2=path2 ...).")
    p.add_argument("--features-key", default="lang_aligned_feats",
                   choices=["lang_aligned_feats", "encoder_feats"],
                   help="Which feature head to PCA. 'lang_aligned_feats' for "
                        "the open-vocab text query path (Plan B v5); "
                        "'encoder_feats' for the BA-side path per RADIO-ViPE "
                        "Sec III-B (preserves geometric content from DINOv2 + "
                        "SAM teachers).")
    args = p.parse_args()

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sources: list[tuple[str, Path]]
    if args.sources is None:
        sources = [(label, REPO_ROOT / path) for label, path in DEFAULT_SOURCES]
    else:
        sources = []
        for token in args.sources:
            if "=" not in token:
                raise SystemExit(f"--sources entries must be label=path (got {token})")
            label, path = token.split("=", 1)
            sources.append((label, Path(path)))

    print(f"[setup] target_dim={args.target_dim}  n_per_source={args.n_per_source:,}",
          flush=True)
    print(f"[setup] sources:", flush=True)
    for label, path in sources:
        print(f"  {label:30s}  {path}  exists={Path(path).exists()}", flush=True)

    X, meta = _sample_features(sources, args.n_per_source, args.seed,
                               features_key=args.features_key)
    mean, components, expl = _fit_pca(X, args.target_dim)
    print(f"[fit] components shape={components.shape}  variance_explained_top{args.target_dim}={expl*100:.2f}%",
          flush=True)

    # Save. Schema sibling-compatible with RADIO-ViPE (Reviewer A audit #3):
    # their loader reads `state["metadata"]["target_dim"]` and similar; we
    # provide both the metadata sub-dict AND our top-level richer keys.
    state = {
        "mean": torch.from_numpy(mean),
        "components": torch.from_numpy(components),
        "metadata": {
            # RADIO-ViPE-compatible sub-keys.
            "target_dim": int(args.target_dim),
            "samples": int(X.shape[0]),
            "feature_dim": int(meta["D"]),
        },
        # HERMES-SLAM extras (top-level for our own code).
        "feature_dim": int(meta["D"]),
        "target_dim": int(args.target_dim),
        "n_samples_total": int(X.shape[0]),
        "fit_variance_explained": float(expl),
        "sources_meta": meta["sources"],
        "seed": int(args.seed),
        "schema": "RADIO-ViPE-compatible (HERMES-SLAM Report 18 B.0.5)",
    }
    torch.save(state, str(out_path))
    size_kb = out_path.stat().st_size / 1024
    print(f"[save] {out_path}  ({size_kb:.1f} KB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
