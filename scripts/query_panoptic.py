"""
B.4 — open-vocab query over Plan B's track-level scene representation.

For any text vocabulary, returns the top-K matching global tracks per query
plus their per-track point cloud (B.3) for visualization / mesh assignment.

Pipeline:
  1. Load tracks.npz       (B.2): track_emb_pca (G, 256) fp16, scene_id_offset.
  2. Load PCA basis        (B.0.5): mean (D,), components (target_dim, D).
  3. Load instance_volumes (B.3): points (P, 3), track_offsets, point_src_kf.
  4. Decode each track's pca-256 → (D=1536) lang feature.
  5. Encode the user's queries via the SAME 80-template OpenAI ImageNet
     ensemble used at Phase B' precompute (CLAUDE.md §1: identical text-
     feature space; otherwise feature/text embeddings silently mis-align).
  6. Cosine `cos(track_emb_decoded, t_query)`; rank top-K per query.
  7. Save results to a JSON + optional rerun visualization.

POSE / OUTPUT NOTE: track_centroid_w from B.2 and points from B.3 are
in OpenGL c2w world frame (no NICE-SLAM Y/Z flip). Visualizers that
expect OpenCV need to flip Y/Z columns.

Usage:
    python scripts/query_panoptic.py \\
        --tracks Outputs/Replica/room0/tracks.npz \\
        --volumes Outputs/Replica/room0/instance_volumes.npz \\
        --pca-basis weights/pca_basis.pt \\
        --queries person chair monitor table floor wall ceiling \\
        --top-k 5 \\
        --output Outputs/Replica/room0/query_top5.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cudnn.deterministic = True

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.mono_priors.radseg.radseg_encoder import RADSegEncoder


def _build_text_only_encoder(radio_version: str, lang_adaptor: str, device: str) -> RADSegEncoder:
    """Same SigLIP-2 text encoder used by query_radseg_features.py — keeps
    feature/text spaces aligned. Pays the RADIO image-tower load cost
    (~10-15s) but never calls its image path."""
    is_v4 = "v4" in radio_version.lower()
    return RADSegEncoder(
        device=device,
        model_version=radio_version,
        lang_model=lang_adaptor,
        return_radio_features=True,
        compile=False,
        amp=False,
        predict=False,
        sam_refinement=False,
        sam3=is_v4,
        sam_ckpt="",
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", required=True, type=str)
    p.add_argument("--volumes", default=None, type=str,
                   help="Default: <tracks.parent>/instance_volumes.npz")
    p.add_argument("--pca-basis", default="weights/pca_basis.pt", type=str)
    p.add_argument("--queries", required=True, nargs="+",
                   help="One or more text queries.")
    p.add_argument("--top-k", default=5, type=int,
                   help="Top-K tracks per query.")
    p.add_argument("--cos-floor", default=0.04, type=float,
                   help="Drop matches below this raw cosine. Natural range "
                        "for RADIO+SigLIP-2-g on Replica indoor is 0.04-0.15 "
                        "(verified by direct Phase B prime probe on room0 "
                        "features); RADSeg paper applies temp-100 softmax to "
                        "amplify discrimination. Default 0.04 keeps the bottom "
                        "of the natural range.")
    p.add_argument("--softmax-temp", default=100.0, type=float,
                   help="Per-query softmax temperature for the AMPLIFIED "
                        "score. Mirrors RADSeg paper §4 (temp=100). The "
                        "cosine itself is the source of truth; softmax just "
                        "gives a 0-1 confidence for ranking.")
    p.add_argument("--radio-version", default="c-radio_v4-h", type=str)
    p.add_argument("--lang-adaptor", default=None, type=str,
                   help="Auto-detected from radio-version when omitted "
                        "(siglip2-g for v4, siglip2 for v3).")
    p.add_argument("--device", default="cuda:0", type=str)
    p.add_argument("--output", default=None, type=str,
                   help="JSON output path. Default: <tracks.parent>/query_<first>.json")
    args = p.parse_args()

    trk_path = Path(args.tracks)
    if not trk_path.is_absolute(): trk_path = REPO_ROOT / trk_path
    vol_path = Path(args.volumes) if args.volumes else trk_path.parent / "instance_volumes.npz"
    if not vol_path.is_absolute(): vol_path = REPO_ROOT / vol_path
    basis_path = Path(args.pca_basis)
    if not basis_path.is_absolute(): basis_path = REPO_ROOT / basis_path
    if args.output is None:
        first_safe = args.queries[0].replace(" ", "_").replace("/", "_")
        out_path = trk_path.parent / f"query_{first_safe}.json"
    else:
        out_path = Path(args.output)
        if not out_path.is_absolute(): out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[setup] tracks={trk_path}\n[setup] volumes={vol_path}\n[setup] basis={basis_path}",
          flush=True)
    print(f"[setup] queries={args.queries}  top_k={args.top_k}", flush=True)

    # Load tracks + basis.
    trk = np.load(trk_path)
    track_emb_pca = trk["track_emb_pca"]                            # (G, 256) fp16
    G = int(trk["n_tracks_total"])
    print(f"[load] G={G} tracks, emb dim {track_emb_pca.shape[1]}", flush=True)

    state = torch.load(str(basis_path), map_location="cpu", weights_only=False)
    pca_mean = state["mean"].numpy().astype(np.float32)             # (D,)
    pca_components = state["components"].numpy().astype(np.float32)  # (target_dim, D)
    D = int(state["feature_dim"])
    target_dim = int(state["target_dim"])
    if track_emb_pca.shape[1] != target_dim:
        print(f"[ERR] track emb dim {track_emb_pca.shape[1]} != basis target_dim {target_dim}",
              flush=True)
        return 1

    # Decode track embeddings: e_d = e_pca @ components + mean.
    decoded = (track_emb_pca.astype(np.float32) @ pca_components + pca_mean)  # (G, D)
    # L2-normalize for cosine.
    norms = np.linalg.norm(decoded, axis=1, keepdims=True) + 1e-8
    track_emb_d = decoded / norms                                     # (G, D)

    # Load volumes for top-K rendering metadata.
    if vol_path.exists():
        vol = np.load(vol_path)
        vol_track_offsets = vol["track_offsets"]
        vol_n_points = vol["track_n_points"]
        print(f"[volumes] {int(vol['points'].shape[0])} total points across {len(vol_n_points)} tracks",
              flush=True)
    else:
        vol = None
        vol_track_offsets = None
        vol_n_points = None
        print(f"[volumes] {vol_path} missing — top-K results will lack point counts", flush=True)

    # Build text encoder (one-time SigLIP-2 load).
    is_v4 = "v4" in args.radio_version.lower()
    if args.lang_adaptor is None:
        lang_adaptor = "siglip2-g" if is_v4 else "siglip2"
    else:
        lang_adaptor = args.lang_adaptor
    t_text = time.time()
    encoder = _build_text_only_encoder(args.radio_version, lang_adaptor, args.device)
    print(f"[encoder] {args.radio_version} + {lang_adaptor} loaded in {time.time()-t_text:.1f}s",
          flush=True)

    with torch.no_grad():
        text_emb = encoder.encode_labels(args.queries, onehot=False)      # (Q, D)
    text_emb = text_emb.to(torch.float32).cpu().numpy()
    text_emb = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-8)
    Q = text_emb.shape[0]

    if text_emb.shape[1] != D:
        print(f"[ERR] text emb dim {text_emb.shape[1]} != basis D {D}", flush=True)
        return 1

    # Raw cosine: (G, Q)
    cos = track_emb_d @ text_emb.T                                  # (G, Q)
    # Temperature-100 softmax amplification per RADSeg paper §4.
    soft = np.exp(args.softmax_temp * cos)
    soft = soft / soft.sum(axis=1, keepdims=True)                   # (G, Q)
    # Per-track argmax over queries — useful for "what class is this track?"
    # downstream eval (B.5 mIoU). Print summary.
    track_pred_q = np.argmax(cos, axis=1)
    track_pred_cos = cos[np.arange(G), track_pred_q]
    track_pred_soft = soft[np.arange(G), track_pred_q]
    print(f"\n=== per-track top-1 query (argmax over queries) ===", flush=True)
    from collections import Counter
    pred_counter = Counter(args.queries[q] for q in track_pred_q.tolist())
    for q, cnt in pred_counter.most_common():
        print(f"  {q!r}: {cnt}/{G} tracks", flush=True)

    print(f"\n=== top-K tracks per query ===", flush=True)
    results = {"queries": list(args.queries), "tracks_total": G,
               "scene_id_offset": int(trk.get("scene_id_offset", 0)),
               "softmax_temp": float(args.softmax_temp),
               "cos_floor": float(args.cos_floor),
               "results": {}}
    for q_idx, q in enumerate(args.queries):
        col = cos[:, q_idx]
        soft_col = soft[:, q_idx]
        order = np.argsort(col)[::-1]
        topk = []
        for gid in order[: args.top_k]:
            score = float(col[gid])
            if score < args.cos_floor:
                break
            topk.append({
                "global_track_id": int(gid),
                "cos_score": score,
                "soft_score": float(soft_col[gid]),
                "n_points": int(vol_n_points[gid]) if vol_n_points is not None else None,
                "kf_count": int(trk["track_kf_count"][gid]),
                "n_instances": int(trk["track_n_instances"][gid]),
                "centroid_w": [float(x) for x in trk["track_centroid_w"][gid]],
                "total_vis": int(trk["track_total_vis"][gid]),
            })
        print(f"  [{q_idx}] {q!r}: top cos={col.max():.4f}, "
              f"{len(topk)} matches >= {args.cos_floor}", flush=True)
        for entry in topk:
            print(f"      gid={entry['global_track_id']:3d}  cos={entry['cos_score']:.4f}  "
                  f"soft={entry['soft_score']:.3f}  "
                  f"kfs={entry['kf_count']}  n_inst={entry['n_instances']}  "
                  f"centroid={[round(x,2) for x in entry['centroid_w']]}", flush=True)
        results["results"][q] = topk

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[save] {out_path}  ({out_path.stat().st_size/1024:.1f} KB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
