"""Plan-v2 §Step 4 / B.4 — LERF-style open-vocab retrieval with
canonical-phrase rejection + multi-instance return.

Implements the LERF relevancy score (Kerr et al., ICCV 2023,
arXiv:2303.09553, lerf/encoders/openclip_encoder.py:get_relevancy)
on top of HERMES-SLAM's 65 SigLIP-2-large embedded tracks (B.3 Fix C):

  relevancy(φ, q) = min_i  σ( 10·φ·q, 10·φ·c_i )[0]
                  = min_i  exp(10·φ·q) / (exp(10·φ·q) + exp(10·φ·c_i))

  where  φ  = per-track L2-normed SigLIP-2 embedding (1024-d)
         q  = SigLIP-2 text embedding of the user query
         c_i ∈ {"object", "things", "stuff", "texture"}   (LERF + OVI-MAP)
         T  = 10 (LERF temperature INSIDE the 2-way softmax; NOT
              SigLIP-2's trained logit_scale=100)

Decision rule (LERF natural boundary):
  - relevancy > 0.5  => track matches query (more like query than every
    canonical distractor)
  - relevancy ≤ 0.5  => track REJECTED for this query

Two failure modes from B.3 it fixes:
  (1) Absent queries (e.g. "car" in indoor scene): some canonical
      phrase ("object" or "stuff") will beat the unrelated query for
      every track, so all relevancy < 0.5 → return EMPTY (NO MATCH).
  (2) Multi-instance queries (e.g. "person" with 2 people visible):
      return ALL tracks where relevancy > 0.5, sorted descending —
      not just argmax.

CLAUDE.md sec.6 compliance: every per-query decision (number of
matches, threshold-margin) is printed explicitly. No silent fallbacks.

Sources verified (primary):
  - LERF: Kerr et al. ICCV 2023, arXiv:2303.09553
          official repo: github.com/kerrj/lerf
          lerf/encoders/openclip_encoder.py:get_relevancy (T=10, τ=0.5)
  - OVI-MAP canonical phrases: text_embedding.py:43
          ['object', 'things', 'stuff', 'texture']
          mesh_postprocess_utils.py:98-115 (uses the formula but no τ)
  - SigLIP-2: Tschannen et al. 2025, arXiv:2502.14786
          model.logit_scale ≈ 100 (training-time; we use LERF's T=10 here)

Usage (cvg, droid-w env):
  python scripts/b4_open_vocab_query.py \\
    --embeddings-npz Outputs/.../panoptic_v7_track_embeddings_ovi.npz \\
    --refined-npz    Outputs/.../panoptic_v7_cropformer.npz \\
    --tracks-npz     Outputs/.../panoptic_v7_tracks_deva.npz \\
    --queries chair monitor keyboard person desk fire_extinguisher elephant car \\
    --tau-relevancy 0.5 --temperature 10.0 \\
    --siglip-model google/siglip2-large-patch16-384
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


# LERF canonical phrases (lerf/encoders/openclip_encoder.py)
CANONICAL_PHRASES = ("object", "things", "stuff", "texture")

# plan-v2 §Step 1 default 5+3 set.
DEFAULT_PRESENT = ["chair", "monitor", "keyboard", "person", "desk"]
DEFAULT_ABSENT = ["fire extinguisher", "elephant", "car"]


def lerf_relevancy(phi: np.ndarray,         # (N, D) L2-normed track embeddings
                    q: np.ndarray,           # (Q, D) L2-normed query embeddings
                    c: np.ndarray,           # (C, D) L2-normed canonical embeddings
                    T: float = 10.0,
                    ) -> np.ndarray:
    """Returns (Q, N) relevancy matrix where each entry is

      relevancy_{q, n} = min_i  exp(T·φ·q) / (exp(T·φ·q) + exp(T·φ·c_i))

    Per-element 2-way softmax with `min` over canonical phrases.
    Implementation note: log-sum-exp for numerical stability — the
    pairwise softmax is
        σ(T·s_q - T·s_c)  where σ is the logistic.
    """
    # (Q, N) cosines  (φ and q are L2-normed → cosine == dot)
    s_q = q @ phi.T            # (Q, N)
    # (C, N) cosines
    s_c = c @ phi.T            # (C, N)
    # Pairwise difference: rel_{q, n, i} = σ( T·(s_q - s_c_i) )
    # shape: (Q, N, 1) - (1, N, C) → (Q, N, C); we want min over C → (Q, N).
    diff = s_q[:, :, None] - s_c.T[None, :, :]   # (Q, N, C)
    rel = 1.0 / (1.0 + np.exp(-T * diff))         # logistic σ
    return rel.min(axis=-1)                       # (Q, N)


def _to_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    for attr in ("pooler_output", "text_embeds", "image_embeds",
                 "last_hidden_state"):
        if hasattr(obj, attr) and getattr(obj, attr) is not None:
            t = getattr(obj, attr)
            if t.ndim == 3:
                t = t.mean(dim=1)
            return t
    raise RuntimeError(f"Unknown SigLIP output: {type(obj)}")


def encode_text(model, processor, device, queries: list[str]) -> np.ndarray:
    """Encode queries via SigLIP-2 text encoder, L2-norm."""
    with torch.no_grad():
        inp = processor(text=queries, return_tensors="pt",
                        padding="max_length", max_length=64).to(device)
        out = model.get_text_features(**inp)
        out = _to_tensor(out)
        out = out / (out.norm(dim=-1, keepdim=True) + 1e-9)
    return out.cpu().float().numpy()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings-npz", required=True, type=str)
    p.add_argument("--refined-npz", required=True, type=str)
    p.add_argument("--tracks-npz", required=True, type=str)
    p.add_argument("--queries", nargs="*", default=None,
                   help="List of free-form text queries (space-separated, "
                        "underscore for compound; default = plan-v2 5+3 set)")
    p.add_argument("--canonical-phrases", nargs="*",
                   default=list(CANONICAL_PHRASES),
                   help="LERF canonical phrases for rejection. Default: "
                        "['object', 'things', 'stuff', 'texture']")
    p.add_argument("--tau-relevancy", default=0.5, type=float,
                   help="LERF natural softmax decision boundary; "
                        "tracks above this match the query")
    p.add_argument("--temperature", default=10.0, type=float,
                   help="LERF temperature inside softmax (NOT SigLIP "
                        "logit_scale=100)")
    p.add_argument("--siglip-model", default="google/siglip2-large-patch16-384",
                   type=str)
    p.add_argument("--top-display", default=5, type=int,
                   help="how many top tracks to display per query "
                        "(only those above tau are matches; the rest are "
                        "shown for context with rejected status)")
    p.add_argument("--top-rel-window", default=0.020, type=float,
                   help="precision filter: after the LERF tau-gate, only "
                        "keep tracks whose relevancy is within this much of "
                        "the top-1 relevancy. Cuts the loose tail at the "
                        "edge of the natural cluster.")
    p.add_argument("--max-matches", default=5, type=int,
                   help="safety cap: never return more than this many "
                        "tracks per query. Multi-instance is fine, but a "
                        "single query returning 20 means the rel-window is "
                        "too loose.")
    p.add_argument("--out-json", default=None, type=str)
    args = p.parse_args()

    # Convert underscored queries to spaces ("fire_extinguisher" → "fire extinguisher")
    if args.queries:
        queries = [q.replace("_", " ") for q in args.queries]
    else:
        queries = DEFAULT_PRESENT + DEFAULT_ABSENT

    print(f"[load] embeddings: {args.embeddings_npz}", flush=True)
    emb_blob = np.load(args.embeddings_npz, allow_pickle=True)
    track_emb = emb_blob["embeddings_l2"]               # (n_tracks+1, D)
    n_tracks = int(emb_blob["n_global_tracks"])
    n_views_total = emb_blob["n_views_total"]
    n_views_selected = emb_blob.get("n_views_selected", np.zeros_like(n_views_total))
    print(f"[load] {n_tracks} global tracks, dim={track_emb.shape[1]}",
          flush=True)

    # Filter: tracks with zero embedding (singletons that fell below the
    # B.3 Fix C 1000-px vis_area_thres) should not be searchable.
    valid_track_ids = np.where(np.linalg.norm(track_emb, axis=1) > 0)[0]
    valid_track_ids = valid_track_ids[valid_track_ids > 0]
    phi = track_emb[valid_track_ids]                    # (N_valid, D)
    print(f"[setup] {len(valid_track_ids)} tracks have a valid embedding",
          flush=True)

    # Track → list of (kf, m_global, n_pixels) for best-view reporting.
    refined = np.load(args.refined_npz, allow_pickle=True)
    offsets = refined["seg_kf_offsets"]
    masks = refined["masks"]
    tracks_npz = np.load(args.tracks_npz, allow_pickle=True)
    gids = tracks_npz["global_track_ids"]
    track_views: dict[int, list[tuple[int, int, int]]] = {}
    n_kf = int(refined["n_keyframes"])
    for k_kf in range(n_kf):
        s_off, e_off = int(offsets[k_kf]), int(offsets[k_kf + 1])
        for m_global in range(s_off, e_off):
            gid = int(gids[m_global])
            if gid <= 0:
                continue
            n_pix = int(masks[m_global].sum())
            track_views.setdefault(gid, []).append((k_kf, m_global, n_pix))

    # Load SigLIP-2 text encoder.
    from transformers import AutoModel, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {args.siglip_model} on {device}", flush=True)
    model = AutoModel.from_pretrained(args.siglip_model).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.siglip_model)

    # Encode queries + canonical phrases.
    q_emb = encode_text(model, processor, device, queries)             # (Q, D)
    c_emb = encode_text(model, processor, device, args.canonical_phrases)  # (C, D)
    print(f"[encode] queries {q_emb.shape}  canonical {c_emb.shape}",
          flush=True)

    # LERF relevancy.
    rel = lerf_relevancy(phi, q_emb, c_emb, T=args.temperature)        # (Q, N_valid)

    print(f"\n{'='*82}\nLERF relevancy retrieval — T={args.temperature}, "
          f"τ={args.tau_relevancy}\n"
          f"canonical_phrases = {args.canonical_phrases}\n"
          f"{'='*82}\n", flush=True)

    results = {}
    for qi, q in enumerate(queries):
        rel_q = rel[qi]                                                # (N_valid,)
        # Also report raw cosine for sanity:
        cos_q = phi @ q_emb[qi]                                        # (N_valid,)

        order = np.argsort(-rel_q)
        # Stage 1: LERF tau-gate (open-set rejection)
        passers = [(int(valid_track_ids[i]), float(rel_q[i]), float(cos_q[i]))
                   for i in order if rel_q[i] > args.tau_relevancy]
        # Stage 2: relative-Delta window from top-1 (precision filter)
        #          + max-matches safety cap. This is what keeps "person"
        #          retrieval at ~2 visible persons instead of 20.
        if passers:
            top_rel = passers[0][1]
            matches = [m for m in passers
                       if m[1] >= top_rel - args.top_rel_window]
            if len(matches) > args.max_matches:
                matches = matches[:args.max_matches]
        else:
            matches = []
        all_top = [(int(valid_track_ids[i]), float(rel_q[i]), float(cos_q[i]))
                   for i in order[:args.top_display]]

        if matches:
            tag = f"✓ MATCH ({len(matches)})"
        else:
            tag = "✗ NO-MATCH"
        print(f"\n[query] '{q}'   {tag}", flush=True)
        if matches:
            for rank, (tid, r, c) in enumerate(matches):
                supports = sorted(track_views.get(tid, []),
                                   key=lambda x: -x[2])[:2]
                sup_str = ", ".join(f"KF{s[0]:03d}/m={s[1]}(n={s[2]})"
                                     for s in supports)
                n_v = int(n_views_total[tid])
                n_s = int(n_views_selected[tid])
                print(f"   match-{rank+1}: track {tid:3d}  rel={r:.4f}  "
                      f"cos={c:.4f}  ({n_s}/{n_v} views)  best: {sup_str}",
                      flush=True)
        else:
            # Per CLAUDE.md §6: log WHY no match was found.
            best_rel = float(rel_q[order[0]])
            best_cos = float(cos_q[order[0]])
            print(f"   reason: best track rel={best_rel:.4f} <= τ={args.tau_relevancy} "
                  f"(some canonical phrase beat the query for every track); "
                  f"top cos={best_cos:.4f}", flush=True)
        # Also show top-display for context.
        print("   (context, top-display top-N):", flush=True)
        for rank, (tid, r, c) in enumerate(all_top):
            status = "✓" if r > args.tau_relevancy else " "
            print(f"     {status} top-{rank+1}: track {tid:3d}  rel={r:.4f}  "
                  f"cos={c:.4f}", flush=True)

        results[q] = {
            "n_matches": len(matches),
            "matches": [
                {"track_id": tid, "relevancy": r, "cosine": c}
                for (tid, r, c) in matches
            ],
            "rejected_top": [
                {"track_id": tid, "relevancy": r, "cosine": c}
                for (tid, r, c) in all_top if (tid, r, c) not in matches
            ],
        }

    # Quick aggregate stats.
    n_present = sum(1 for q in queries if q in DEFAULT_PRESENT
                     and results[q]["n_matches"] >= 1)
    n_absent_rejected = sum(1 for q in queries if q in DEFAULT_ABSENT
                             and results[q]["n_matches"] == 0)
    if any(q in DEFAULT_PRESENT for q in queries) and any(q in DEFAULT_ABSENT for q in queries):
        n_present_total = sum(1 for q in queries if q in DEFAULT_PRESENT)
        n_absent_total = sum(1 for q in queries if q in DEFAULT_ABSENT)
        print(f"\n{'='*82}\nplan-v2 5+3 smoke gate:", flush=True)
        print(f"  PRESENT >=1 match: {n_present}/{n_present_total}  "
              f"(gate: >=4/5)", flush=True)
        print(f"  ABSENT  NO-MATCH : {n_absent_rejected}/{n_absent_total}  "
              f"(gate: >=3/3)", flush=True)
        present_ok = n_present >= 4 if n_present_total == 5 else None
        absent_ok = n_absent_rejected >= 3 if n_absent_total == 3 else None
        print(f"  gate verdict     : "
              f"PRESENT {'PASS' if present_ok else 'FAIL'}, "
              f"ABSENT {'PASS' if absent_ok else 'FAIL'}", flush=True)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as fh:
            json.dump({
                "tau_relevancy": args.tau_relevancy,
                "temperature": args.temperature,
                "canonical_phrases": args.canonical_phrases,
                "siglip_model": args.siglip_model,
                "queries": queries,
                "results": results,
            }, fh, indent=2)
        print(f"\n[save] {args.out_json}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
