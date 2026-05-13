"""Plan-v2 §Step 4 / B.4 v2 -- OVI-MAP labelset-argmax retrieval.

REPLACES the relative-Δ top-K approach (which mis-handles multi-instance
queries -- "monitor" with 2 visible returns only 1). Three parallel
research agents read the actual OVI-MAP and LERF code on cvg + GitHub:

  OVI-MAP (Liu et al., arXiv:2603.26541)
    scripts/utils/mesh_postprocess_utils.py:98-113
    scripts/visualizations/vis_view_selection.py:120-134
    scripts/utils/semantic_const.py:161  (REPLICA_51 labelset)
  -> Per-track ARGMAX over a closed labelset, then filter by label.
     Both monitors get argmax "monitor", both surface. No top-K.

  LERF (Kerr et al., ICCV 2023, arXiv:2303.09553)
    lerf/lerf.py:181-186
  -> Continuous heatmap with HARD FLOOR at 0.5. No instance concept.

For HERMES-SLAM's 65 pre-segmented tracks, OVI-MAP's argmax-labelset is
the correct protocol -- it guarantees multi-instance surfacing.

  for each track t:
    sim_label[k] = cos(phi_t, label_emb[k])
    sim_canon[i] = cos(phi_t, canonical_emb[i])
    rela[k] = min_i  sigma(T * (sim_label[k] - sim_canon[i]))
    t.label = LABELSET[argmax(rela)]
    t.label_rela = max(rela)

  for query Q:
    if Q in LABELSET: return [t for t in tracks if t.label == Q]
    else:             # free-form fallback (LERF heatmap)
      rela_Q = lerf_relevancy(phi, q_emb, canonical_emb)
      return [t for t in tracks if rela_Q[t] > 0.5]

Both the labelset and the canonical phrases follow OVI-MAP verbatim
(REPLICA_51 from semantic_const.py:161 + person/keyboard/computer for
the TUM walking office scene; canonical = ['object','things','stuff',
'texture'] from text_embedding.py:42-43).

Usage (cvg, droid-w env):
  python scripts/b4_query_labelset.py \\
    --embeddings-npz Outputs/.../panoptic_v7_track_embeddings_ovi.npz \\
    --refined-npz    Outputs/.../panoptic_v7_cropformer.npz \\
    --tracks-npz     Outputs/.../panoptic_v7_tracks_deva.npz \\
    --queries chair monitor keyboard person desk floor ceiling computer car ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


# OVI-MAP semantic_const.py:161 (REPLICA_51) verbatim:
REPLICA_51 = [
    'wall', 'ceiling', 'floor', 'chair', 'blinds', 'sofa', 'table',
    'rug', 'window', 'lamp', 'door', 'pillow', 'bench', 'tv-screen',
    'cabinet', 'pillar', 'blanket', 'tv-stand', 'cushion', 'bin',
    'vent', 'bed', 'stool', 'picture', 'indoor-plant', 'desk',
    'comforter', 'nightstand', 'shelf', 'vase', 'plant-stand',
    'basket', 'plate', 'monitor', 'pipe', 'panel', 'desk-organizer',
    'wall-plug', 'book', 'box', 'clock', 'sculpture', 'tissue-paper',
    'camera', 'tablet', 'pot', 'bottle', 'candle', 'bowl', 'cloth',
    'switch',
]

# Office-pruned labelset (REPLICA_51 minus labels not in our TUM scene
# that were absorbing tracks via the argmax -- 'vent' was claiming 13
# tracks, 'tv-stand' 5, etc., starving 'floor'/'wall' of their natural
# matches). 28 office-relevant labels + 6 TUM additions.
TUM_OFFICE_LABELS = [
    # structure
    'wall', 'ceiling', 'floor', 'window', 'door',
    # furniture
    'chair', 'desk', 'table', 'cabinet', 'shelf', 'sofa', 'bench',
    'stool',
    # displays + tech
    'monitor', 'tv-screen', 'computer', 'keyboard', 'mouse', 'camera',
    'tablet',
    # accessories
    'lamp', 'book', 'box', 'bottle', 'clock', 'picture',
    # plants
    'indoor-plant',
    # bin
    'bin',
]

# Additions for TUM walking office scene (not in REPLICA_51):
TUM_OFFICE_ADD = [
    'person', 'paper', 'shirt',
]

# OVI-MAP text_embedding.py:42-43 verbatim:
CANONICAL_PHRASES = ['object', 'things', 'stuff', 'texture']

DEFAULT_PRESENT = ["chair", "monitor", "keyboard", "person", "desk"]
DEFAULT_ABSENT = ["fire extinguisher", "elephant", "car"]


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


def encode_text(model, processor, device, texts):
    with torch.no_grad():
        inp = processor(text=texts, return_tensors="pt",
                        padding="max_length", max_length=64).to(device)
        out = _to_tensor(model.get_text_features(**inp))
        out = out / (out.norm(dim=-1, keepdim=True) + 1e-9)
    return out.cpu().float().numpy()


def lerf_per_label_relevancy(phi: np.ndarray,        # (N, D)
                              label_emb: np.ndarray,  # (L, D)
                              canon_emb: np.ndarray,  # (C, D)
                              T: float = 10.0
                              ) -> np.ndarray:        # (N, L)
    """For each track, per-label LERF relevancy:
       rela[n, k] = min_i sigma(T * (phi_n.label_k - phi_n.canon_i))"""
    s_lab = phi @ label_emb.T                          # (N, L)
    s_can = phi @ canon_emb.T                          # (N, C)
    diff = s_lab[:, :, None] - s_can[:, None, :]       # (N, L, C)
    return (1.0 / (1.0 + np.exp(-T * diff))).min(axis=-1)  # (N, L)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings-npz", required=True, type=str)
    p.add_argument("--refined-npz", required=True, type=str)
    p.add_argument("--tracks-npz", required=True, type=str)
    p.add_argument("--queries", nargs="*", default=None)
    p.add_argument("--labelset", nargs="*", default=None,
                   help="closed labelset; default = office-pruned TUM_OFFICE_LABELS + TUM additions")
    p.add_argument("--use-full-replica-51", action="store_true",
                   help="use full REPLICA_51 instead of office-pruned (for paper-faithful Replica eval)")
    p.add_argument("--canonical-phrases", nargs="*", default=CANONICAL_PHRASES)
    p.add_argument("--tau-fallback", default=0.55, type=float,
                   help="LERF threshold for free-form queries NOT in labelset. "
                        "0.55 tighter than LERF's 0.5 natural boundary; "
                        "eliminates the 'dog' false-positive at rel=0.5018.")
    p.add_argument("--temperature", default=10.0, type=float)
    p.add_argument("--siglip-model", default="google/siglip2-large-patch16-384",
                   type=str)
    p.add_argument("--out-json", default=None, type=str)
    args = p.parse_args()

    queries = ([q.replace("_", " ") for q in args.queries]
               if args.queries else DEFAULT_PRESENT + DEFAULT_ABSENT)
    if args.labelset:
        labelset = args.labelset
        labelset_name = "user-supplied"
    elif args.use_full_replica_51:
        labelset = REPLICA_51 + TUM_OFFICE_ADD
        labelset_name = "REPLICA_51 + TUM"
    else:
        labelset = TUM_OFFICE_LABELS + TUM_OFFICE_ADD
        labelset_name = "TUM_OFFICE_pruned + TUM"
    print(f"[setup] labelset = {labelset_name}, size = {len(labelset)}",
          flush=True)
    print(f"[setup] canonical    = {args.canonical_phrases}", flush=True)

    emb_blob = np.load(args.embeddings_npz, allow_pickle=True)
    track_emb = emb_blob["embeddings_l2"]
    n_tracks = int(emb_blob["n_global_tracks"])

    valid_track_ids = np.where(np.linalg.norm(track_emb, axis=1) > 0)[0]
    valid_track_ids = valid_track_ids[valid_track_ids > 0]
    phi = track_emb[valid_track_ids]                        # (N_valid, D)
    print(f"[setup] {len(valid_track_ids)} tracks searchable", flush=True)

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

    from transformers import AutoModel, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {args.siglip_model} on {device}", flush=True)
    model = AutoModel.from_pretrained(args.siglip_model).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.siglip_model)

    label_emb = encode_text(model, processor, device, labelset)
    canon_emb = encode_text(model, processor, device, args.canonical_phrases)
    print(f"[encode] labelset {label_emb.shape}  canonical {canon_emb.shape}",
          flush=True)

    # ARGMAX-OVER-LABELSET (OVI-MAP protocol).
    rela = lerf_per_label_relevancy(phi, label_emb, canon_emb,
                                     T=args.temperature)         # (N_valid, L)
    track_argmax = rela.argmax(axis=1)                            # (N_valid,)
    track_argmax_rela = rela.max(axis=1)                          # (N_valid,)
    track_label = [labelset[i] for i in track_argmax]

    print(f"\n{'='*82}", flush=True)
    print(f"PER-TRACK ARGMAX (OVI-MAP vis_view_selection.py:120 protocol):",
          flush=True)
    print(f"{'='*82}", flush=True)
    track_to_label = {}
    for n_idx, tid in enumerate(valid_track_ids):
        track_to_label[int(tid)] = (track_label[n_idx],
                                     float(track_argmax_rela[n_idx]))
    # Histogram by label.
    from collections import Counter
    label_hist = Counter(track_label)
    for lab, count in label_hist.most_common(15):
        print(f"  {lab:25s}: {count:3d} tracks", flush=True)

    # Per-query: filter or fallback.
    print(f"\n{'='*82}\nQUERY RESULTS\n{'='*82}", flush=True)
    results = {}
    for q in queries:
        results[q] = {"protocol": None, "matches": []}
        # Normalise query for labelset match.
        q_norm = q.strip().lower()
        labelset_norm = [L.lower() for L in labelset]
        if q_norm in labelset_norm:
            results[q]["protocol"] = "labelset_argmax"
            li = labelset_norm.index(q_norm)
            matched = [int(valid_track_ids[n_idx])
                       for n_idx in range(len(valid_track_ids))
                       if track_argmax[n_idx] == li]
            for tid in matched:
                lab, r = track_to_label[tid]
                supports = sorted(track_views.get(tid, []),
                                   key=lambda x: -x[2])[:1]
                sup_str = (f"KF{supports[0][0]:03d}/n={supports[0][2]}"
                           if supports else "?")
                results[q]["matches"].append({
                    "track_id": tid, "label_rela": r,
                    "best_view": sup_str})
            tag = (f"✓ MATCH ({len(matched)})"
                   if matched else "✗ NO-MATCH (label has 0 tracks)")
            print(f"\n[query] '{q}'  [labelset]  {tag}", flush=True)
            for m in results[q]["matches"]:
                print(f"   track {m['track_id']:3d}  rela={m['label_rela']:.4f}  "
                      f"best: {m['best_view']}", flush=True)
        else:
            results[q]["protocol"] = "free_form_lerf"
            q_emb = encode_text(model, processor, device, [q])
            # Per-track per-query relevancy
            s_q = phi @ q_emb.T                          # (N_valid, 1)
            s_c = phi @ canon_emb.T                      # (N_valid, C)
            diff = s_q[:, :, None] - s_c[:, None, :]
            rela_q = (1.0 / (1.0 + np.exp(-args.temperature * diff))
                       ).min(axis=-1).squeeze(-1)         # (N_valid,)
            order = np.argsort(-rela_q)
            matches = [(int(valid_track_ids[i]), float(rela_q[i]))
                       for i in order if rela_q[i] > args.tau_fallback]
            for tid, r in matches:
                supports = sorted(track_views.get(tid, []),
                                   key=lambda x: -x[2])[:1]
                sup_str = (f"KF{supports[0][0]:03d}/n={supports[0][2]}"
                           if supports else "?")
                results[q]["matches"].append({
                    "track_id": tid, "label_rela": r,
                    "best_view": sup_str})
            tag = (f"✓ MATCH ({len(matches)})"
                   if matches else "✗ NO-MATCH")
            print(f"\n[query] '{q}'  [free-form, not in labelset]  {tag}",
                  flush=True)
            if not matches:
                best_rel = float(rela_q[order[0]])
                print(f"   reason: top rel={best_rel:.4f} <= τ={args.tau_fallback}",
                      flush=True)
            for m in results[q]["matches"]:
                print(f"   track {m['track_id']:3d}  rel={m['label_rela']:.4f}  "
                      f"best: {m['best_view']}", flush=True)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as fh:
            json.dump({
                "labelset": labelset,
                "canonical": args.canonical_phrases,
                "tau_fallback": args.tau_fallback,
                "temperature": args.temperature,
                "queries": queries,
                "results": results,
                "track_labels": {str(tid): list(lab)
                                  for tid, lab in track_to_label.items()},
            }, fh, indent=2)
        print(f"\n[save] {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
