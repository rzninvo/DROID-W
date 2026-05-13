"""Plan-v2 §Step 1 (carried into B.4) — 5+3 smoke open-vocab query gate.

Plan-v2 acceptance gate:
  - 5 present queries on freiburg3_walking_static: chair, monitor,
    keyboard, person, desk -> top-1 retrieved track overlaps the
    correct entity for >= 4/5.
  - 3 absent queries: fire extinguisher, elephant, car -> per-track
    top-1 cosine stays below an empirical floor for >= 3/3.

Uses SigLIP-2-large text encoder (same as image encoder in B.3) so
text and image embeddings live in the same 1024-d space.

Output: prints the top-3 tracks per query with cosine scores, plus
per-query top-1 track + best-supporting (KF, mask) so the user can
eyeball-check in the video.

Usage (cvg, droid-w env):
  python scripts/b3_smoke_query.py \\
    --embeddings-npz Outputs/.../panoptic_v7_track_embeddings.npz \\
    --refined-npz    Outputs/.../panoptic_v7_cropformer.npz \\
    --tracks-npz     Outputs/.../panoptic_v7_tracks_deva.npz \\
    --siglip-model   google/siglip2-large-patch16-384
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


PRESENT = ["chair", "monitor", "keyboard", "person", "desk"]
ABSENT = ["fire extinguisher", "elephant", "car"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings-npz", required=True, type=str)
    p.add_argument("--refined-npz", required=True, type=str)
    p.add_argument("--tracks-npz", required=True, type=str)
    p.add_argument("--siglip-model", default="google/siglip2-large-patch16-384",
                   type=str)
    p.add_argument("--top-k", default=3, type=int)
    args = p.parse_args()

    emb_blob = np.load(args.embeddings_npz, allow_pickle=True)
    track_emb = emb_blob["embeddings_l2"]               # (n_tracks+1, D)
    n_tracks = int(emb_blob["n_global_tracks"])
    n_views_total = emb_blob["n_views_total"]
    n_views_selected = emb_blob["n_views_selected"]
    print(f"[load] {n_tracks} tracks, dim={track_emb.shape[1]}", flush=True)

    refined = np.load(args.refined_npz, allow_pickle=True)
    offsets = refined["seg_kf_offsets"]
    tracks_npz = np.load(args.tracks_npz, allow_pickle=True)
    gids = tracks_npz["global_track_ids"]

    # Track -> list of (kf, m_global, n_pixels) for "best supporting view"
    track_supporting: dict[int, list[tuple[int, int, int]]] = {}
    masks = refined["masks"]
    n_kf = int(refined["n_keyframes"])
    for k_kf in range(n_kf):
        s_off, e_off = int(offsets[k_kf]), int(offsets[k_kf + 1])
        for m_global in range(s_off, e_off):
            gid = int(gids[m_global])
            if gid <= 0:
                continue
            n_pix = int(masks[m_global].sum())
            track_supporting.setdefault(gid, []).append((k_kf, m_global, n_pix))

    # Load SigLIP-2 text encoder.
    from transformers import AutoModel, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.siglip_model).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.siglip_model)
    print(f"[load] {args.siglip_model} on {device}", flush=True)

    def _to_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return obj
        for attr in ("pooler_output", "text_embeds", "last_hidden_state"):
            if hasattr(obj, attr):
                t = getattr(obj, attr)
                if t is None:
                    continue
                if t.ndim == 3:
                    t = t.mean(dim=1)
                return t
        raise RuntimeError(f"Unknown SigLIP text output: {type(obj)}")

    def encode_text(queries: list[str]) -> np.ndarray:
        # SigLIP-2 prepends "a photo of a/an X" template in some examples,
        # but plain "X" works for retrieval; we report raw cosine.
        prompts = [f"a photo of a {q}" for q in queries]
        with torch.no_grad():
            inp = processor(text=prompts, return_tensors="pt",
                            padding="max_length", max_length=64).to(device)
            emb = _to_tensor(model.get_text_features(**inp))
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.cpu().float().numpy()                  # (Q, D)

    queries = PRESENT + ABSENT
    text_emb = encode_text(queries)                       # (8, 1024)
    print(f"[encode] text {text_emb.shape}", flush=True)

    # Cosine: text_emb @ track_emb.T, both L2-normed already.
    # Skip row 0 (no track), and rows with zero embedding.
    valid_track_ids = np.where(np.linalg.norm(track_emb, axis=1) > 0)[0]
    valid_track_ids = valid_track_ids[valid_track_ids > 0]
    sims_full = text_emb @ track_emb.T                    # (Q, n_tracks+1)
    sims = sims_full[:, valid_track_ids]                  # (Q, n_valid)

    print(f"\n{'='*78}\nPRESENT queries (acceptance: >=4/5 should be "
          f"identifiable as correct in the video)\n{'='*78}", flush=True)
    for qi, q in enumerate(PRESENT):
        order = np.argsort(-sims[qi])[:args.top_k]
        print(f"\n[query] '{q}'", flush=True)
        for rank, idx in enumerate(order):
            track_id = int(valid_track_ids[idx])
            score = float(sims[qi, idx])
            supports = sorted(track_supporting.get(track_id, []),
                              key=lambda x: -x[2])[:2]
            sup_str = ", ".join(f"KF{s[0]:03d}/m={s[1]}(n={s[2]})"
                                 for s in supports)
            n_views = int(n_views_selected[track_id])
            n_total = int(n_views_total[track_id])
            print(f"  top-{rank+1}: track {track_id:3d}  cosine={score:.4f}  "
                  f"({n_views}/{n_total} views)  best-views: {sup_str}",
                  flush=True)

    print(f"\n{'='*78}\nABSENT queries (acceptance: >=3/3 should have lower "
          f"top-1 cosine than the PRESENT floor)\n{'='*78}", flush=True)
    present_top1_cosines = [float(np.max(sims[qi])) for qi in range(len(PRESENT))]
    floor_estimate = float(np.percentile(present_top1_cosines, 50))
    print(f"  (PRESENT top-1 median cosine = {floor_estimate:.4f})", flush=True)
    for qi_abs, q in enumerate(ABSENT):
        qi = len(PRESENT) + qi_abs
        order = np.argsort(-sims[qi])[:args.top_k]
        top1_score = float(sims[qi, order[0]])
        below_floor = "✓ below floor" if top1_score < floor_estimate else "✗ above floor"
        print(f"\n[query] '{q}' (top-1 {top1_score:.4f}) [{below_floor}]",
              flush=True)
        for rank, idx in enumerate(order):
            track_id = int(valid_track_ids[idx])
            score = float(sims[qi, idx])
            supports = sorted(track_supporting.get(track_id, []),
                              key=lambda x: -x[2])[:1]
            sup_str = (f"KF{supports[0][0]:03d}/m={supports[0][1]}"
                       if supports else "?")
            print(f"  top-{rank+1}: track {track_id:3d}  cosine={score:.4f}  "
                  f"({sup_str})", flush=True)

    # Quick separation stats.
    present_t1 = np.array([float(np.max(sims[qi]))
                            for qi in range(len(PRESENT))])
    absent_t1 = np.array([float(np.max(sims[len(PRESENT) + qi]))
                           for qi in range(len(ABSENT))])
    print(f"\n{'='*78}\nGate stats", flush=True)
    print(f"  PRESENT top-1 cosines: min={present_t1.min():.4f}  "
          f"max={present_t1.max():.4f}  mean={present_t1.mean():.4f}",
          flush=True)
    print(f"  ABSENT  top-1 cosines: min={absent_t1.min():.4f}  "
          f"max={absent_t1.max():.4f}  mean={absent_t1.mean():.4f}",
          flush=True)
    print(f"  separation: PRESENT mean - ABSENT mean = "
          f"{present_t1.mean() - absent_t1.mean():+.4f}", flush=True)
    n_absent_below = int((absent_t1 < present_t1.min()).sum())
    print(f"  ABSENT below PRESENT-min: {n_absent_below}/3 "
          f"(plan-v2 gate: >=3/3)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
