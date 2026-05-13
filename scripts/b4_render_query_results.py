"""Plan-v2 §Step 4 / B.4 -- render multi-instance LERF retrieval.

For each query in the plan-v2 5+3 set (or user-supplied), render a tile
that shows EVERY track with relevancy > τ (multi-instance), highlighted
in DISTINCT colours. Absent queries get a NO-MATCH banner.

Tile layout per query:
  [RGB best-view KF] | [RGB + ALL matched tracks overlaid in distinct
                       HSV colours] | [pure mask-colour map of matched
                       tracks only, on dark background]
Stacked vertically with the query label + relevancy + cosine + match
count + status banner.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


DEFAULT_PRESENT = ["chair", "monitor", "keyboard", "person", "desk"]
DEFAULT_ABSENT = ["fire extinguisher", "elephant", "car"]
CANONICAL_PHRASES = ("object", "things", "stuff", "texture")


def lerf_relevancy(phi: np.ndarray, q: np.ndarray, c: np.ndarray,
                    T: float = 10.0) -> np.ndarray:
    s_q = q @ phi.T
    s_c = c @ phi.T
    diff = s_q[:, :, None] - s_c.T[None, :, :]
    rel = 1.0 / (1.0 + np.exp(-T * diff))
    return rel.min(axis=-1)


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


def encode_text(model, processor, device, queries):
    with torch.no_grad():
        inp = processor(text=queries, return_tensors="pt",
                        padding="max_length", max_length=64).to(device)
        out = _to_tensor(model.get_text_features(**inp))
        out = out / (out.norm(dim=-1, keepdim=True) + 1e-9)
    return out.cpu().float().numpy()


def hsv_palette(n: int, phi: float = 0.6180339887498949) -> np.ndarray:
    """HSV golden-ratio palette, same as Fix A renderer. Returns (n, 3) RGB."""
    hues = ((np.arange(max(n, 1)) * phi) % 1.0).astype(np.float32)
    sats = 0.70 + 0.30 * ((np.arange(max(n, 1)) * 0.7) % 1.0)
    vals = 0.85 + 0.15 * ((np.arange(max(n, 1)) * 0.3 + 0.5) % 1.0)
    hsv = np.stack([hues * 179.0, sats * 255.0, vals * 255.0], axis=-1)
    hsv = hsv.reshape(-1, 1, 3).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape(-1, 3)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings-npz", required=True, type=str)
    p.add_argument("--refined-npz", required=True, type=str)
    p.add_argument("--tracks-npz", required=True, type=str)
    p.add_argument("--rgb-dir", required=True, type=str)
    p.add_argument("--out", required=True, type=str)
    p.add_argument("--queries", nargs="*", default=None)
    p.add_argument("--tau-relevancy", default=0.5, type=float)
    p.add_argument("--temperature", default=10.0, type=float)
    p.add_argument("--siglip-model", default="google/siglip2-large-patch16-384",
                   type=str)
    args = p.parse_args()

    if args.queries:
        queries = [q.replace("_", " ") for q in args.queries]
    else:
        queries = DEFAULT_PRESENT + DEFAULT_ABSENT

    emb_blob = np.load(args.embeddings_npz, allow_pickle=True)
    track_emb = emb_blob["embeddings_l2"]
    refined = np.load(args.refined_npz, allow_pickle=True)
    masks = refined["masks"]
    offsets = refined["seg_kf_offsets"]
    kf_gi = refined["kf_global_indices"]
    n_kf = int(refined["n_keyframes"])
    tracks_npz = np.load(args.tracks_npz, allow_pickle=True)
    gids = tracks_npz["global_track_ids"]

    # Track -> [(kf, m_global, n_pixels)]
    track_views: dict[int, list[tuple[int, int, int]]] = {}
    for k_kf in range(n_kf):
        s_off, e_off = int(offsets[k_kf]), int(offsets[k_kf + 1])
        for m_global in range(s_off, e_off):
            gid = int(gids[m_global])
            if gid <= 0:
                continue
            n_pix = int(masks[m_global].sum())
            track_views.setdefault(gid, []).append((k_kf, m_global, n_pix))

    rgb_dir = Path(args.rgb_dir)
    rgb_files = sorted(f for f in rgb_dir.iterdir() if f.suffix == ".png")

    valid_track_ids = np.where(np.linalg.norm(track_emb, axis=1) > 0)[0]
    valid_track_ids = valid_track_ids[valid_track_ids > 0]
    phi = track_emb[valid_track_ids]
    print(f"[setup] {len(valid_track_ids)} tracks searchable", flush=True)

    # Encode + relevancy.
    from transformers import AutoModel, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.siglip_model).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.siglip_model)
    q_emb = encode_text(model, processor, device, queries)
    c_emb = encode_text(model, processor, device, list(CANONICAL_PHRASES))
    rel = lerf_relevancy(phi, q_emb, c_emb, T=args.temperature)
    cos = phi @ q_emb.T                                              # (N, Q) — use as fallback display

    tiles = []
    for qi, q in enumerate(queries):
        is_absent_set = q in DEFAULT_ABSENT
        rel_q = rel[qi]
        cos_q = cos[:, qi]
        order = np.argsort(-rel_q)
        matches = [(int(valid_track_ids[i]), float(rel_q[i]), float(cos_q[i]))
                   for i in order if rel_q[i] > args.tau_relevancy]

        # Pick the best-view KF for visualization. If multiple matches, pick the
        # KF that maximises the SUM of matched-track mask areas (the KF where
        # most of the query is visible).
        if matches:
            matched_track_set = {m[0] for m in matches}
            best_kf = -1
            best_total = 0
            for k_kf in range(n_kf):
                s_off, e_off = int(offsets[k_kf]), int(offsets[k_kf + 1])
                total = 0
                for m_global in range(s_off, e_off):
                    gid = int(gids[m_global])
                    if gid in matched_track_set:
                        total += int(masks[m_global].sum())
                if total > best_total:
                    best_total = total
                    best_kf = k_kf
            if best_kf < 0:
                # fallback: KF of top-1 match's biggest view
                tid = matches[0][0]
                supports = sorted(track_views.get(tid, []),
                                   key=lambda x: -x[2])
                best_kf = supports[0][0] if supports else 0
        else:
            # NO MATCH: show a representative KF for visual reference.
            best_kf = n_kf // 2

        gi = int(kf_gi[best_kf])
        bgr = cv2.imread(str(rgb_files[gi]))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        # Build overlay + pure-mask panel with DISTINCT colours per matched track.
        n_matches = len(matches)
        palette = hsv_palette(max(n_matches, 1))
        overlay = rgb.copy().astype(np.float32)
        pure = np.full((H, W, 3), 40, dtype=np.uint8)
        legend_lines = []
        for mi, (tid, r, c) in enumerate(matches):
            colour = palette[mi]
            # Highlight this track's mask in the best_kf (if visible there).
            s_off, e_off = int(offsets[best_kf]), int(offsets[best_kf + 1])
            for m_global in range(s_off, e_off):
                if int(gids[m_global]) == tid:
                    mask = masks[m_global]
                    overlay[mask] = (overlay[mask] * 0.45
                                      + colour.astype(np.float32) * 0.55)
                    pure[mask] = colour
                    break
            legend_lines.append(f"#{mi+1} track {tid} rel={r:.3f} cos={c:.3f}")

        overlay = overlay.astype(np.uint8)

        # Construct the 3-panel row.
        sep = np.full((H, 6, 3), 255, dtype=np.uint8)
        row = np.concatenate([rgb, sep, overlay, sep, pure], axis=1)

        # Banner.
        if n_matches > 0:
            status = "MATCH"
            badge_col = (40, 200, 60)
        else:
            status = "NO-MATCH"
            badge_col = (220, 50, 50)
        banner_h = 28 + 22 * max(len(legend_lines), 1)
        banner = np.full((banner_h, row.shape[1], 3), 255, dtype=np.uint8)
        label = (f"[{'ABSENT' if is_absent_set else 'PRESENT'}] '{q}' "
                 f"{status} ({n_matches} {'match' if n_matches==1 else 'matches'})  "
                 f"best-KF {best_kf:03d}")
        cv2.putText(banner, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, badge_col, 2, cv2.LINE_AA)
        if n_matches == 0:
            # Show why: the best track's relevancy, so the user sees the floor.
            best_rel = float(rel_q[order[0]])
            best_cos = float(cos_q[order[0]])
            cv2.putText(banner, f"top track rel={best_rel:.3f} <= tau={args.tau_relevancy} "
                                  f"(some canonical phrase wins)",
                        (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (90, 90, 90), 1, cv2.LINE_AA)
        for li, line in enumerate(legend_lines[:8]):
            colour = palette[li]
            cv2.putText(banner, line, (10, 46 + 22 * li),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        tuple(int(c) for c in colour), 1, cv2.LINE_AA)

        tile = np.concatenate([banner, row], axis=0)
        tiles.append(tile)

    if not tiles:
        print("[ERR] no tiles", flush=True)
        return 1
    W_row = tiles[0].shape[1]
    vsep = np.full((6, W_row, 3), 220, dtype=np.uint8)
    out = tiles[0]
    for t in tiles[1:]:
        # All tiles have the same W_row by construction; heights may differ.
        if t.shape[1] != W_row:
            t = cv2.copyMakeBorder(t, 0, 0, 0, W_row - t.shape[1],
                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))
        out = np.concatenate([out, vsep, t], axis=0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f"[save] {out_path}  ({out.shape[1]}x{out.shape[0]})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
