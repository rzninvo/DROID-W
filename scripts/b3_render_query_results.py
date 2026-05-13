"""Render the 5+3 smoke-query top-1 retrieval as a single figure.

For each query: pick the top-1 track, find the best-pixel-count KF it
appears in, save a triptych: RGB | RGB+mask-overlay | RGB+box-around-mask.
Stitch all 8 queries (5 present + 3 absent) into a single PNG.

Usage:
  python scripts/b3_render_query_results.py \\
    --embeddings-npz Outputs/.../panoptic_v7_track_embeddings.npz \\
    --refined-npz    Outputs/.../panoptic_v7_cropformer.npz \\
    --tracks-npz     Outputs/.../panoptic_v7_tracks_deva.npz \\
    --rgb-dir        datasets/.../rgb \\
    --out            Outputs/.../b3_smoke_query_results.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


PRESENT = ["chair", "monitor", "keyboard", "person", "desk"]
ABSENT = ["fire extinguisher", "elephant", "car"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings-npz", required=True, type=str)
    p.add_argument("--refined-npz", required=True, type=str)
    p.add_argument("--tracks-npz", required=True, type=str)
    p.add_argument("--rgb-dir", required=True, type=str)
    p.add_argument("--out", required=True, type=str)
    p.add_argument("--siglip-model", default="google/siglip2-large-patch16-384",
                   type=str)
    args = p.parse_args()

    emb_blob = np.load(args.embeddings_npz, allow_pickle=True)
    track_emb = emb_blob["embeddings_l2"]
    refined = np.load(args.refined_npz, allow_pickle=True)
    masks = refined["masks"]
    offsets = refined["seg_kf_offsets"]
    kf_gi = refined["kf_global_indices"]
    n_kf = int(refined["n_keyframes"])
    tracks_npz = np.load(args.tracks_npz, allow_pickle=True)
    gids = tracks_npz["global_track_ids"]

    # Build track -> list of (kf, m_global, n_pixels)
    track_supporting: dict[int, list[tuple[int, int, int]]] = {}
    for k_kf in range(n_kf):
        s_off, e_off = int(offsets[k_kf]), int(offsets[k_kf + 1])
        for m_global in range(s_off, e_off):
            gid = int(gids[m_global])
            if gid <= 0:
                continue
            n_pix = int(masks[m_global].sum())
            track_supporting.setdefault(gid, []).append((k_kf, m_global, n_pix))

    rgb_dir = Path(args.rgb_dir)
    rgb_files = sorted(f for f in rgb_dir.iterdir() if f.suffix == ".png")

    # Encode text queries.
    from transformers import AutoModel, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.siglip_model).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.siglip_model)

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
        raise RuntimeError(f"Unknown output: {type(obj)}")

    queries = PRESENT + ABSENT
    prompts = [f"a photo of a {q}" for q in queries]
    with torch.no_grad():
        inp = processor(text=prompts, return_tensors="pt",
                        padding="max_length", max_length=64).to(device)
        emb = _to_tensor(model.get_text_features(**inp))
        emb = torch.nn.functional.normalize(emb, dim=-1)
    text_emb = emb.cpu().float().numpy()

    valid = np.where(np.linalg.norm(track_emb, axis=1) > 0)[0]
    valid = valid[valid > 0]
    sims = text_emb @ track_emb.T
    sims_valid = sims[:, valid]

    tiles = []
    for qi, q in enumerate(queries):
        is_absent = qi >= len(PRESENT)
        order = np.argsort(-sims_valid[qi])
        top1_idx = int(order[0])
        track_id = int(valid[top1_idx])
        score = float(sims_valid[qi, top1_idx])

        supports = sorted(track_supporting.get(track_id, []),
                           key=lambda x: -x[2])
        if not supports:
            print(f"[WARN] '{q}': track {track_id} has no support")
            continue
        k_kf, m_global, n_pix = supports[0]
        gi = int(kf_gi[k_kf])
        bgr = cv2.imread(str(rgb_files[gi]))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        mask = masks[m_global]

        # Triptych: RGB | RGB+green overlay where mask | bbox highlight
        overlay = rgb.copy()
        overlay[mask] = (overlay[mask].astype(np.float32) * 0.5
                         + np.array([0, 255, 0], dtype=np.float32) * 0.5)\
                        .astype(np.uint8)
        ys, xs = np.where(mask)
        y0, x0, y1, x1 = int(ys.min()), int(xs.min()), int(ys.max())+1, int(xs.max())+1
        boxed = rgb.copy()
        col = (255, 60, 60) if is_absent else (60, 255, 60)
        cv2.rectangle(boxed, (x0, y0), (x1, y1), col, 3)

        sep = np.full((H, 4, 3), 255, dtype=np.uint8)
        row = np.concatenate([rgb, sep, overlay, sep, boxed], axis=1)

        # Label band
        band = np.full((48, row.shape[1], 3), 255, dtype=np.uint8)
        label = f"[{'ABSENT' if is_absent else 'PRESENT'}] '{q}' -> track {track_id}  cosine={score:.4f}  KF{k_kf:03d} (n_px={n_pix})"
        cv2.putText(band, label, (10, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2, cv2.LINE_AA)
        tile = np.concatenate([band, row], axis=0)
        tiles.append(tile)

    # Stack vertically with separators.
    if not tiles:
        print("[ERR] no tiles to save", flush=True)
        return 1
    W = tiles[0].shape[1]
    vsep = np.full((6, W, 3), 220, dtype=np.uint8)
    out = tiles[0]
    for t in tiles[1:]:
        out = np.concatenate([out, vsep, t], axis=0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f"[save] {out_path}  ({out.shape[1]}x{out.shape[0]})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
