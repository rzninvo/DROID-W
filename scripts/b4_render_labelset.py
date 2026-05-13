"""Plan-v2 §Step 4 / B.4 v2 -- render labelset-argmax retrieval.

Same protocol as b4_query_labelset.py (OVI-MAP vis_view_selection.py:
120-134): per-track argmax over closed labelset, filter by label.

For each query in the 8+8 set, render a tile showing every track whose
argmax label == query, highlighted in distinct HSV-golden-ratio colours.
Free-form queries (not in labelset) fall back to LERF heatmap with τ=0.5.
NO-MATCH tiles get a red banner with explanation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


TUM_OFFICE_LABELS = [
    'wall', 'ceiling', 'floor', 'window', 'door',
    'chair', 'desk', 'table', 'cabinet', 'shelf', 'sofa', 'bench', 'stool',
    'monitor', 'tv-screen', 'computer', 'keyboard', 'mouse', 'camera',
    'tablet',
    'lamp', 'book', 'box', 'bottle', 'clock', 'picture',
    'indoor-plant', 'bin',
]
TUM_OFFICE_ADD = ['person', 'paper', 'shirt']
CANONICAL_PHRASES = ['object', 'things', 'stuff', 'texture']
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


def hsv_palette(n: int, phi: float = 0.6180339887498949) -> np.ndarray:
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
    p.add_argument("--tau-fallback", default=0.55, type=float)
    p.add_argument("--temperature", default=10.0, type=float)
    p.add_argument("--siglip-model", default="google/siglip2-large-patch16-384",
                   type=str)
    args = p.parse_args()

    queries = ([q.replace("_", " ") for q in args.queries]
               if args.queries else
               ["chair", "monitor", "keyboard", "person", "desk",
                "floor", "ceiling", "computer",
                "car", "elephant", "fire extinguisher", "airplane",
                "boat", "dog", "tree", "bicycle"])
    labelset = TUM_OFFICE_LABELS + TUM_OFFICE_ADD
    labelset_norm = [L.lower() for L in labelset]

    emb_blob = np.load(args.embeddings_npz, allow_pickle=True)
    track_emb = emb_blob["embeddings_l2"]
    refined = np.load(args.refined_npz, allow_pickle=True)
    masks = refined["masks"]
    offsets = refined["seg_kf_offsets"]
    kf_gi = refined["kf_global_indices"]
    n_kf = int(refined["n_keyframes"])
    tracks_npz = np.load(args.tracks_npz, allow_pickle=True)
    gids = tracks_npz["global_track_ids"]

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

    from transformers import AutoModel, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.siglip_model).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.siglip_model)

    label_emb = encode_text(model, processor, device, labelset)
    canon_emb = encode_text(model, processor, device, CANONICAL_PHRASES)
    # Per-track argmax over labelset.
    s_lab = phi @ label_emb.T
    s_can = phi @ canon_emb.T
    diff = s_lab[:, :, None] - s_can[:, None, :]
    rela = (1.0 / (1.0 + np.exp(-args.temperature * diff))).min(axis=-1)
    track_argmax = rela.argmax(axis=1)

    tiles = []
    for q in queries:
        q_norm = q.strip().lower()
        is_absent_set = q in DEFAULT_ABSENT
        protocol = "labelset" if q_norm in labelset_norm else "free-form"

        if q_norm in labelset_norm:
            li = labelset_norm.index(q_norm)
            matched_track_ids = [int(valid_track_ids[n_idx])
                                  for n_idx in range(len(valid_track_ids))
                                  if track_argmax[n_idx] == li]
            matches = [(tid, float(rela[n_idx, li]))
                       for n_idx, tid in enumerate(valid_track_ids)
                       if track_argmax[n_idx] == li]
        else:
            q_emb = encode_text(model, processor, device, [q])
            s_q = phi @ q_emb.T
            diff_q = s_q[:, :, None] - s_can[:, None, :]
            rela_q = (1.0 / (1.0 + np.exp(-args.temperature * diff_q))
                       ).min(axis=-1).squeeze(-1)
            order = np.argsort(-rela_q)
            matches = [(int(valid_track_ids[i]), float(rela_q[i]))
                       for i in order if rela_q[i] > args.tau_fallback]
            matched_track_ids = [m[0] for m in matches]

        # Pick best-view KF.
        if matched_track_ids:
            track_set = set(matched_track_ids)
            best_kf = -1
            best_total = 0
            for k_kf in range(n_kf):
                s_off, e_off = int(offsets[k_kf]), int(offsets[k_kf + 1])
                total = 0
                for m_global in range(s_off, e_off):
                    if int(gids[m_global]) in track_set:
                        total += int(masks[m_global].sum())
                if total > best_total:
                    best_total = total
                    best_kf = k_kf
            if best_kf < 0:
                best_kf = n_kf // 2
        else:
            best_kf = n_kf // 2

        gi = int(kf_gi[best_kf])
        bgr = cv2.imread(str(rgb_files[gi]))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        palette = hsv_palette(max(len(matches), 1))
        overlay = rgb.copy().astype(np.float32)
        pure = np.full((H, W, 3), 40, dtype=np.uint8)
        legend_lines = []
        for mi, (tid, r) in enumerate(matches):
            colour = palette[mi]
            s_off, e_off = int(offsets[best_kf]), int(offsets[best_kf + 1])
            for m_global in range(s_off, e_off):
                if int(gids[m_global]) == tid:
                    m = masks[m_global]
                    overlay[m] = overlay[m] * 0.40 + colour.astype(np.float32) * 0.60
                    pure[m] = colour
                    break
            legend_lines.append(f"#{mi+1} track {tid} rela={r:.3f}")

        overlay = overlay.astype(np.uint8)
        sep = np.full((H, 6, 3), 255, dtype=np.uint8)
        row = np.concatenate([rgb, sep, overlay, sep, pure], axis=1)

        if matches:
            status = "MATCH"
            badge_col = (40, 200, 60)
        else:
            status = "NO-MATCH"
            badge_col = (220, 50, 50)
        banner_h = 28 + 22 * max(len(legend_lines), 1)
        banner = np.full((banner_h, row.shape[1], 3), 255, dtype=np.uint8)
        label = (f"[{'ABSENT' if is_absent_set else 'PRESENT'}] '{q}'  "
                 f"[{protocol}]  {status} ({len(matches)})  "
                 f"best-KF {best_kf:03d}")
        cv2.putText(banner, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, badge_col, 2, cv2.LINE_AA)
        if not matches:
            cv2.putText(banner, "no track has argmax-label == query "
                                  "(or relevancy < 0.5 for free-form)",
                        (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (90, 90, 90), 1, cv2.LINE_AA)
        for li, line in enumerate(legend_lines[:8]):
            colour = palette[li]
            cv2.putText(banner, line, (10, 46 + 22 * li),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        tuple(int(c) for c in colour), 1, cv2.LINE_AA)
        tiles.append(np.concatenate([banner, row], axis=0))

    if not tiles:
        return 1
    W_row = tiles[0].shape[1]
    vsep = np.full((6, W_row, 3), 220, dtype=np.uint8)
    out = tiles[0]
    for t in tiles[1:]:
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
