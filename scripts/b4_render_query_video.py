"""Render a single open-vocab query result as a per-KF MP4 video.

For each of the 83 KFs, highlight all matched tracks visible in that KF
with distinct HSV-golden-ratio colours. Triptych layout per frame:
  [RGB native] | [RGB + matched mask overlay] | [pure mask-colour]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


CANONICAL_PHRASES = ("object", "things", "stuff", "texture")


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


def lerf_relevancy(phi, q, c, T=10.0):
    s_q = q @ phi.T
    s_c = c @ phi.T
    diff = s_q[:, :, None] - s_c.T[None, :, :]
    rel = 1.0 / (1.0 + np.exp(-T * diff))
    return rel.min(axis=-1)


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
    p.add_argument("--query", required=True, type=str,
                   help="single open-vocab text query (multi-word ok)")
    p.add_argument("--tau-relevancy", default=0.5, type=float)
    p.add_argument("--temperature", default=10.0, type=float)
    p.add_argument("--top-rel-window", default=0.020, type=float)
    p.add_argument("--max-matches", default=10, type=int)
    p.add_argument("--fps", default=6, type=int)
    p.add_argument("--sep-px", default=8, type=int)
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

    valid_track_ids = np.where(np.linalg.norm(track_emb, axis=1) > 0)[0]
    valid_track_ids = valid_track_ids[valid_track_ids > 0]
    phi = track_emb[valid_track_ids]
    print(f"[setup] {len(valid_track_ids)} tracks searchable", flush=True)

    from transformers import AutoModel, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.siglip_model).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.siglip_model)

    q_emb = encode_text(model, processor, device, [args.query])
    c_emb = encode_text(model, processor, device, list(CANONICAL_PHRASES))
    rel = lerf_relevancy(phi, q_emb, c_emb, T=args.temperature).squeeze(0)

    order = np.argsort(-rel)
    passers = [(int(valid_track_ids[i]), float(rel[i])) for i in order
               if rel[i] > args.tau_relevancy]
    if not passers:
        print(f"[query] '{args.query}': NO-MATCH (top rel={rel.max():.4f})",
              flush=True)
        return 1
    top_rel = passers[0][1]
    matches = [m for m in passers if m[1] >= top_rel - args.top_rel_window]
    if len(matches) > args.max_matches:
        matches = matches[:args.max_matches]
    matched_track_ids = [m[0] for m in matches]
    print(f"[query] '{args.query}': {len(matches)} matches",
          flush=True)
    for tid, r in matches:
        print(f"   track {tid:3d}  rel={r:.4f}", flush=True)

    palette = hsv_palette(len(matches))
    track_to_colour = {tid: palette[i] for i, (tid, _) in enumerate(matches)}

    rgb_dir = Path(args.rgb_dir)
    rgb_files = sorted(f for f in rgb_dir.iterdir() if f.suffix == ".png")

    # Pre-read first KF to size the video.
    bgr0 = cv2.imread(str(rgb_files[int(kf_gi[0])]))
    H, W = bgr0.shape[:2]
    sep = np.full((H, args.sep_px, 3), 255, dtype=np.uint8)
    triptych_w = W * 3 + args.sep_px * 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(args.fps),
                             (triptych_w, H))
    if not writer.isOpened():
        print(f"[ERR] cannot open VideoWriter", flush=True)
        return 1

    # For each KF, paint matched tracks visible in that KF.
    for k_kf in range(n_kf):
        gi = int(kf_gi[k_kf])
        bgr = cv2.imread(str(rgb_files[gi]))
        if bgr is None:
            continue
        if bgr.shape[:2] != (H, W):
            bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        overlay = rgb.copy().astype(np.float32)
        pure = np.full((H, W, 3), 40, dtype=np.uint8)
        s_off, e_off = int(offsets[k_kf]), int(offsets[k_kf + 1])
        n_visible = 0
        for m_global in range(s_off, e_off):
            tid = int(gids[m_global])
            if tid not in track_to_colour:
                continue
            mask = masks[m_global]
            colour = track_to_colour[tid]
            overlay[mask] = (overlay[mask] * 0.45
                              + colour.astype(np.float32) * 0.55)
            pure[mask] = colour
            n_visible += 1
        overlay = overlay.astype(np.uint8)
        row = np.concatenate([rgb, sep, overlay, sep, pure], axis=1)
        # Annotate KF + query + visible count
        label = (f"KF {k_kf:03d}/{n_kf-1}  query='{args.query}'  "
                 f"matches={len(matches)}  visible-here={n_visible}")
        cv2.putText(row, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(row, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
        if (k_kf + 1) % 20 == 0:
            print(f"[render] kf {k_kf+1}/{n_kf}  visible-here={n_visible}",
                  flush=True)
    writer.release()
    print(f"[done] wrote {out_path}  ({triptych_w}x{H} @ {args.fps} fps, "
          f"{n_kf} frames)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
