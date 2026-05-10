"""
Phase B' — interactive open-vocab querying of saved RADSeg features.

Loads `<scene>/radseg_features.npz` (Phase B' precomputed lang-aligned
features) + `<scene>/video.npz` (RGB) and answers any text query in
milliseconds: text-encode → cosine → softmax → argmax.

Reuses the upstream RADSeg recipe verbatim — 80-template OpenAI ImageNet
ensemble, temperature-100 softmax, optional prompt-denoising — so masks
match what `RadioGrounder.segment(rgb)` would have produced *with
sam_refinement=False*.

IMPORTANT: SAM-3 refinement is NOT applied here. Phase B (radseg_masks.npz)
masks DID include SAM-3, so they will not match Phase B' query output —
expect ~5 pp higher coverage in Phase B' on classes SAM-3 was strict on
(consistent with sam3_refinement.py's score_threshold=0.7 suppressing
low-confidence instances). To get SAM-3-refined masks for new queries,
re-run scripts/precompute_radseg_masks.py with the new --queries.

Usage (cvg):
    python scripts/query_radseg_features.py \\
        --scene Outputs/TUM_RGBD/freiburg3_walking_static \\
        --queries person hand laptop "white wall" carpet \\
        --threshold 0.50 \\
        --output Outputs/TUM_RGBD/freiburg3_walking_static/query_test.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.mono_priors.radseg.radseg_encoder import RADSegEncoder

# Same palette as scripts/test_radio_segmentation.py for visual continuity.
_PALETTE_BGR = [
    ( 60,  60, 230), ( 60, 230,  60), (230,  60,  60),
    ( 60, 230, 230), (230,  60, 230), (230, 230,  60),
    ( 60, 150, 230), (200, 100, 230),
]


def _build_text_only_encoder(radio_version: str, lang_adaptor: str, device: str) -> RADSegEncoder:
    """Spin up the same RADSeg encoder used at precompute time so its
    `encode_labels()` (80-template ensemble) produces the SAME text
    embeddings — required to keep feature/text spaces aligned.

    We pay the RADIO image-tower load cost (~10-15s, ~3.5 GB) but never
    call its image path. Cleaner factoring would be a stand-alone
    `RADSegTextEncoder` that loads only the lang_adaptor sub-module, but
    that's upstream API surgery (`torch.hub.load("NVlabs/RADIO", ...)`
    returns a coupled bundle).

    NB: `sam3=is_v4` is required even though we don't refine — v4-h's
    adaptor list is ['siglip2-g', 'dino_v3_7b', 'sam3'], so passing
    sam3=False would request an absent "sam" adaptor and crash. The
    SAM-3 forward pass itself is gated separately by `sam_refinement`."""
    is_v4 = "v4" in radio_version.lower()
    return RADSegEncoder(
        device=device,
        model_version=radio_version,
        lang_model=lang_adaptor,
        return_radio_features=True,
        compile=False,
        amp=False,
        predict=False,
        sam_refinement=False,            # no SAM forward at runtime
        sam3=is_v4,                      # required for v4-h adaptor naming only
        sam_ckpt="",
    )


def _overlay_mask(bgr, mask, color):
    if not mask.any():
        return bgr
    out = bgr.copy()
    cb = np.array(color, dtype=np.uint8)
    out[mask] = (0.55 * out[mask] + 0.45 * cb).astype(np.uint8)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, type=Path)
    p.add_argument("--queries", required=True, nargs="+",
                   help="One or more text queries (e.g. person hand laptop floor).")
    p.add_argument("--threshold", default=0.50, type=float)
    p.add_argument("--prompt-denoising-thresh", default=0.5, type=float,
                   help="Per-image max-softmax floor; classes whose max softmax over "
                        "the keyframe is below this are zeroed (radseg paper default).")
    p.add_argument("--features", default=None, type=Path,
                   help="Path to radseg_features.npz. Default: <scene>/radseg_features.npz")
    p.add_argument("--video", default=None, type=Path,
                   help="Path to video.npz. Default: <scene>/video.npz")
    p.add_argument("--output", default=None, type=Path,
                   help="Output combined.mp4 path. Default: <scene>/query_<first_query>.mp4")
    p.add_argument("--fps", default=5, type=int)
    p.add_argument("--device", default="cuda:0", type=str)
    args = p.parse_args()

    scene = args.scene.resolve()
    feats_path = args.features.resolve() if args.features else scene / "radseg_features.npz"
    video_path = args.video.resolve() if args.video else scene / "video.npz"
    if args.output is None:
        safe = args.queries[0].replace(" ", "_").replace("/", "_")
        out_path = scene / f"query_{safe}.mp4"
    else:
        out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[setup] features={feats_path}", flush=True)
    print(f"[setup] queries={args.queries}  threshold={args.threshold}", flush=True)

    m = np.load(feats_path)
    feats = m["lang_aligned_feats"]                # (N, D, h, w) fp16
    radio_version = str(m["radio_version"])
    lang_adaptor = str(m["lang_adaptor"])
    N, D, hp, wp = feats.shape
    H, W = int(m["image_hw"][0]), int(m["image_hw"][1])
    print(f"[setup] feats {feats.shape}  radio={radio_version}  lang={lang_adaptor}",
          flush=True)

    # Text encode (same recipe as RADSeg paper: 80-template ensemble,
    # mean over templates, L2 normalize).
    t_text = time.time()
    encoder = _build_text_only_encoder(radio_version, lang_adaptor, args.device)
    # Provenance equality check — Reviewer A audit (Report 16): the dim
    # check alone (line below) is necessary but insufficient — two adaptors
    # with same dim but different weights would silently produce
    # mis-aligned cosines.
    runtime_radio = encoder.model_version
    runtime_lang = encoder.lang_adaptor.__class__.__name__
    if runtime_radio != radio_version:
        print(f"[ERR] runtime radio={runtime_radio} != saved radio={radio_version}; "
              f"feature/text spaces incompatible. Aborting.", flush=True)
        return 1
    with torch.no_grad():
        text_emb = encoder.encode_labels(args.queries, onehot=False)  # (Q, D)
    text_emb = text_emb.to(torch.float32).contiguous()
    text_emb = F.normalize(text_emb, dim=-1)
    Q = text_emb.shape[0]
    print(f"[text] encoded {Q} queries in {time.time() - t_text:.1f}s  text_emb dim={text_emb.shape[1]}",
          flush=True)
    if text_emb.shape[1] != D:
        print(f"[ERR] text dim {text_emb.shape[1]} != feature dim {D} — feature/text "
              f"spaces incompatible. Did the lang_adaptor change since precompute?",
              flush=True)
        return 1

    # Free encoder GPU memory (we only needed encode_labels).
    del encoder
    torch.cuda.empty_cache()

    # Load images.
    z = np.load(video_path)
    images = z["images"]
    if images.dtype != np.uint8:
        images = (images * 255.0).clip(0, 255).astype(np.uint8)
    if len(images) < N:
        print(f"[WARN] features: video.npz has {len(images)} keyframes but "
              f"features file has {N}; using min", flush=True)
        N = min(N, len(images))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    BANNER_H = 50
    vw = cv2.VideoWriter(str(out_path), fourcc, args.fps, (W, H + BANNER_H))

    colors = [_PALETTE_BGR[i % len(_PALETTE_BGR)] for i in range(Q)]

    text_emb_g = text_emb.to(args.device)
    n_pixels_per_query = np.zeros(Q, dtype=np.int64)

    t_loop = time.time()
    for i in range(N):
        # (D, h, w) fp16 → (D, h*w) fp32 normalized
        f = torch.from_numpy(feats[i]).to(args.device, dtype=torch.float32)
        f = f.view(D, -1)
        f = F.normalize(f, dim=0)

        # Cosine: (Q, h*w)
        cos_sim = text_emb_g @ f                                        # (Q, hw)
        cos_sim = cos_sim.view(Q, hp, wp)

        # 100x temperature softmax over Q (RADSeg paper).
        sim = torch.softmax(100.0 * cos_sim, dim=0)                     # (Q, hp, wp)

        # Prompt denoising: zero out classes whose per-image max softmax is below threshold.
        if args.prompt_denoising_thresh > 0:
            max_per_class, _ = sim.view(Q, -1).max(dim=1)               # (Q,)
            low_conf = max_per_class < args.prompt_denoising_thresh
            sim = sim * (~low_conf)[:, None, None]

        # Upsample similarity from feat-grid → image res.
        sim_up = F.interpolate(sim.unsqueeze(0), size=(H, W),
                               mode="bilinear", align_corners=False)    # (1, Q, H, W)
        best_score, best_query = sim_up.squeeze(0).max(dim=0)           # (H, W) each
        below = best_score < args.threshold
        best_query = torch.where(below, torch.full_like(best_query, -1), best_query)
        best_query = best_query.cpu().numpy()
        best_score = best_score.cpu().numpy()

        rgb = images[i].transpose(1, 2, 0)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out_img = bgr.copy()
        for qi in range(Q):
            mq = best_query == qi
            if mq.any():
                out_img = _overlay_mask(out_img, mq, colors[qi])
                n_pixels_per_query[qi] += int(mq.sum())

        banner = np.zeros((BANNER_H, W, 3), dtype=np.uint8)
        cv2.putText(banner, f"KF {i:3d}/{N}  source: radseg_features.npz (text-time query, no RADIO re-run)",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(banner, f"thr={args.threshold:.2f}  {radio_version}+{lang_adaptor}",
                    (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,220,255), 1, cv2.LINE_AA)
        legend_x = W - 8 - 100
        cy0 = 12
        for qi, q in enumerate(args.queries):
            chip_y = cy0 + qi * 8
            if chip_y + 4 > BANNER_H - 2:
                break
            cv2.rectangle(banner, (legend_x, chip_y - 4), (legend_x + 8, chip_y + 2),
                          tuple(int(c) for c in colors[qi]), -1)
            cv2.putText(banner, q, (legend_x + 12, chip_y + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255,255,255), 1, cv2.LINE_AA)

        vw.write(np.vstack([banner, out_img]))

        del f, cos_sim, sim, sim_up, best_score, best_query

    vw.release()
    walltime = time.time() - t_loop
    print(f"[done] {N} keyframes in {walltime:.1f}s "
          f"(mean {1000*walltime/N:.1f} ms/kf, no RADIO forward)", flush=True)
    print(f"[save] {out_path}  ({out_path.stat().st_size/1024/1024:.1f} MB)", flush=True)
    print(f"\n=== per-query coverage (across {N} keyframes) ===", flush=True)
    total = N * H * W
    for qi, q in enumerate(args.queries):
        print(f"  [{qi}] {q:<25} : {100*n_pixels_per_query[qi]/total:5.2f}% pixels", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
