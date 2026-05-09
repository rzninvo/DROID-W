"""
Sanity-check `<scene>/radseg_masks.npz` against a fresh
`RadioGrounder.segment()` call on a small random sample of keyframes.

The encoder runs at amp=False with model.eval(), and SAM-3 hard-replaces
the noisy RADIO grounding output (radseg_encoder.py:355-357), so the
pipeline is genuinely deterministic — EXACT roundtrip is the expected
and required behaviour. Any deviation is a bug, not fp16 noise.

Verifies:
- File schema is complete (all expected fields present)
- best_query/best_score shapes match (n_keyframes, H, W)
- best_query values are in [-1, len(queries) - 1]
- Re-running grounder.segment() on K random keyframes yields THE SAME
  best_query / best_score, with separate checks for:
    - exact bq agreement (per-pixel, must be >= 99.9% per CLAUDE.md §0)
    - -1 sentinel agreement (catches threshold drift)
    - per-query IoU >= 0.99 (catches whole-class flips)

Usage (cvg):
    python scripts/verify_radseg_masks.py \\
        --scene Outputs/TUM_RGBD/freiburg3_walking_static \\
        --n-spotcheck 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.mono_priors.radio_grounding import RadioGrounder


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, type=Path)
    p.add_argument("--masks", default=None, type=Path,
                   help="Default: <scene>/radseg_masks.npz")
    p.add_argument("--n-spotcheck", default=3, type=int,
                   help="How many random keyframes to re-ground for roundtrip.")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--device", default="cuda:0", type=str)
    args = p.parse_args()

    scene = args.scene.resolve()
    masks_path = args.masks if args.masks is not None else scene / "radseg_masks.npz"
    video_path = scene / "video.npz"
    print(f"[setup] masks={masks_path}", flush=True)

    m = np.load(masks_path)
    expected = {
        "best_query", "best_score", "queries", "threshold",
        "radio_version", "lang_adaptor", "sam_refinement",
        "n_keyframes", "image_hw",
    }
    missing = expected - set(m.files)
    if missing:
        print(f"[FAIL] {masks_path} missing fields: {missing}", flush=True)
        return 1
    print(f"[ok] all expected fields present: {sorted(m.files)}", flush=True)

    bq = m["best_query"]
    bs = m["best_score"]
    queries = list(m["queries"])
    threshold = float(m["threshold"])
    Q = len(queries)
    n_kf = int(m["n_keyframes"])
    H, W = int(m["image_hw"][0]), int(m["image_hw"][1])

    if bq.shape != (n_kf, H, W):
        print(f"[FAIL] best_query shape {bq.shape} != ({n_kf}, {H}, {W})",
              flush=True)
        return 1
    if bs.shape != (n_kf, H, W):
        print(f"[FAIL] best_score shape {bs.shape} != ({n_kf}, {H}, {W})",
              flush=True)
        return 1
    print(f"[ok] shapes consistent: ({n_kf}, {H}, {W})", flush=True)

    bq_min = int(bq.min()); bq_max = int(bq.max())
    if bq_min < -1 or bq_max >= Q:
        print(f"[FAIL] best_query range [{bq_min}, {bq_max}] not in [-1, {Q-1}]",
              flush=True)
        return 1
    print(f"[ok] best_query range [{bq_min}, {bq_max}] is in [-1, {Q-1}]",
          flush=True)

    # ── Roundtrip check ────────────────────────────────────────────
    z = np.load(video_path)
    images = z["images"]
    if images.dtype != np.uint8:
        images = (images * 255.0).clip(0, 255).astype(np.uint8)
    if len(images) < n_kf:
        print(f"[WARN] video.npz has {len(images)} keyframes but mask file "
              f"says {n_kf}; spot-checking on min({n_kf}, {len(images)})",
              flush=True)

    grounder = RadioGrounder(
        device=args.device,
        radio_version=str(m["radio_version"]),
        lang_adaptor=str(m["lang_adaptor"]),
        sam_refinement=bool(m["sam_refinement"]),
    )
    grounder.set_queries(queries)

    rng = np.random.default_rng(args.seed)
    idx_to_check = rng.choice(min(n_kf, len(images)),
                              size=min(args.n_spotcheck, n_kf, len(images)),
                              replace=False)
    print(f"[setup] spot-checking keyframes {sorted(idx_to_check.tolist())}",
          flush=True)

    BQ_PCT_GATE = 99.9          # tightened from 99.0 — pipeline is deterministic
    SENTINEL_GATE = 99.9         # -1 ↔ -1 agreement separately
    PER_QUERY_IOU_GATE = 0.99    # catches whole-class flips
    n_pass = 0
    for kf_idx in idx_to_check:
        rgb = images[int(kf_idx)].transpose(1, 2, 0)
        result = grounder.segment(rgb)
        live_bq = result["best_query"]
        live_bs = result["best_score"]
        below = live_bs < threshold
        live_bq = np.where(below, -1, live_bq).astype(np.int16)
        live_bs_fp16 = live_bs.astype(np.float16)

        saved_bq = bq[int(kf_idx)]
        saved_bs = bs[int(kf_idx)]

        # Pixel-level bq agreement.
        bq_pct = float((saved_bq == live_bq).mean()) * 100.0
        # bs at fp16 precision (lossy roundtrip).
        bs_l2 = float(np.linalg.norm(
            saved_bs.astype(np.float32) - live_bs_fp16.astype(np.float32)
        )) / np.sqrt(saved_bs.size)
        # Sentinel agreement (catches threshold drift).
        sent_match = (saved_bq == -1) == (live_bq == -1)
        sent_pct = float(sent_match.mean()) * 100.0
        # Per-query IoU (catches whole-class flips).
        per_q_iou = []
        for qi in range(Q):
            sm = (saved_bq == qi); lm = (live_bq == qi)
            if not (sm.any() or lm.any()):
                continue
            inter = int((sm & lm).sum())
            union = int((sm | lm).sum())
            per_q_iou.append(inter / max(1, union))
        min_iou = min(per_q_iou) if per_q_iou else 1.0

        ok_bq = bq_pct >= BQ_PCT_GATE
        ok_sent = sent_pct >= SENTINEL_GATE
        ok_iou = min_iou >= PER_QUERY_IOU_GATE
        if ok_bq and ok_sent and ok_iou:
            tag = "PASS" if not (bq_pct == 100.0 and sent_pct == 100.0 and min_iou == 1.0) else "EXACT"
            print(f"  [{int(kf_idx):4d}] {tag}  bq={bq_pct:6.2f}%  "
                  f"sent={sent_pct:6.2f}%  min_iou={min_iou:.4f}  bs_l2={bs_l2:.4f}",
                  flush=True)
            n_pass += 1
        else:
            print(f"  [{int(kf_idx):4d}] FAIL  bq={bq_pct:6.2f}% (gate {BQ_PCT_GATE})  "
                  f"sent={sent_pct:6.2f}% (gate {SENTINEL_GATE})  "
                  f"min_iou={min_iou:.4f} (gate {PER_QUERY_IOU_GATE})  "
                  f"bs_l2={bs_l2:.4f}",
                  flush=True)

    print(f"\n=== verify summary ===", flush=True)
    print(f"  {n_pass}/{len(idx_to_check)} keyframes passed roundtrip",
          flush=True)
    if n_pass < len(idx_to_check):
        print(f"  [FAIL] saved masks deviate from live grounder — pipeline "
              f"determinism assumed (amp=False, eval mode); deviation is a bug.",
              flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
