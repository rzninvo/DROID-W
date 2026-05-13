"""Plan-v2 §Step 4 / B.2 (Path D) — DEVA→HERMES-SLAM track schema.

Reads DEVA's per-KF RGB-packed track-ID PNGs (CUSTOM.md output format,
pixel value = R + G*256 + B*65536) and the input panoptic_v7_cropformer.npz,
and writes panoptic_v7_tracks_deva.npz in the SAME flat schema as our
Fix A/Fix B trackers so the v2 renderer reuses unchanged.

Assignment rule (per CropFormer mask M_{t,j} at KF t):
  1. Load DEVA's track-id map  T_t  (480x640 int64).
  2. T_t[M_{t,j}] gives the DEVA labels under this mask.
  3. argmax-count of those labels (excluding label 0 = background) ->
     the DEVA track id we assign to mask j.
  4. If all pixels under the mask map to label 0, this CropFormer mask
     has no DEVA track and is marked as skipped (gid=-1).

DEVA track ids are large random integers; we renumber to 1..N_global
compact integers preserving identity, so the HSV palette renderer can
hash them cleanly.

Usage (cvg, droid-w env):
  python scripts/deva_to_panoptic_v7_tracks.py \
    --refined-npz Outputs/.../panoptic_v7_cropformer.npz \
    --deva-output DEVA_runs/walking_static/output/Annotations/walking_static \
    --rgb-dir     datasets/.../rgb \
    --out         Outputs/.../panoptic_v7_tracks_deva.npz
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def decode_deva_png(p: Path) -> np.ndarray:
    """RGB PNG -> int64 track-id map via id = R + G*256 + B*65536.

    Returns (H, W) int64 with 0 for background.
    """
    im = np.array(Image.open(p))
    if im.ndim == 2:
        return im.astype(np.int64)
    return im[..., 0].astype(np.int64) + im[..., 1].astype(np.int64) * 256 \
         + im[..., 2].astype(np.int64) * 65536


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--refined-npz", required=True, type=str)
    p.add_argument("--deva-output", required=True, type=str,
                   help="path to <deva_out>/Annotations/<seq>/")
    p.add_argument("--rgb-dir", required=True, type=str)
    p.add_argument("--out", required=True, type=str)
    p.add_argument("--min-pixels", default=5, type=int,
                   help="if a mask has < this many pixels mapped to a non-zero "
                        "DEVA id, mark it as skipped")
    args = p.parse_args()

    t0 = time.time()
    refined = np.load(args.refined_npz, allow_pickle=True)
    masks = refined["masks"]                # (N_total, H, W) bool
    offsets = refined["seg_kf_offsets"]
    kf_gi = refined["kf_global_indices"]
    n_kf = int(refined["n_keyframes"])
    H, W = int(masks.shape[1]), int(masks.shape[2])
    n_masks_total = int(masks.shape[0])
    print(f"[load] {n_masks_total} CropFormer masks across {n_kf} KFs",
          flush=True)

    rgb_dir = Path(args.rgb_dir)
    rgb_files = sorted(f for f in rgb_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg") and not f.name.startswith("depth"))

    deva_dir = Path(args.deva_output)
    if not deva_dir.exists():
        print(f"[ERR] DEVA output not found at {deva_dir}", flush=True)
        return 1

    global_track_ids = -np.ones(n_masks_total, dtype=np.int64)
    deva_id_to_compact: dict[int, int] = {}     # large random DEVA id -> 1..N
    next_compact = 1
    n_assigned, n_skipped = 0, 0

    for k in range(n_kf):
        gi = int(kf_gi[k])
        stem = rgb_files[gi].name[:-4]              # 1341846226.817920
        deva_png = deva_dir / f"{stem}.png"
        if not deva_png.exists():
            print(f"[WARN] kf {k}: no DEVA output at {deva_png}", flush=True)
            continue
        T = decode_deva_png(deva_png)                # (H, W) int64

        s_off, e_off = int(offsets[k]), int(offsets[k + 1])
        kf_assigned_this_step = 0
        for m_global in range(s_off, e_off):
            mask = masks[m_global]
            under = T[mask]
            under = under[under > 0]
            if under.size < args.min_pixels:
                n_skipped += 1
                continue
            # argmax-count of DEVA ids under this mask
            vals, counts = np.unique(under, return_counts=True)
            best_deva_id = int(vals[counts.argmax()])
            if best_deva_id not in deva_id_to_compact:
                deva_id_to_compact[best_deva_id] = next_compact
                next_compact += 1
            global_track_ids[m_global] = deva_id_to_compact[best_deva_id]
            n_assigned += 1
            kf_assigned_this_step += 1

        if (k + 1) % 10 == 0 or k == n_kf - 1:
            print(f"[kf {k+1:3d}/{n_kf}] tracks_so_far={next_compact-1}  "
                  f"assigned_this_kf={kf_assigned_this_step}  "
                  f"total_assigned={n_assigned}  skipped={n_skipped}",
                  flush=True)

    track_kf_counts = np.zeros(next_compact, dtype=np.int64)
    for k in range(n_kf):
        s_off, e_off = int(offsets[k]), int(offsets[k + 1])
        labs = global_track_ids[s_off:e_off]
        for l in set(labs.tolist()):
            if l > 0:
                track_kf_counts[l] += 1
    n_global_tracks = int((track_kf_counts > 0).sum())

    print(f"\n[done] {next_compact-1} raw DEVA ids -> {n_global_tracks} compact tracks",
          flush=True)
    print(f"[done] assigned={n_assigned}  skipped(<{args.min_pixels} pts)={n_skipped}",
          flush=True)
    for n_at in (1, 2, 3, 5, 10, 20, 50):
        print(f"[hist] tracks visible in >={n_at:3d} KFs: "
              f"{int((track_kf_counts >= n_at).sum())}", flush=True)
    print(f"[done] wall={time.time()-t0:.1f}s", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             schema_version=np.int64(2),
             algorithm=np.array("DEVA_propagator_v1.0_semionline", dtype="U64"),
             n_keyframes=np.int64(n_kf),
             n_masks_total=np.int64(n_masks_total),
             n_global_tracks=np.int64(n_global_tracks),
             global_track_ids=global_track_ids,
             track_kf_counts=track_kf_counts,
             min_pixels=np.int64(args.min_pixels),
             walltime_seconds=np.float32(time.time() - t0))
    print(f"[save] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
