"""Plan-v2 §Step 4 / B.2 (Path D) — CropFormer→DEVA adapter.

Reads panoptic_v7_cropformer.npz and writes the DEVA "external masks"
on-disk format (CUSTOM.md / EVALUATION.md "demo" path), then symlinks
the TUM RGB stream into DEVA's img_path.

Output layout:
  DEVA_runs/<seq>/images/<timestamp>.png   <- 743 symlinks to TUM rgb/
  DEVA_runs/<seq>/masks/<timestamp>.png    <- 83 indexed PNGs (palette
                                              mode 'P', pixel value =
                                              local mask id 1..N_k)
  DEVA_runs/<seq>/masks/<timestamp>.json   <- 83 JSONs
      [{"id": k, "isthing": true, "category_id": 0, "score": float}, ...]

DEVA's detection_video_reader iterates sorted(os.listdir(image_dir)) so
the TUM timestamp filenames carry the temporal ordering correctly. The
sparse mask annotations at the 83 KF positions trigger DEVA's
detection-fusion every keyframe; the 660 unannotated frames propagate
through the memory network.

CropFormer masks may overlap (it is a top-k entity proposer, not strict
panoptic). We flatten to a single index map by painting in
score-ASCENDING order so high-confidence masks win pixel ownership.

Usage (cvg, droid-w env):
  python scripts/cropformer_to_deva.py \
    --refined-npz Outputs/.../panoptic_v7_cropformer.npz \
    --video-npz   Outputs/.../video.npz \
    --rgb-dir     datasets/.../rgb \
    --seq-name    walking_static \
    --out-root    DEVA_runs/walking_static
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--refined-npz", required=True, type=str)
    p.add_argument("--video-npz", required=True, type=str)
    p.add_argument("--rgb-dir", required=True, type=str)
    p.add_argument("--seq-name", required=True, type=str,
                   help="single-word sequence name DEVA will use as <video>")
    p.add_argument("--out-root", required=True, type=str,
                   help="output directory; will contain images/ and masks/")
    args = p.parse_args()

    refined = np.load(args.refined_npz, allow_pickle=True)
    masks = refined["masks"]                  # (N, H, W) bool
    offsets = refined["seg_kf_offsets"]        # (n_kf+1,)
    scores = refined["seg_scores"]             # (N,)
    kf_gi = refined["kf_global_indices"]       # (n_kf,)
    n_kf = int(refined["n_keyframes"])
    H, W = int(masks.shape[1]), int(masks.shape[2])
    print(f"[load] {masks.shape[0]} CropFormer masks, {n_kf} KFs, {H}x{W}",
          flush=True)

    rgb_dir = Path(args.rgb_dir)
    rgb_files = sorted(f for f in rgb_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg") and not f.name.startswith("depth"))
    print(f"[load] {len(rgb_files)} TUM RGB frames in {rgb_dir}", flush=True)

    out_root = Path(args.out_root)
    img_out = out_root / "images" / args.seq_name
    mask_out = out_root / "masks" / args.seq_name
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    # Symlink only the 83 KF RGBs (option (a): no propagation context).
    # DEVA's eval_with_detections.py requires mask+JSON on EVERY frame of
    # img_dir once per_vid_json is enabled (line 183). Feeding 660 mask-less
    # non-KF RGBs raises "Per-video json is enabled but not found". The
    # alternative is to feed mask+JSON on every frame (empty for non-KFs),
    # which buys little because DEVA's propagator is trained on contiguous
    # 24-FPS video and our ~0.3s KF spacing is already a stretch.
    n_linked = 0
    for k in range(n_kf):
        gi = int(kf_gi[k])
        src = rgb_files[gi]
        dst = img_out / src.name
        if not (dst.exists() or dst.is_symlink()):
            os.symlink(src.resolve(), dst)
            n_linked += 1
    print(f"[link] {n_linked} KF symlinks created (option a — 83 KFs only)",
          flush=True)

    # Write per-KF indexed PNG + JSON.
    max_ids_per_kf = 0
    for k in range(n_kf):
        gi = int(kf_gi[k])
        rgb_name = rgb_files[gi].name                # e.g. 1341846226.817920.png
        stem = rgb_name[:-4]                          # 1341846226.817920

        s_off, e_off = int(offsets[k]), int(offsets[k + 1])
        kf_masks = masks[s_off:e_off]
        kf_scores = scores[s_off:e_off]
        n_kept = kf_masks.shape[0]
        if n_kept > 255:
            print(f"[WARN] kf {k}: {n_kept} masks > 255 (paletted PNG limit). "
                  f"Will need long-id mode; recheck CropFormer score threshold.",
                  flush=True)

        # Paint masks in score-ASCENDING order so highest-confidence masks
        # win pixel ownership (overwrite earlier paints).
        order = np.argsort(kf_scores)
        idx_map = np.zeros((H, W), dtype=np.uint8)
        for rank, m_i in enumerate(order):
            idx_map[kf_masks[m_i]] = m_i + 1     # +1 so 0 = background

        # Save as palette PNG.
        im = Image.fromarray(idx_map, mode="P")
        # Build a simple palette (DEVA does not require it visually correct,
        # but giving each id a distinct colour helps debugging).
        palette = np.zeros((256, 3), dtype=np.uint8)
        rng = np.random.default_rng(k)
        palette[1:n_kept + 1] = rng.integers(64, 256, size=(n_kept, 3),
                                              dtype=np.uint8)
        im.putpalette(palette.flatten().tolist())
        im.save(mask_out / f"{stem}.png")

        # JSON: one entry per LOCAL mask id present in idx_map.
        present = np.unique(idx_map)
        present = present[present > 0]
        meta = [{"id": int(i), "isthing": True, "category_id": 0,
                 "score": float(kf_scores[i - 1])}
                for i in present]
        with open(mask_out / f"{stem}.json", "w") as fh:
            json.dump(meta, fh)
        max_ids_per_kf = max(max_ids_per_kf, len(meta))

        if (k + 1) % 10 == 0 or k == n_kf - 1:
            print(f"[write] kf {k+1}/{n_kf}  ({stem}) {n_kept} masks "
                  f"-> {len(meta)} present ids", flush=True)

    print(f"\n[done] {n_kf} KF mask pairs written to {mask_out}", flush=True)
    print(f"[done] {n_linked} RGB symlinks at {img_out}", flush=True)
    print(f"[done] max ids per kf = {max_ids_per_kf} "
          f"(palette PNG safe up to 255)", flush=True)
    print(f"\nNext: cd /home/cvg/HERMES-SLAM/DEVA && "
          f"python evaluation/eval_with_detections.py \\\n"
          f"  --img_path  {out_root}/images \\\n"
          f"  --mask_path {out_root}/masks \\\n"
          f"  --dataset demo --temporal_setting semionline \\\n"
          f"  --output    {out_root}/output \\\n"
          f"  --chunk_size 4 --amp",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
