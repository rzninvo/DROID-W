"""Render the 83-KF CropFormer triptych as an mp4.

Style matches tests/smoke_cropformer_actual_kf9.py:
  panel 1: RGB
  panel 2: RGB + per-entity random colour overlay (0.5/0.5 blend)
  panel 3: pure mask-colour map
8-px white separator between panels.

Loads B.1 raw output (panoptic_v7_cropformer.npz) — NOT the B.1.5 refined
variant — so the visualisation matches the kf9_cropformer_actual style
the user picked.

Usage (cvg, droid-w env):
  python scripts/render_panoptic_v7_video.py \
    --npz Outputs/TUM_RGBD/freiburg3_walking_static/panoptic_v7_cropformer.npz \
    --rgb-dir datasets/TUM_RGBD/rgbd_dataset_freiburg3_walking_static/rgb \
    --out Outputs/TUM_RGBD/freiburg3_walking_static/walking_static_v7_b1.mp4 \
    --fps 6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, type=str)
    p.add_argument("--rgb-dir", required=True, type=str)
    p.add_argument("--out", required=True, type=str)
    p.add_argument("--fps", default=6, type=int)
    p.add_argument("--score-thresh", default=0.5, type=float)
    p.add_argument("--sep-px", default=8, type=int)
    args = p.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    masks_all = data["masks"]                # (N, H, W) bool
    offsets = data["seg_kf_offsets"]         # (n_kf+1,)
    scores_all = data["seg_scores"]          # (N,)
    kf_gi = data["kf_global_indices"]        # (n_kf,)
    n_kf = int(data["n_keyframes"])
    H, W = int(masks_all.shape[1]), int(masks_all.shape[2])
    img_hw_meta = (int(data["image_hw"][0]), int(data["image_hw"][1]))
    print(f"[load] {args.npz}: {n_kf} KFs, {masks_all.shape[0]} entities, "
          f"masks={H}x{W} (meta image_hw={img_hw_meta})", flush=True)

    rgb_dir = Path(args.rgb_dir)
    rgb_files = sorted(f for f in rgb_dir.iterdir() if f.suffix == ".png")
    print(f"[load] {len(rgb_files)} RGB frames in {rgb_dir}", flush=True)

    sep = np.full((H, args.sep_px, 3), 255, dtype=np.uint8)
    triptych_w = W * 3 + args.sep_px * 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(args.fps), (triptych_w, H))
    if not writer.isOpened():
        print(f"[ERR] could not open VideoWriter for {out_path}", flush=True)
        return 1

    rng = np.random.default_rng(0)

    for k in range(n_kf):
        gi = int(kf_gi[k])
        rgb_path = rgb_files[gi]
        bgr = cv2.imread(str(rgb_path))
        if bgr is None or bgr.shape[:2] != (H, W):
            if bgr is None:
                print(f"[WARN] kf {k}: cv2.imread returned None for {rgb_path}", flush=True)
                continue
            bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        s, e = int(offsets[k]), int(offsets[k + 1])
        kf_masks = masks_all[s:e]
        kf_scores = scores_all[s:e]
        keep = kf_scores >= args.score_thresh
        kf_masks = kf_masks[keep]
        n_kept = len(kf_masks)

        palette = rng.integers(64, 256, size=(max(1, n_kept + 1), 3)).astype(np.uint8)
        inst = np.zeros_like(rgb)
        for i, m in enumerate(kf_masks):
            inst[m] = palette[(i % (len(palette) - 1)) + 1]
        blend = (rgb.astype(np.float32) * 0.5 + inst.astype(np.float32) * 0.5).astype(np.uint8)

        triptych = np.concatenate([rgb, sep, blend, sep, inst], axis=1)

        label = f"KF {k:03d} / {n_kf - 1}  gi={gi:03d}  n={n_kept}"
        cv2.putText(triptych, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(triptych, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(cv2.cvtColor(triptych, cv2.COLOR_RGB2BGR))
        if (k + 1) % 10 == 0 or k == n_kf - 1:
            print(f"[render] kf {k+1}/{n_kf}  entities={n_kept}", flush=True)

    writer.release()
    print(f"[done] wrote {out_path}  ({triptych_w}x{H} @ {args.fps} fps, {n_kf} frames)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
