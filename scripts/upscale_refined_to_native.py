"""Upscale B.1.5 refined masks from BA-grid 384×512 to TUM native 480×640
via nearest-neighbour. Output keeps the same schema as B.1 raw npz so
downstream scripts (cropformer_to_deva, Path C, etc.) work unchanged.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--refined-npz", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--target-h", default=480, type=int)
    p.add_argument("--target-w", default=640, type=int)
    args = p.parse_args()

    d = np.load(args.refined_npz, allow_pickle=True)
    masks = d["masks"]               # (N, 384, 512) bool
    N, H, W = masks.shape
    print(f"[load] {N} masks at {H}x{W} -> upscaling to {args.target_h}x{args.target_w}",
          flush=True)
    out_masks = np.zeros((N, args.target_h, args.target_w), dtype=bool)
    for i in range(N):
        out_masks[i] = cv2.resize(
            masks[i].astype(np.uint8),
            (args.target_w, args.target_h),
            interpolation=cv2.INTER_NEAREST).astype(bool)
        if (i + 1) % 500 == 0:
            print(f"[upscale] {i+1}/{N}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             schema_version=d["schema_version"],
             proposer=d["proposer"],
             n_keyframes=d["n_keyframes"],
             image_hw=np.array([args.target_h, args.target_w], dtype=np.int64),
             kf_global_indices=d["kf_global_indices"],
             masks=out_masks,
             seg_kf_offsets=d["seg_kf_offsets"],
             seg_scores=d["seg_scores"],
             walltime_seconds=d.get("walltime_seconds", np.float32(0)))
    print(f"[save] {out_path}  (upscaled to {args.target_h}x{args.target_w})",
          flush=True)


if __name__ == "__main__":
    main()
