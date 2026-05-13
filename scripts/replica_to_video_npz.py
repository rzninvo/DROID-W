"""Build a HERMES-SLAM-compatible video.npz from Replica's GT depth + poses.

For Replica eval we skip DROID-W front-end and use the ground-truth
camera poses + depth maps as if they were the front-end output. This is
the cleanest comparison to OVI-MAP Table 2 (they also use sensor depth).

Output schema matches what b3_track_embeddings_pathC.py expects:
  poses          : (n_kf, 4, 4)  cam-to-world
  droid_disps_up : (n_kf, H, W)  inverse depth at full image res
  mono_disps     : (n_kf, H, W)  same (we don't have a separate mono est.)
  intrinsics     : (n_kf, 4)     [fx, fy, cx, cy] divided by 8 (so script's
                                  multiply-by-Hd/48=H/48 gives full intrinsics)
  scale          : 1.0           depth = scale / disp = 1/disp = depth_meters
  timestamps     : (n_kf,)       frame indices in original 2000-frame seq
  images         : optional (n_kf, 3, H, W) — not used downstream
  dino_feats     : zeros (only used by DROID-W front-end which we skip)

Usage (cvg, droid-w env):
  python scripts/replica_to_video_npz.py \\
    --replica-dir /home/cvg/datasets/Replica/Replica/office0 \\
    --cam-params /home/cvg/datasets/Replica/Replica/cam_params.json \\
    --out         Outputs/Replica/office0/video.npz \\
    --kf-stride 10 --max-frames 2000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--replica-dir", required=True, type=str,
                   help="path to Replica/<scene>/ (contains results/ + traj.txt)")
    p.add_argument("--cam-params", required=True, type=str,
                   help="path to cam_params.json")
    p.add_argument("--out", required=True, type=str)
    p.add_argument("--kf-stride", default=10, type=int,
                   help="sample every N frames (OVI-MAP convention: 10)")
    p.add_argument("--max-frames", default=2000, type=int)
    args = p.parse_args()

    cam = json.load(open(args.cam_params))["camera"]
    W, H = int(cam["w"]), int(cam["h"])
    fx, fy, cx, cy = float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"])
    depth_scale = float(cam["scale"])
    print(f"[setup] image {H}x{W}, K=[fx={fx} fy={fy} cx={cx} cy={cy}], "
          f"depth_scale={depth_scale}", flush=True)

    rep_dir = Path(args.replica_dir)
    poses_all = np.loadtxt(rep_dir / "traj.txt").reshape(-1, 4, 4)
    results_dir = rep_dir / "results"
    n_total = min(args.max_frames, poses_all.shape[0])
    kf_indices = list(range(0, n_total, args.kf_stride))
    n_kf = len(kf_indices)
    print(f"[setup] {n_kf} keyframes sampled from {n_total} total "
          f"(stride={args.kf_stride})", flush=True)

    poses = np.zeros((n_kf, 4, 4), dtype=np.float32)
    disps = np.zeros((n_kf, H, W), dtype=np.float32)
    images = np.zeros((n_kf, 3, H, W), dtype=np.uint8)
    timestamps = np.array(kf_indices, dtype=np.float32)

    for k, gi in enumerate(kf_indices):
        poses[k] = poses_all[gi].astype(np.float32)
        depth_png = cv2.imread(str(results_dir / f"depth{gi:06d}.png"),
                                cv2.IMREAD_UNCHANGED)
        depth_m = depth_png.astype(np.float32) / depth_scale            # metres
        # Disp = 1/depth (with eps to avoid divide by zero on holes).
        valid = depth_m > 1e-3
        disp = np.zeros_like(depth_m)
        disp[valid] = 1.0 / depth_m[valid]                              # 1/m
        disps[k] = disp
        bgr = cv2.imread(str(results_dir / f"frame{gi:06d}.jpg"))
        if bgr is not None:
            images[k] = np.transpose(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                                      (2, 0, 1))
        if (k + 1) % 20 == 0:
            print(f"[load] {k+1}/{n_kf}", flush=True)

    # Intrinsics "at BA grid scale" (divide by 8) so b3_pathC's
    #   intrinsics_full = intr_ba * (Hd / 48.0)
    # with Hd=H gives back the full intrinsics:
    #   intr_full = intr_ba * (H/48) = (full/8) * (H/48)
    # That only works if H/48 = 8, i.e. H=384. For Replica H=680 it won't.
    # Cleaner: store intrinsics at FULL resolution; pre-divide by Hd/48
    # to cancel the script's multiply.
    intr_factor = H / 48.0
    intr_ba = np.array([fx / intr_factor, fy / intr_factor,
                         cx / intr_factor, cy / intr_factor],
                        dtype=np.float32)
    intrinsics = np.tile(intr_ba[None, :], (n_kf, 1))
    print(f"[setup] intrinsics_ba_grid (divided by H/48 = {intr_factor:.3f}): "
          f"{intr_ba} -> full = {intr_ba * intr_factor}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             images=images,
             poses=poses,
             tum_poses=np.zeros((n_kf, 7), dtype=np.float32),         # placeholder
             mono_disps=disps,
             droid_disps_up=disps,
             droid_disps=disps[:, ::8, ::8].astype(np.float32),       # decimated for compat
             intrinsics=intrinsics,
             uncertainties=np.zeros((n_kf, H // 8, W // 8), dtype=np.float32),
             dino_feats=np.zeros((n_kf, 27, 36, 384), dtype=np.float32),  # not used downstream
             scale=np.float64(1.0),                                    # depth_m already metric
             timestamps=timestamps)
    print(f"[save] {out_path}  (n_kf={n_kf}, image={H}x{W})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
