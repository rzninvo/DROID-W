"""
Step 0.c verifier (plan-v2): cross-view semantic-residual sanity.

For DROID-W's stored KF correspondences, the semantic residual
    r_sem(u) = 1 - cos(F_i(u), F_j(mu_ij(u)))
must be lower on rigid STATIC regions than on moving / dynamic regions
(Goal C's binary mask defines the split). If this holds by a meaningful
margin (gate: median dynamic / median static >= 1.5), the features carry
geometry-correlated signal and Steps 2/3 may proceed. If not: STOP per
plan-v2 Step 0.c.

Pipeline:
  - Resample lang_aligned_feats from (D, h_native, w_native) to BA grid
    (D, H_ba, W_ba) once per KF.
  - For each (i, j) pair, backproject KF i's BA pixels with droid_disps[i]
    and poses[i], forward project to KF j with poses[j], bilinearly sample
    feats_ba[j] at mu_ij, compute r_sem on valid pixels.
  - Downsample Goal C's per-frame mask (image res) to BA grid via NN, use
    it to split r_sem into static vs dynamic.
  - Aggregate over pairs; report medians + IQRs and the ratio.

Pose convention: this script tries the standard DROID-SLAM convention
(poses is T_w2c, i.e. world-to-camera). If the median flow magnitude on
walking_static is non-trivially > 5 px after projection (camera is nearly
static here), the script flips the convention and re-runs to confirm.

Usage (cvg, droid-w conda env):
    python scripts/verify_step0c_cross_view_sanity.py \\
        --features Outputs/TUM_RGBD/freiburg3_walking_static/radseg_features.npz \\
        --video    Outputs/TUM_RGBD/freiburg3_walking_static/video.npz \\
        --masks    datasets/TUM_RGBD/rgbd_dataset_freiburg3_walking_static/dynamic_masks.npz
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cudnn.deterministic = True
torch.manual_seed(0)


def _parse_pair(s: str) -> tuple[int, int]:
    a, b = s.split(",")
    return int(a), int(b)


def _project(disps_ba_i: torch.Tensor, K_ba: torch.Tensor,
             T_ij: torch.Tensor, H_ba: int, W_ba: int):
    """Project pixel grid of KF i to KF j using disp_i and T_ij = T_j @ inv(T_i)."""
    u_grid = torch.arange(W_ba).float()
    v_grid = torch.arange(H_ba).float()
    vv, uu = torch.meshgrid(v_grid, u_grid, indexing="ij")
    z_i = 1.0 / disps_ba_i.clamp(min=1e-3)
    x_i = (uu - K_ba[2]) / K_ba[0] * z_i
    y_i = (vv - K_ba[3]) / K_ba[1] * z_i
    pts = torch.stack([x_i, y_i, z_i, torch.ones_like(z_i)], dim=-1).reshape(-1, 4)
    pts_j = (pts @ T_ij.T).reshape(H_ba, W_ba, 4)
    z_j = pts_j[..., 2].clamp(min=1e-3)
    mu_u = K_ba[0] * pts_j[..., 0] / z_j + K_ba[2]
    mu_v = K_ba[1] * pts_j[..., 1] / z_j + K_ba[3]
    valid = (mu_u >= 0) & (mu_u < W_ba) & (mu_v >= 0) & (mu_v < H_ba) & (pts_j[..., 2] > 0)
    return mu_u, mu_v, valid


def _run(poses_inv_first: bool, *, feats_ba_n, disps_ba, poses, intrinsics_full,
         masks_per_frame, kf_indices, static_is_true, pairs, H_ba, W_ba,
         H_img, W_img):
    """Run the histogram aggregation with a specified pose convention."""
    scale_x = W_ba / W_img
    scale_y = H_ba / H_img
    flow_mags = []
    r_static_all, r_dynamic_all = [], []
    valid_frac_all = []

    for i, j in pairs:
        T_i = poses[i]
        T_j = poses[j]
        # poses_inv_first=True means: T_ij = T_j @ inv(T_i)  (DROID convention, T_w2c)
        # poses_inv_first=False means: T_ij = inv(T_j) @ T_i  (T_c2w convention)
        if poses_inv_first:
            T_ij = T_j @ torch.linalg.inv(T_i)
        else:
            T_ij = torch.linalg.inv(T_j) @ T_i

        K_ba = intrinsics_full[i].clone()
        K_ba[0] *= scale_x; K_ba[1] *= scale_y
        K_ba[2] *= scale_x; K_ba[3] *= scale_y

        mu_u, mu_v, valid = _project(disps_ba[i], K_ba, T_ij, H_ba, W_ba)

        u_grid = torch.arange(W_ba).float().expand(H_ba, W_ba)
        v_grid = torch.arange(H_ba).float().unsqueeze(1).expand(H_ba, W_ba)
        flow_mag = torch.sqrt((mu_u - u_grid) ** 2 + (mu_v - v_grid) ** 2)
        flow_mags.append(float(flow_mag[valid].median()))

        # grid_sample expects normalised [-1, 1]
        mu_x_n = 2.0 * mu_u / (W_ba - 1) - 1.0
        mu_y_n = 2.0 * mu_v / (H_ba - 1) - 1.0
        grid = torch.stack([mu_x_n, mu_y_n], dim=-1).unsqueeze(0)
        F_j_at_mu = F.grid_sample(feats_ba_n[j].unsqueeze(0), grid,
                                  mode="bilinear", padding_mode="border",
                                  align_corners=False).squeeze(0)
        F_j_at_mu = F_j_at_mu / (F_j_at_mu.norm(dim=0, keepdim=True) + 1e-8)
        cs = (feats_ba_n[i] * F_j_at_mu).sum(dim=0)
        r_sem = (1.0 - cs).clamp(min=0)

        frame_i = int(kf_indices[i])
        mask_full = torch.from_numpy(masks_per_frame[frame_i].astype(np.uint8))
        mask_ba = F.interpolate(mask_full.unsqueeze(0).unsqueeze(0).float(),
                                size=(H_ba, W_ba), mode="nearest").squeeze() > 0.5
        static_mask = mask_ba if static_is_true else ~mask_ba
        dyn_mask = ~static_mask

        static_mask = static_mask & valid
        dyn_mask = dyn_mask & valid

        valid_frac_all.append(float(valid.float().mean()))
        r_static_all.append(r_sem[static_mask].cpu().numpy())
        r_dynamic_all.append(r_sem[dyn_mask].cpu().numpy())

    return (np.concatenate(r_static_all) if r_static_all else np.array([]),
            np.concatenate(r_dynamic_all) if r_dynamic_all else np.array([]),
            np.array(flow_mags), np.array(valid_frac_all))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--masks", required=True, type=Path)
    p.add_argument("--pairs", nargs="+", type=_parse_pair,
                   default=[(5, 10), (15, 20), (25, 30), (40, 45), (55, 60), (70, 75)])
    p.add_argument("--ba-stride", default=8, type=int)
    p.add_argument("--gate-ratio", default=1.5, type=float)
    p.add_argument("--out-npz", default=None, type=Path,
                   help="Optional: dump per-pair r_sem arrays for later plotting")
    args = p.parse_args()

    feats_d = np.load(args.features, allow_pickle=False)
    video_d = np.load(args.video, allow_pickle=False)
    masks_d = np.load(args.masks, allow_pickle=False)

    feats_native = torch.from_numpy(feats_d["lang_aligned_feats"]).float()
    kf_indices = torch.from_numpy(feats_d["kf_indices"]).long()
    poses = torch.from_numpy(video_d["poses"]).float()
    disps_ba = torch.from_numpy(video_d["droid_disps"]).float()
    intrinsics_full = torch.from_numpy(video_d["intrinsics"]).float()
    masks_per_frame = masks_d["mask"]
    n_dyn = masks_d["n_dyn_per_frame"]

    N_kf, D, h_n, w_n = feats_native.shape
    _, H_ba, W_ba = disps_ba.shape
    H_img, W_img = masks_per_frame.shape[1:]

    # Determine mask convention: True=STATIC or True=DYNAMIC?
    frame_test = int(kf_indices[10])
    static_count = int(masks_per_frame[frame_test].sum())
    total_px = H_img * W_img
    static_is_true = (total_px - static_count) == int(n_dyn[frame_test])
    print(f"[step0c] mask convention check on frame {frame_test}: "
          f"True={'STATIC' if static_is_true else 'DYNAMIC'}", flush=True)
    print(f"[step0c] N_kf={N_kf}, D={D}, native=({h_n},{w_n}), "
          f"BA grid=({H_ba},{W_ba}), image=({H_img},{W_img})", flush=True)
    print(f"[step0c] mask='{masks_d['queries']}' radio={masks_d['radio_version']}", flush=True)

    # Resample features and normalise once
    feats_ba = F.interpolate(feats_native, size=(H_ba, W_ba), mode="bilinear",
                             align_corners=False)
    feats_ba_n = feats_ba / (feats_ba.norm(dim=1, keepdim=True) + 1e-8)

    # Try standard DROID convention first (poses = T_w2c -> T_ij = T_j @ inv(T_i))
    print(f"[step0c] running with convention A: T_ij = T_j @ inv(T_i)", flush=True)
    r_static_a, r_dyn_a, flows_a, validf_a = _run(
        True, feats_ba_n=feats_ba_n, disps_ba=disps_ba, poses=poses,
        intrinsics_full=intrinsics_full, masks_per_frame=masks_per_frame,
        kf_indices=kf_indices, static_is_true=static_is_true, pairs=args.pairs,
        H_ba=H_ba, W_ba=W_ba, H_img=H_img, W_img=W_img,
    )
    print(f"[step0c]   median flow per pair (px): "
          f"{[f'{x:.2f}' for x in flows_a]}", flush=True)
    print(f"[step0c]   valid fraction per pair  : "
          f"{[f'{x:.2f}' for x in validf_a]}", flush=True)

    # If projection produces wildly inconsistent flow (median > 10 px on
    # walking_static where the camera is near-stationary), the pose
    # convention is likely the other way around — try and compare.
    if float(np.median(flows_a)) > 10.0:
        print(f"[step0c] flow too large for walking_static; "
              f"trying convention B: T_ij = inv(T_j) @ T_i", flush=True)
        r_static_b, r_dyn_b, flows_b, validf_b = _run(
            False, feats_ba_n=feats_ba_n, disps_ba=disps_ba, poses=poses,
            intrinsics_full=intrinsics_full, masks_per_frame=masks_per_frame,
            kf_indices=kf_indices, static_is_true=static_is_true, pairs=args.pairs,
            H_ba=H_ba, W_ba=W_ba, H_img=H_img, W_img=W_img,
        )
        print(f"[step0c] B flow per pair (px): "
              f"{[f'{x:.2f}' for x in flows_b]}", flush=True)
        if float(np.median(flows_b)) < float(np.median(flows_a)):
            print("[step0c] adopting convention B", flush=True)
            r_static, r_dynamic = r_static_b, r_dyn_b
        else:
            print("[step0c] sticking with convention A", flush=True)
            r_static, r_dynamic = r_static_a, r_dyn_a
    else:
        r_static, r_dynamic = r_static_a, r_dyn_a

    if r_static.size == 0 or r_dynamic.size == 0:
        print("[ERR] empty static or dynamic set", flush=True)
        return 2

    med_s = float(np.median(r_static))
    med_d = float(np.median(r_dynamic))
    p25_s = float(np.percentile(r_static, 25))
    p75_s = float(np.percentile(r_static, 75))
    p25_d = float(np.percentile(r_dynamic, 25))
    p75_d = float(np.percentile(r_dynamic, 75))
    ratio = med_d / max(med_s, 1e-9)

    print(f"[step0c] n_static_px={len(r_static)}, n_dynamic_px={len(r_dynamic)}",
          flush=True)
    print(f"[step0c] r_sem^static : median={med_s:.4f}  IQR=({p25_s:.4f},{p75_s:.4f})",
          flush=True)
    print(f"[step0c] r_sem^dynamic: median={med_d:.4f}  IQR=({p25_d:.4f},{p75_d:.4f})",
          flush=True)
    print(f"[step0c] ratio (dyn/static): {ratio:.3f}  (gate: >= {args.gate_ratio})",
          flush=True)

    if args.out_npz is not None:
        args.out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out_npz, r_static=r_static, r_dynamic=r_dynamic,
                 median_static=med_s, median_dynamic=med_d, ratio=ratio,
                 pairs=np.array(args.pairs))
        print(f"[step0c] dumped per-pixel arrays -> {args.out_npz}", flush=True)

    if ratio >= args.gate_ratio:
        print(f"[step0c] PASS  ({ratio:.3f} >= {args.gate_ratio})", flush=True)
        return 0

    print(f"[WARN] step0c cross_view_sanity: expected dyn/static median ratio "
          f">= {args.gate_ratio}, got {ratio:.3f}, fallback=STOP "
          f"(features do not carry geometry-correlated signal; do not "
          f"proceed to Steps 2/3 per plan-v2)", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
