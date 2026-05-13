"""Plan-v2 §Step 4 / B.1.5 v8 — OVI-MAP code-faithful depth-refinement.

Upgrades v7 with TWO improvements ranked by OVI-MAP-faithfulness:

  FIX #1 -- tighter Sobel edge threshold tau_edge = 0.10 m/px (was 0.15).
       Matches OVI-MAP's tighter discontinuity ratio (RGB-D: ~0.05).
       Improves chair-seat-vs-floor edge detection (the floor-bleed
       Agent B measured at ~16 percent on track 7).

  FIX #2 -- 13x13 PCA surface normals + 24-neighbor concavity test.
       Ports OVI-MAP's depth_segmentation.cpp:476-617 verbatim.
       Their primary geometric-segmentation signal is convexity from
       normals, NOT depth gradient. Catches surface bends (chair seat
       to back) where depth is continuous but the surface curves --
       a failure mode pure Sobel misses.

Reference (primary source, file:line cited per CLAUDE.md sec.7):
  ovi-map-ref/mapping_ros_ws/.../depth_segmentation/depth_segmentation/
    src/depth_segmentation.cpp:292-301   3-D point map via cv2.rgbd.depthTo3d
    src/depth_segmentation.cpp:448-475   normals (kDepthWindowFilter 13x13 PCA)
    common.h:243-302                       computeOwnNormals impl
    src/depth_segmentation.cpp:304-352   discontinuity via dilate-minus-erode
    src/depth_segmentation.cpp:476-617   min-convexity (24-neighbor concavity)
    src/depth_segmentation.cpp:623-665   edge = convexity - min(disc+dist, 1)
  Hyperparameters (common.h):
    discontinuity_ratio = 0.01 / 0.05    (we keep ours plus add a ratio test)
    mask_threshold = -0.0005             (concavity threshold)
    normals window_size = 13
    min_size = 500 (CC)

Pipeline (per KF):
  1. depth = scale / max(mono_disps, eps), Gaussian sigma=1.5, clip [0.1, 10] m.
  2. 3-D points via cv2.rgbd.depthTo3d on the smoothed depth + scaled intrinsics.
  3. Surface normals via 13x13 PCA covariance (vectorized batched eigh).
  4. Depth-disc gradient map (Sobel on smoothed depth, tau_edge = 0.10 m/px).
  5. Convexity-via-normal map: 24-neighbor concavity test, threshold -0.0005.
  6. Combined edge = sobel_edge OR concavity_edge.
  7. For each CropFormer entity mask: erode by edge map, find CCs, group by
     depth, split into K children (same as v7 logic).

Output: panoptic_v7_cropformer_refined_v8.npz (same schema as v7).

Note: keeps mono_disps as the depth source. Agent B's empirical data showed
mono_disps is sharper at chair boundaries (edge-sharpness ratio 1.24 vs
droid_disps_up's 1.06). Switching to droid_disps_up would HURT, not help.

Usage (cvg, droid-w env):
  python scripts/refine_panoptic_with_depth_v8_normals.py \\
    --in-npz   Outputs/.../panoptic_v7_cropformer.npz \\
    --video-npz Outputs/.../video.npz \\
    --out      Outputs/.../panoptic_v7_cropformer_refined_v8.npz \\
    --tau-edge 0.10 --concavity-thresh -0.0005 --min-cc-size 100
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from numpy.lib.stride_tricks import sliding_window_view


def compute_normals_gradient(points_3d: np.ndarray,
                              valid: np.ndarray) -> np.ndarray:
    """Fast cross-product normals (OVI-MAP kDepthGradient mode, equivalent
    to OpenCV cv2.rgbd.RgbdNormals with method=0).

    Reference: depth_segmentation_node.cpp:108-119 -- OVI-MAP offers
    kDepthGradient as a faster alternative to kDepthWindowFilter (PCA).
    For our 480x640 mono-disparity, PCA on 13x13 windows is ~minutes/KF;
    cross-product gradients is ~10 ms/KF and good enough for the
    chair-seat-vs-floor concavity case (large angle change).

    Algorithm: dx, dy of 3D points via Sobel; normal = (dy x dx) /  |..|.
    Re-oriented toward camera per OVI-MAP common.h:264-275.
    """
    # Sobel of each 3D component.
    sx = lambda c: cv2.Sobel(c, cv2.CV_32F, 1, 0, ksize=3)
    sy = lambda c: cv2.Sobel(c, cv2.CV_32F, 0, 1, ksize=3)
    dx = np.stack([sx(points_3d[..., 0]),
                    sx(points_3d[..., 1]),
                    sx(points_3d[..., 2])], axis=-1)
    dy = np.stack([sy(points_3d[..., 0]),
                    sy(points_3d[..., 1]),
                    sy(points_3d[..., 2])], axis=-1)
    normals = np.cross(dy, dx)
    norm_mag = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-9
    normals = (normals / norm_mag).astype(np.float32)
    # Re-orient toward camera: flip where n . p > 0 (OVI-MAP common.h:264-275).
    flip = (normals * points_3d).sum(axis=-1) > 0
    normals[flip] = -normals[flip]
    normals[~valid] = 0.0
    return normals


def compute_concavity_map(points_3d: np.ndarray,
                          normals: np.ndarray,
                          kernel: int = 5) -> np.ndarray:
    """OVI-MAP min-convexity via 24-neighbor concavity test
    (depth_segmentation.cpp:476-617).

    For each pixel, examines 24 neighbors in a 5x5 window. For each neighbor:
        vec = (p_neighbor - p_self) / |...|
        concavity_offset = -(vec . n_neighbor)
    Takes the MINIMUM concavity across the 24 offsets. Strong negative values
    (< -0.0005) indicate sharp surface concavity (a "trough" or bend).

    Returns (H, W) float, more negative = more concave/edge-like.
    """
    H, W, _ = points_3d.shape
    half = kernel // 2
    convexity = np.full((H, W), np.inf, dtype=np.float32)
    valid_pts = np.linalg.norm(points_3d, axis=-1) > 1e-6
    valid_nrm = np.linalg.norm(normals, axis=-1) > 1e-6

    for di in range(-half, half + 1):
        for dj in range(-half, half + 1):
            if di == 0 and dj == 0:
                continue
            # Roll neighbor data to align with current pixel.
            p_neigh = np.roll(points_3d, shift=(-di, -dj), axis=(0, 1))
            n_neigh = np.roll(normals, shift=(-di, -dj), axis=(0, 1))
            valid_n = np.roll(valid_pts & valid_nrm,
                              shift=(-di, -dj), axis=(0, 1))

            vec = p_neigh - points_3d
            vec_norm = np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9
            vec_unit = vec / vec_norm
            # Concavity at this offset: -(vec_unit . n_neighbor)
            concav = -(vec_unit * n_neigh).sum(axis=-1)                # (H, W)
            # Only update where both pixels are valid.
            valid = valid_pts & valid_nrm & valid_n
            concav[~valid] = np.inf  # don't update
            convexity = np.minimum(convexity, concav)

    # Where convexity == inf (no valid neighbors), set to 0 (no signal).
    convexity[convexity == np.inf] = 0.0
    return convexity


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in-npz", required=True, type=str,
                   help="B.1 raw CropFormer npz")
    p.add_argument("--video-npz", required=True, type=str)
    p.add_argument("--out", required=True, type=str)
    p.add_argument("--tau-edge", default=0.10, type=float,
                   help="Sobel-on-depth gradient threshold (m/px). "
                        "FIX #1: 0.10 (tighter than v7's 0.15).")
    p.add_argument("--concavity-thresh", default=-0.0005, type=float,
                   help="FIX #2: OVI-MAP common.h:88 mask_threshold")
    p.add_argument("--normals-window", default=13, type=int,
                   help="OVI-MAP common.h:42 normals.window_size")
    p.add_argument("--min-cc-size", default=100, type=int,
                   help="Minimum CC pixel count to spawn a child mask")
    p.add_argument("--depth-overlap-thresh", default=0.20, type=float,
                   help="Depth gap (m) above which a CC starts a new group")
    p.add_argument("--gauss-sigma", default=1.5, type=float)
    p.add_argument("--depth-clip-min", default=0.1, type=float)
    p.add_argument("--depth-clip-max", default=10.0, type=float)
    p.add_argument("--disp-eps", default=1e-3, type=float)
    args = p.parse_args()

    FORMAT = '%(asctime)s.%(msecs)06d %(levelname)-8s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%H:%M:%S')
    logging.info(f"v8 params: tau_edge={args.tau_edge} concav={args.concavity_thresh} "
                 f"normals_win={args.normals_window} min_cc={args.min_cc_size}")

    t0 = time.time()
    raw = np.load(args.in_npz, allow_pickle=True)
    video = np.load(args.video_npz, allow_pickle=True)

    masks = raw["masks"]                       # (N, H_m, W_m) bool
    offsets = raw["seg_kf_offsets"]
    scores = raw["seg_scores"]
    kf_gi = raw["kf_global_indices"]
    n_kf = int(raw["n_keyframes"])
    image_hw = raw["image_hw"]
    H_m, W_m = int(masks.shape[1]), int(masks.shape[2])

    mono_disps = video["mono_disps"]            # (n_kf, H_d, W_d)
    intr_ba = video["intrinsics"]               # (n_kf, 4) at 48x64
    scale = float(video["scale"])
    Hd, Wd = mono_disps.shape[1], mono_disps.shape[2]
    intrinsics_full = intr_ba * (Hd / 48.0)     # at depth res
    logging.info(f"input: {masks.shape[0]} masks @ {H_m}x{W_m}, "
                 f"depth @ {Hd}x{Wd}, scale={scale:.4f}")

    # Output buffers.
    out_masks_list = []
    out_offsets = [0]
    out_scores = []
    n_split, n_kept = 0, 0

    for k_kf in range(n_kf):
        # 1. Depth from disparity.
        disp = mono_disps[k_kf].astype(np.float32)
        depth = scale / np.maximum(disp, args.disp_eps)
        depth = gaussian_filter(depth, sigma=args.gauss_sigma)
        depth = np.clip(depth, args.depth_clip_min, args.depth_clip_max)

        # 2. 3-D point map (cv2.rgbd.depthTo3d equivalent — manual since
        #    cv2.rgbd isn't in opencv-python without contrib).
        fx, fy, cx, cy = intrinsics_full[k_kf]
        u_grid, v_grid = np.meshgrid(np.arange(Wd, dtype=np.float32),
                                      np.arange(Hd, dtype=np.float32))
        z = depth.astype(np.float32)
        x = (u_grid - cx) * z / fx
        y = (v_grid - cy) * z / fy
        points = np.stack([x, y, z], axis=-1)                  # (Hd, Wd, 3)
        valid_pt = np.abs(points[..., 2]) > 1e-3

        # 3. Cross-product gradient normals (OVI-MAP kDepthGradient mode,
        #    fast alternative to PCA per depth_segmentation_node.cpp:108).
        normals = compute_normals_gradient(points, valid_pt)

        # 4. Depth-disc Sobel (FIX #1: tau_edge=0.10).
        gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx * gx + gy * gy)
        sobel_edge = grad_mag > args.tau_edge                            # (Hd, Wd)

        # 5. Concavity (FIX #2: OVI-MAP depth_segmentation.cpp:476-617).
        concav = compute_concavity_map(points, normals, kernel=5)
        concav_edge = concav < args.concavity_thresh

        # 6. Combined edge.
        edge_map = sobel_edge | concav_edge

        # Resize edge_map to mask resolution (480x640) if different.
        if (Hd, Wd) != (H_m, W_m):
            edge_at_mask = cv2.resize(edge_map.astype(np.uint8),
                                       (W_m, H_m),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
            depth_at_mask = cv2.resize(depth.astype(np.float32),
                                        (W_m, H_m),
                                        interpolation=cv2.INTER_LINEAR)
        else:
            edge_at_mask = edge_map
            depth_at_mask = depth

        s_off, e_off = int(offsets[k_kf]), int(offsets[k_kf + 1])
        for m_global in range(s_off, e_off):
            mask = masks[m_global]
            score = float(scores[m_global])
            # Carve out edge pixels.
            mask_no_edge = mask & ~edge_at_mask
            # Find CCs.
            n_cc, lab_map = cv2.connectedComponents(
                mask_no_edge.astype(np.uint8))
            # Group CCs by depth.
            cc_descs = []                          # (mean_depth, mask)
            for lab in range(1, n_cc):
                cc_mask = (lab_map == lab)
                if cc_mask.sum() < args.min_cc_size:
                    continue
                cc_depth = depth_at_mask[cc_mask]
                cc_descs.append((float(cc_depth.mean()), cc_mask))
            if len(cc_descs) <= 1:
                # No split needed (or none viable). Keep original mask.
                out_masks_list.append(mask)
                out_scores.append(score)
                n_kept += 1
                continue
            # Sort by depth.
            cc_descs.sort(key=lambda x: x[0])
            # Group by depth gap.
            groups = [[cc_descs[0]]]
            for i in range(1, len(cc_descs)):
                prev_max = max(g[0] for g in groups[-1])
                if cc_descs[i][0] - prev_max > args.depth_overlap_thresh:
                    groups.append([cc_descs[i]])
                else:
                    groups[-1].append(cc_descs[i])
            if len(groups) <= 1:
                out_masks_list.append(mask)
                out_scores.append(score)
                n_kept += 1
                continue
            # Materialise one child per group.
            for g in groups:
                child_mask = np.zeros_like(mask)
                for _, cc in g:
                    child_mask |= cc
                out_masks_list.append(child_mask)
                out_scores.append(score)
            n_split += 1
        out_offsets.append(len(out_masks_list))

        if (k_kf + 1) % 10 == 0:
            logging.info(f"[kf {k_kf+1:3d}/{n_kf}] "
                          f"split={n_split} kept={n_kept} "
                          f"elapsed={time.time()-t0:.1f}s")

    out_masks_arr = np.array(out_masks_list, dtype=bool)
    out_offsets_arr = np.array(out_offsets, dtype=np.int64)
    out_scores_arr = np.array(out_scores, dtype=np.float32)
    logging.info(f"DONE: {len(out_masks_list)} output masks "
                  f"(vs {masks.shape[0]} input); split={n_split} kept={n_kept}; "
                  f"wall={time.time()-t0:.1f}s")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             schema_version=np.int64(2),
             proposer=np.array("CropFormer_swinL_3x_v8_pca_concav", dtype="U64"),
             n_keyframes=np.int64(n_kf),
             image_hw=image_hw,
             kf_global_indices=kf_gi,
             masks=out_masks_arr,
             seg_kf_offsets=out_offsets_arr,
             seg_scores=out_scores_arr,
             tau_edge=np.float32(args.tau_edge),
             concavity_thresh=np.float32(args.concavity_thresh),
             normals_window=np.int64(args.normals_window),
             min_cc_size=np.int64(args.min_cc_size),
             depth_overlap_thresh=np.float32(args.depth_overlap_thresh),
             walltime_seconds=np.float32(time.time() - t0))
    logging.info(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
