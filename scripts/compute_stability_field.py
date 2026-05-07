"""
Phase A — offline stability-field + Barron-alpha visualization.

Faithful port of the per-pixel temporal-stability mechanism from
RADIO-ViPE upstream (Yakovlev et al., arXiv 2604.26067, April 2026):

    vipe/slam/ba/terms.py:467-504   →  pairwise cosine similarity via grid_sample
    vipe/slam/ba/terms.py:506-532   →  S(u) = mean_j(cs) * (1 - var_j(cs))
    vipe/slam/ba/terms.py:538-572   →  three-regime piecewise alpha mapping
    vipe/slam/ba/terms.py:578-612   →  edge alpha = min(S_i, S_j); we just take S_i
                                       directly because we visualise per-frame S

This is the non-SLAM-loop diagnostic: load `video.npz`, reproject each
keyframe's pixels into a temporal window of neighbours using saved
DROID-W pose + depth, sample DINOv2 (or RADIO if `--use-radio`)
features at the reprojected locations, take cosine similarities, and
build the stability field S per keyframe. Visualise S and alpha as
overlay videos.

Usage:
    python scripts/compute_stability_field.py \\
        --scene Outputs/TUM_RGBD/freiburg3_walking_static \\
        --window 5

Outputs (under <scene>/stability/):
    stability.mp4   — per-keyframe heatmap of S overlaid on RGB
    alpha.mp4       — same but with the three Barron regimes colour-coded
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Pose helpers — match seg_model._pose_to_Rt convention exactly.
# tum_poses are stored as world-to-camera 7-vec (tx,ty,tz,qx,qy,qz,qw)
# per DROID-W; we invert below to get camera-to-world for unprojection.
# ----------------------------------------------------------------------

def _quat_to_R(qx, qy, qz, qw):
    n = qx*qx + qy*qy + qz*qz + qw*qw
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    return np.array([
        [1 - s*(qy*qy+qz*qz),   s*(qx*qy-qz*qw),   s*(qx*qz+qy*qw)],
        [    s*(qx*qy+qz*qw), 1 - s*(qx*qx+qz*qz), s*(qy*qz-qx*qw)],
        [    s*(qx*qz-qy*qw),   s*(qy*qz+qx*qw), 1 - s*(qx*qx+qy*qy)],
    ], dtype=np.float64)


def _T_world_from_cam(pose_7):
    """Inverse of stored cam_T_world → returns world_T_cam (4×4)."""
    tx, ty, tz, qx, qy, qz, qw = pose_7
    T_cw = np.eye(4, dtype=np.float64)
    T_cw[:3, :3] = _quat_to_R(qx, qy, qz, qw)
    T_cw[:3, 3] = (tx, ty, tz)
    return np.linalg.inv(T_cw)


def _T_cam_from_world(pose_7):
    """The stored cam_T_world directly (4×4)."""
    tx, ty, tz, qx, qy, qz, qw = pose_7
    T_cw = np.eye(4, dtype=np.float64)
    T_cw[:3, :3] = _quat_to_R(qx, qy, qz, qw)
    T_cw[:3, 3] = (tx, ty, tz)
    return T_cw


# ----------------------------------------------------------------------
# Build per-pair correspondence grid via reprojection.
# ----------------------------------------------------------------------

@torch.no_grad()
def _correspondence_grid(depth_i, T_w_i, T_j_w, K, device):
    """For each pixel (u,v) in frame i, return (u',v') in frame j as a
    grid_sample-ready normalized tensor (1, H, W, 2) ∈ [-1, 1].

    Pixels with negative depth in frame j or out-of-bounds reprojection
    are flagged in `valid` (1, H, W) and zeroed in the grid (sampled to
    border).
    """
    H, W = depth_i.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    us, vs = torch.meshgrid(
        torch.arange(W, device=device, dtype=torch.float64),
        torch.arange(H, device=device, dtype=torch.float64),
        indexing="xy",
    )
    z_i = depth_i
    valid_z = z_i > 1e-3

    # Camera-frame point in i.
    X_cam_i = torch.stack([
        (us - cx) * z_i / fx,
        (vs - cy) * z_i / fy,
        z_i,
    ], dim=-1)  # (H, W, 3)

    # Lift to world.
    T_w_i_t = torch.tensor(T_w_i, device=device, dtype=torch.float64)
    T_j_w_t = torch.tensor(T_j_w, device=device, dtype=torch.float64)
    R_w_i, t_w_i = T_w_i_t[:3, :3], T_w_i_t[:3, 3]
    R_j_w, t_j_w = T_j_w_t[:3, :3], T_j_w_t[:3, 3]
    X_w = X_cam_i @ R_w_i.T + t_w_i  # (H, W, 3)

    # Project into j.
    X_cam_j = X_w @ R_j_w.T + t_j_w  # (H, W, 3)
    z_j = X_cam_j[..., 2]
    valid = valid_z & (z_j > 1e-3)
    z_safe = z_j.clamp_min(1e-3)
    u_j = fx * X_cam_j[..., 0] / z_safe + cx
    v_j = fy * X_cam_j[..., 1] / z_safe + cy
    in_bounds = (u_j >= 0) & (u_j < W) & (v_j >= 0) & (v_j < H)
    valid = valid & in_bounds

    # Normalize to [-1, 1] for grid_sample.
    grid_u = u_j * 2.0 / (W - 1) - 1.0
    grid_v = v_j * 2.0 / (H - 1) - 1.0
    grid = torch.stack([grid_u, grid_v], dim=-1).unsqueeze(0).float()  # (1, H, W, 2)
    return grid, valid.unsqueeze(0)  # (1, H, W)


# ----------------------------------------------------------------------
# Stability field computation.
# Mirrors vipe/slam/ba/terms.py:_compute_all_cosine_sims +
# _compute_stability_fields.
# ----------------------------------------------------------------------

@torch.no_grad()
def stability_for_scene(images, poses, depths, K, dino_feats,
                        window: int, device: str,
                        upstream_zero_invalid: bool = True):
    """Compute per-keyframe stability field S of shape (N, H, W).

    Per upstream RADIO-ViPE (vipe/slam/ba/terms.py:457-532):
        cs[i,j,u] = cosine_sim( feat_i[u], grid_sample(feat_j, coord_ij[u]) )
        — and upstream multiplies by valid_map (line 500), so invalid
          pixels contribute 0 to the cosine sum. The mean/var are then
          taken over ALL n_neighbors (not just valid ones). This biases
          low-overlap pixels toward low S — see the corner-red artifact.

        S_i(u)    = mean_j(cs[i,j,u]) * (1 - var_j(cs[i,j,u]))

    `upstream_zero_invalid=True`  → match upstream exactly: divide by
                                    n_neighbors, zeros for invalid.
    `upstream_zero_invalid=False` → divide by n_valid only (less biased
                                    at boundaries; differs from upstream).
    """
    N, _, H, W = images.shape
    feat_h, feat_w = dino_feats.shape[1], dino_feats.shape[2]
    print(f"[stability] N={N}, image={H}x{W}, feat={feat_h}x{feat_w} "
          f"(D={dino_feats.shape[3]}), window=±{window}, "
          f"upstream_zero_invalid={upstream_zero_invalid}", flush=True)

    feats_t = torch.from_numpy(dino_feats).float().to(device)        # (N, h, w, D)
    feats_t = feats_t.permute(0, 3, 1, 2).contiguous()               # (N, D, h, w)
    feats_t = F.normalize(feats_t, dim=1, eps=1e-8)

    sum_cs = torch.zeros(N, H, W, device=device, dtype=torch.float32)
    sum_cs2 = torch.zeros(N, H, W, device=device, dtype=torch.float32)
    n_valid = torch.zeros(N, H, W, device=device, dtype=torch.float32)
    n_total = torch.zeros(N, device=device, dtype=torch.float32)  # window-bounded count

    t0 = time.time()
    for i in range(N):
        T_w_i = _T_world_from_cam(poses[i])
        depth_i = torch.from_numpy(depths[i]).double().to(device)
        feat_i_full = F.interpolate(
            feats_t[i:i+1], size=(H, W), mode="bilinear", align_corners=True,
        )                                                            # (1, D, H, W)

        n_neigh_i = 0
        for j in range(max(0, i - window), min(N, i + window + 1)):
            if j == i:
                continue
            T_j_w = _T_cam_from_world(poses[j])
            grid, valid = _correspondence_grid(
                depth_i, T_w_i, T_j_w, K, device,
            )
            feat_j_full = F.interpolate(
                feats_t[j:j+1], size=(H, W), mode="bilinear", align_corners=True,
            )                                                        # (1, D, H, W)
            sampled = F.grid_sample(
                feat_j_full, grid, mode="bilinear",
                padding_mode="border", align_corners=True,
            )                                                        # (1, D, H, W)
            sampled = F.normalize(sampled, dim=1, eps=1e-8)

            cs = (feat_i_full * sampled).sum(dim=1, keepdim=False)   # (1, H, W)
            v = valid.float()
            cs_v = cs * v   # zero out invalid (upstream line 500)

            sum_cs[i] += cs_v[0]
            sum_cs2[i] += (cs_v * cs_v)[0]   # squared-with-zero matches upstream var
            n_valid[i] += v[0]
            n_neigh_i += 1
        n_total[i] = float(n_neigh_i)

        if i % 10 == 0:
            print(f"  KF {i:3d}/{N}  ({time.time()-t0:5.1f}s)", flush=True)

    if upstream_zero_invalid:
        # Divide by total window neighbours (n_total broadcast per-frame).
        denom = n_total.view(N, 1, 1).clamp_min(1.0)
    else:
        denom = n_valid.clamp_min(1.0)

    mean_cs = sum_cs / denom
    mean_cs2 = sum_cs2 / denom
    var_cs = (mean_cs2 - mean_cs * mean_cs).clamp_min(0.0)
    S = (mean_cs * (1.0 - var_cs)).clamp(0.0, 1.0)

    n_zero = int((n_valid.sum(dim=(1, 2)) == 0).sum().item())
    if n_zero > 0:
        print(f"[WARN] stability: {n_zero} keyframes had zero overlap "
              f"with the temporal window — their S is left at 0", flush=True)
    return S.cpu().numpy(), n_valid.cpu().numpy()


# ----------------------------------------------------------------------
# Three-regime alpha mapping (faithful port of _stability_to_alpha).
# ----------------------------------------------------------------------

def stability_to_alpha(S: np.ndarray,
                      thresh_movable: float = 0.35,
                      thresh_static: float = 0.75,
                      alpha_dynamic: float = -2.0,  # Geman-McClure
                      alpha_huber: float = 1.0,     # Huber-ish
                      alpha_static: float = 2.0):   # L2
    """Piecewise-linear S → α via two lerp segments. Differentiable
    everywhere.

    Defaults match upstream RADIO-ViPE:
        vipe/slam/ba/terms.py:145-149   class default kwargs
        vipe/slam/components/buffer.py:529-533  config getter defaults
    Note α=-2 is Geman-McClure (Barron's family), more aggressive than
    Cauchy (α=0) at suppressing outliers."""
    S = np.clip(S, 0.0, 1.0)
    t_lo = thresh_movable
    t_hi = thresh_static
    t_mid = 0.5 * (t_lo + t_hi)

    t_lower = np.clip((S - t_lo) / max(t_mid - t_lo, 1e-6), 0.0, 1.0)
    t_upper = np.clip((S - t_mid) / max(t_hi - t_mid, 1e-6), 0.0, 1.0)
    alpha = alpha_dynamic + (alpha_huber - alpha_dynamic) * t_lower
    alpha = alpha + (alpha_static - alpha_huber) * t_upper
    return alpha


# ----------------------------------------------------------------------
# Visualisation.
# ----------------------------------------------------------------------

def _heatmap_overlay(rgb_bgr, scalar_0_1, colormap=cv2.COLORMAP_TURBO,
                     blend: float = 0.55):
    """Overlay a heatmap (0..1 scalar) on a BGR image."""
    h = (np.clip(scalar_0_1, 0, 1) * 255).astype(np.uint8)
    h_color = cv2.applyColorMap(h, colormap)
    return cv2.addWeighted(rgb_bgr, blend, h_color, 1.0 - blend, 0)


def _alpha_overlay(rgb_bgr, alpha, mask=None):
    """Three-regime colour code.
    α near -2 → red (dynamic)
    α near 1  → yellow (movable)
    α near 2  → blue (static)
    Pixels where mask is False are passed through (no overlay) so they
    don't get a spurious dynamic colour.
    """
    H, W = alpha.shape
    # Normalise alpha from [-2, 2] to [0, 1] for colour mapping.
    a01 = np.clip((alpha + 2.0) / 4.0, 0.0, 1.0)
    out = np.zeros((H, W, 3), dtype=np.uint8)
    # B (static / blue) ↑ as a01 ↑
    out[..., 0] = (255 * a01).astype(np.uint8)
    # R (dynamic / red) ↑ as a01 ↓
    out[..., 2] = (255 * (1.0 - a01)).astype(np.uint8)
    # G (movable / yellow) peaks at a01 ≈ 0.75 (α=1)
    g = 1.0 - np.abs(a01 - 0.75) * 4.0
    out[..., 1] = (255 * np.clip(g, 0.0, 1.0)).astype(np.uint8)
    blended = cv2.addWeighted(rgb_bgr, 0.55, out, 0.45, 0)
    if mask is not None:
        m3 = mask.astype(bool)[..., None]
        blended = np.where(m3, blended, rgb_bgr)
    return blended


# ----------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--window", type=int, default=5,
                   help="±N temporal-window for cosine sims (default 5)")
    p.add_argument("--min-overlap", type=int, default=1,
                   help="Pixels with valid-neighbour count < this are masked "
                        "(rendered as raw RGB, NOT mapped to dynamic). "
                        "Set 1 to keep upstream behaviour (corners red); "
                        "raise to suppress boundary false-positives.")
    p.add_argument("--upstream-zero", action="store_true", default=True,
                   help="Match upstream: divide by total window neighbours "
                        "(zeros for invalid). Default True (faithful port).")
    p.add_argument("--no-upstream-zero", dest="upstream_zero",
                   action="store_false",
                   help="Divide by valid count only (less boundary bias, "
                        "non-faithful).")
    p.add_argument("--thresh-movable", type=float, default=0.35)
    p.add_argument("--thresh-static", type=float, default=0.75)
    p.add_argument("--tag", type=str, default="",
                   help="Suffix for output files (e.g. 'w5_mo3' produces "
                        "stability_w5_mo3.mp4)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--max-keyframes", type=int, default=None)
    args = p.parse_args()

    scene = args.scene.resolve()
    out_dir = scene / "stability"
    out_dir.mkdir(exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    print(f"[setup] scene={scene}  output={out_dir}  tag={args.tag or '(none)'}", flush=True)

    npz = np.load(scene / "video.npz")
    images = npz["images"]
    if images.dtype != np.uint8:
        images = (images * 255).clip(0, 255).astype(np.uint8)
    poses = npz["tum_poses"]
    disps_coarse = npz["droid_disps"]      # (N, h_c, w_c) — at DROID coarse 1/8 grid
    disps_up = npz["droid_disps_up"]       # (N, H, W) — full image resolution
    depths = 1.0 / np.clip(disps_up, 1e-3, None)
    intr = npz["intrinsics"]
    if intr.ndim == 2 and intr.shape[1] == 4:
        fx, fy, cx, cy = intr[0]
    else:
        fx, fy, cx, cy = float(intr[0,0]), float(intr[1,1]), float(intr[0,2]), float(intr[1,2])

    if "dino_feats" not in npz.files:
        print(f"[ERR] {scene}/video.npz has no 'dino_feats' field — re-run "
              f"SLAM with mono_prior.save_feature: True or "
              f"tracking.uncertainty_params.activate: True", flush=True)
        return 1
    dino_feats = npz["dino_feats"]   # (N, h, w, D)

    N, _, H, W = images.shape
    if args.max_keyframes is not None:
        N = min(N, args.max_keyframes)
        images = images[:N]; poses = poses[:N]; depths = depths[:N]; dino_feats = dino_feats[:N]
        disps_coarse = disps_coarse[:N]

    # Stored intrinsics in video.npz are at the DROID coarse 1/8 grid
    # (matches `droid_disps`, NOT `droid_disps_up`). We use depths at
    # image resolution (depth_up = 1/disps_up), so K must be scaled up
    # from the coarse grid to image resolution.
    ch_coarse, cw_coarse = disps_coarse.shape[1], disps_coarse.shape[2]
    sx, sy = W / cw_coarse, H / ch_coarse        # typically 8.0 for DROID
    K = np.array([[fx*sx, 0, cx*sx], [0, fy*sy, cy*sy], [0, 0, 1]], dtype=np.float64)
    print(f"[setup] N={N}, image={H}x{W}, coarse={ch_coarse}x{cw_coarse}, "
          f"K_image=fx{K[0,0]:.1f} cx{K[0,2]:.1f} (scale {sx:.2f}x)", flush=True)

    S, n_overlap = stability_for_scene(
        images, poses, depths, K, dino_feats,
        window=args.window, device=args.device,
        upstream_zero_invalid=args.upstream_zero,
    )

    alpha = stability_to_alpha(S, thresh_movable=args.thresh_movable,
                               thresh_static=args.thresh_static)
    valid_mask = n_overlap >= args.min_overlap        # (N, H, W) bool

    # ------------------------------------------------------------------
    # Diagnostics: corner vs center stats. Corner = outer 5% border;
    # center = inner 50% (radius). Reported only for VALID pixels.
    # ------------------------------------------------------------------
    border_w = max(1, int(W * 0.05)); border_h = max(1, int(H * 0.05))
    corner_mask = np.zeros((H, W), dtype=bool)
    corner_mask[:border_h, :] = True; corner_mask[-border_h:, :] = True
    corner_mask[:, :border_w] = True; corner_mask[:, -border_w:] = True
    yy, xx = np.mgrid[:H, :W]
    cx_, cy_ = W / 2.0, H / 2.0
    rad = np.sqrt(((xx - cx_) / (W/2))**2 + ((yy - cy_) / (H/2))**2)
    center_mask = rad <= 0.5

    valid_corner = valid_mask & corner_mask[None, ...]
    valid_center = valid_mask & center_mask[None, ...]
    n_corner = int(valid_corner.sum()); n_center = int(valid_center.sum())
    if n_corner > 0:
        red_corner = float((S[valid_corner] < args.thresh_movable).mean()) * 100.0
        mean_S_corner = float(S[valid_corner].mean())
    else:
        red_corner = mean_S_corner = float("nan")
    if n_center > 0:
        red_center = float((S[valid_center] < args.thresh_movable).mean()) * 100.0
        mean_S_center = float(S[valid_center].mean())
    else:
        red_center = mean_S_center = float("nan")
    masked_pct = 100.0 * (1.0 - float(valid_mask.mean()))
    print(f"[diag] tag={args.tag or '(none)'}  window=±{args.window}  "
          f"min_overlap={args.min_overlap}  upstream_zero={args.upstream_zero}  "
          f"thresh_mov={args.thresh_movable}  thresh_static={args.thresh_static}",
          flush=True)
    print(f"[diag] masked_pct={masked_pct:5.2f}%  "
          f"corner_red%={red_corner:5.2f}  center_red%={red_center:5.2f}  "
          f"mean_S_corner={mean_S_corner:.3f}  mean_S_center={mean_S_center:.3f}",
          flush=True)

    # Save raw arrays for downstream / regression.
    npz_path = out_dir / f"stability{suffix}.npz"
    np.savez(npz_path,
             S=S, alpha=alpha, n_overlap=n_overlap, valid_mask=valid_mask,
             window=args.window, min_overlap=args.min_overlap,
             upstream_zero=args.upstream_zero,
             thresh_movable=args.thresh_movable,
             thresh_static=args.thresh_static)
    print(f"[save] {npz_path}  S min={S.min():.3f} max={S.max():.3f} mean={S.mean():.3f}",
          flush=True)

    # Render videos.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    s_path = out_dir / f"stability{suffix}.mp4"
    a_path = out_dir / f"alpha{suffix}.mp4"
    o_path = out_dir / f"n_overlap{suffix}.mp4"
    s_w = cv2.VideoWriter(str(s_path), fourcc, args.fps, (W, H))
    a_w = cv2.VideoWriter(str(a_path), fourcc, args.fps, (W, H))
    o_w = cv2.VideoWriter(str(o_path), fourcc, args.fps, (W, H))
    n_max = max(1.0, float(n_overlap.max()))
    for i in range(N):
        bgr = cv2.cvtColor(images[i].transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        s_w.write(_heatmap_overlay(bgr, S[i]))
        a_w.write(_alpha_overlay(bgr, alpha[i], mask=valid_mask[i]))
        o_w.write(_heatmap_overlay(bgr, n_overlap[i] / n_max))
    s_w.release(); a_w.release(); o_w.release()
    print(f"[done] {s_path}\n[done] {a_path}\n[done] {o_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
