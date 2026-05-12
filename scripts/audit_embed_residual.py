"""
Plan-v2 Step 3a (D.1a) — detached log of E_embed per KF-graph edge.

For each KF pair (i, j) in a synthetic temporal-window edge graph, compute:
  - per-edge mean r_embed^2 = mean_u (1 - cos(F_i(u), F_j(mu_ij(u))))^2
  - per-edge ATE proxy: max(err_i, err_j) where err_k is the GT-aligned
    per-KF position error (mm).

Then Spearman correlation between the two. **Gate (plan-v2 §Step 3a):
rho >= 0.2 to proceed to D.1b** (the active residual). Below that,
features are too smooth or geometry-correlation is too weak; D.1b will
add noise rather than constraint.

Inputs (per scene output dir):
  - video.npz                : poses (N_kf,7) tum_poses (tx ty tz qx qy qz qw),
                               droid_disps (N_kf, h_ba, w_ba), intrinsics,
                               timestamps (N_kf,)
  - gt_poses.txt             : TUM format (timestamp tx ty tz qx qy qz qw)
  - traj/est_poses_full.txt  : frame_idx tx ty tz qx qy qz qw  (one row per frame)
  - traj/metrics_full_traj.txt : has scale / rotation / translation alignment
  - radseg_features.npz      : lang_aligned_feats (N_kf_precomp, D, h_n, w_n) fp16

Optionally takes a PCA basis to project features to K=256 before cosine
(matches D.2 paradigm).

Usage (cvg, droid-w env):
  python scripts/audit_embed_residual.py \
    --scene Outputs/TUM_RGBD/freiburg3_walking_xyz_d2_off \
    --pca-basis weights/pca_basis.pt \
    --window 5 \
    --out Outputs/TUM_RGBD/freiburg3_walking_xyz_d2_off/embed_residual_audit_v0.npz
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cudnn.deterministic = True


# ─── Trajectory utilities ────────────────────────────────────────────────────

def _read_tum_poses(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read TUM-format trajectory: timestamp tx ty tz qx qy qz qw."""
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            rows.append([float(x) for x in parts])
    arr = np.asarray(rows, dtype=np.float64)
    return arr[:, 0], arr[:, 1:8]                                      # (N,) , (N, 7)


def _parse_metrics_alignment(path: Path) -> tuple[float, np.ndarray, np.ndarray]:
    """Parse scale / rotation (3x3) / translation (3,) from metrics_*.txt."""
    text = path.read_text()
    scale_m = re.search(r"scale:\s*([0-9eE.+-]+)", text)
    rot_m = re.search(r"rotation:\s*\n\[\[(.+?)\]\]", text, re.DOTALL)
    tr_m = re.search(r"translation:\s*\[([^\]]+)\]", text)
    scale = float(scale_m.group(1))
    rot_str = "[[" + rot_m.group(1) + "]]"
    # Strip the [[ ]] and parse 3x3 numbers
    inner = rot_m.group(1)
    rows = [r for r in inner.replace("[", "").replace("]", "").splitlines() if r.strip()]
    R = np.array([[float(x) for x in r.split()] for r in rows], dtype=np.float64)
    assert R.shape == (3, 3), f"R parse failed: shape {R.shape}"
    t = np.array([float(x) for x in tr_m.group(1).split()], dtype=np.float64)
    return scale, R, t


def _align_est(est_xyz: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply Sim3 alignment: aligned = scale * R @ est + t."""
    return (scale * (R @ est_xyz.T)).T + t


def _nearest_idx(ts_target: np.ndarray, ts_ref: np.ndarray) -> np.ndarray:
    """For each ts in ts_target, find index in ts_ref with nearest timestamp."""
    out = np.empty(len(ts_target), dtype=np.int64)
    j = 0
    ref_sorted_idx = np.argsort(ts_ref)
    ref_sorted = ts_ref[ref_sorted_idx]
    for i, t in enumerate(ts_target):
        k = np.searchsorted(ref_sorted, t)
        # check k-1 and k for closer
        cands = []
        if k > 0:
            cands.append(k - 1)
        if k < len(ref_sorted):
            cands.append(k)
        best = min(cands, key=lambda c: abs(ref_sorted[c] - t))
        out[i] = ref_sorted_idx[best]
    return out


# ─── Pose / projection utilities ────────────────────────────────────────────

def _qtvec_to_T(qtvec: np.ndarray) -> np.ndarray:
    """(7,) tx ty tz qx qy qz qw -> (4, 4) c2w transform (TUM convention)."""
    tx, ty, tz, qx, qy, qz, qw = qtvec
    # Quaternion -> rotation matrix
    xx, yy, zz, ww = qx*qx, qy*qy, qz*qz, qw*qw
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    xw, yw, zw = qx*qw, qy*qw, qz*qw
    R = np.array([
        [1 - 2*(yy + zz),   2*(xy - zw),     2*(xz + yw)],
        [2*(xy + zw),       1 - 2*(xx + zz), 2*(yz - xw)],
        [2*(xz - yw),       2*(yz + xw),     1 - 2*(xx + yy)],
    ], dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = (tx, ty, tz)
    return T


def _project(disps_ba_i, K_ba, T_ij, H_ba, W_ba):
    """Project pixel grid of KF i to KF j via disp_i and T_ij = T_j @ inv(T_i).
    T_ij is the camera-i to camera-j transform (T_w2c convention)."""
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


# ─── PCA projection (optional, matches D.2) ──────────────────────────────────

def _pca_project(feats: torch.Tensor, mean: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
    """(N, D_raw, h, w) -> (N, K, h, w) via (X - mean) @ components.T."""
    N, D, h, w = feats.shape
    flat = feats.permute(0, 2, 3, 1).reshape(-1, D).float()
    out = (flat - mean) @ components.T
    K = out.shape[1]
    return out.reshape(N, h, w, K).permute(0, 3, 1, 2).contiguous()


# ─── Spearman correlation (scipy if available, hand-rolled otherwise) ────────

def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float | None]:
    """Returns (rho, p-value or None)."""
    try:
        from scipy import stats
        r = stats.spearmanr(x, y)
        return float(r.correlation), float(r.pvalue)
    except Exception:
        # Hand-rolled rho without p-value.
        def _rank(a):
            order = np.argsort(a)
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(1, len(a) + 1)
            return ranks
        rx = _rank(x); ry = _rank(y)
        n = len(x)
        d = rx - ry
        rho = 1.0 - 6.0 * np.sum(d * d) / (n * (n*n - 1))
        return float(rho), None


# ─── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, type=Path,
                   help="Scene output dir (must contain video.npz + gt_poses.txt + traj/...)")
    p.add_argument("--features", default=None, type=Path,
                   help="Default: <scene>/radseg_features.npz")
    p.add_argument("--pca-basis", default=None, type=Path,
                   help="Optional .pt with {mean, components}; mirrors D.2 PCA-256 paradigm")
    p.add_argument("--window", default=5, type=int,
                   help="Edge graph: connect KF i to KF j for |i-j| in [1, window]")
    p.add_argument("--ba-stride", default=8, type=int)
    p.add_argument("--out", required=True, type=Path,
                   help=".npz dump of per-edge arrays")
    p.add_argument("--gate-rho", default=0.2, type=float)
    p.add_argument("--features-key", default="lang_aligned_feats",
                   choices=["lang_aligned_feats", "encoder_feats"],
                   help="Which feature head to audit. 'encoder_feats' per "
                        "RADIO-ViPE Sec III-B preserves geometric content.")
    args = p.parse_args()

    scene = args.scene
    feats_path = args.features or (scene / "radseg_features.npz")
    metrics_path = scene / "traj" / "metrics_full_traj.txt"
    est_path = scene / "traj" / "est_poses_full.txt"
    gt_path = scene / "gt_poses.txt"

    for needed in (feats_path, metrics_path, est_path, gt_path, scene / "video.npz"):
        if not needed.exists():
            print(f"[ERR] missing {needed}", flush=True)
            return 2

    # ── 1. Sim3 alignment from metrics file
    scale, R_align, t_align = _parse_metrics_alignment(metrics_path)
    print(f"[step3a] alignment: scale={scale:.6f}, t={t_align}", flush=True)

    # ── 2. Per-frame est + gt trajectories (timestamps)
    # est_poses_full has frame-index column instead of timestamp
    est_arr = np.loadtxt(est_path, dtype=np.float64)
    est_frame_idx = est_arr[:, 0].astype(np.int64)
    est_xyz = est_arr[:, 1:4]                                          # (N_frames, 3)
    est_xyz_aligned = _align_est(est_xyz, scale, R_align, t_align)

    gt_ts, gt_q = _read_tum_poses(gt_path)
    gt_xyz = gt_q[:, :3]

    # ── 3. KF timestamps from video.npz
    video = np.load(scene / "video.npz", allow_pickle=False)
    kf_timestamps = video["timestamps"]                                # float32 (N_kf,)
    poses_qt = video["tum_poses"]                                      # (N_kf, 7) c2w TUM
    droid_disps = video["droid_disps"]                                 # (N_kf, h_ba, w_ba)
    intrinsics_full = video["intrinsics"]                              # (N_kf, 4)

    N_kf = len(kf_timestamps)
    H_ba, W_ba = droid_disps.shape[1:]
    H_img = video["images"].shape[2]
    W_img = video["images"].shape[3]
    print(f"[step3a] N_kf={N_kf}, BA=({H_ba},{W_ba}), image=({H_img},{W_img})", flush=True)

    # ── 4. Map KF -> GT/est frame index.
    # video.npz['timestamps'] stores integer frame indices (fp32, max ~852 for
    # walking_xyz), NOT Unix timestamps. Index est/GT arrays directly by frame.
    kf_to_gt = kf_timestamps.astype(np.int64)
    assert kf_to_gt.min() >= 0 and kf_to_gt.max() < min(len(est_xyz_aligned), len(gt_xyz)), (
        f"KF frame index out of bounds: range [{kf_to_gt.min()}, {kf_to_gt.max()}] "
        f"vs est={len(est_xyz_aligned)}, gt={len(gt_xyz)}"
    )
    est_xyz_kf = est_xyz_aligned[kf_to_gt]                             # (N_kf, 3)
    gt_xyz_kf = gt_xyz[kf_to_gt]                                       # (N_kf, 3)
    err_per_kf_m = np.linalg.norm(est_xyz_kf - gt_xyz_kf, axis=1)      # (N_kf,) in meters
    err_per_kf_mm = err_per_kf_m * 1000.0
    print(f"[step3a] per-KF Sim3-aligned position error: median={np.median(err_per_kf_mm):.2f} mm, "
          f"max={err_per_kf_mm.max():.2f} mm", flush=True)

    # ── 4b. Per-KF est + gt 7-vec for per-edge RPE proxy ──
    # Plan-v2 review (user): per-edge Relative Pose Error is a stronger,
    # more LOCAL proxy than the per-KF Sim3-aligned absolute error.
    # For edge (i,j): RPE_ij = || trans(T_ij_est) - trans(T_ij_gt) || where
    # T_ij = T_j @ inv(T_i) is the i->j camera transform (independent of
    # any global SLAM-to-GT alignment except scale).
    est_arr_q = est_arr[:, 1:8]                                        # (N_frames, 7) c2w TUM
    est_q_kf  = est_arr_q[kf_to_gt]                                    # (N_kf, 7)
    gt_q_kf   = gt_q[kf_to_gt]                                         # (N_kf, 7) tx ty tz qx qy qz qw
    est_T_kf = np.stack([_qtvec_to_T(est_q_kf[i]) for i in range(N_kf)], axis=0)
    gt_T_kf  = np.stack([_qtvec_to_T(gt_q_kf[i])  for i in range(N_kf)], axis=0)

    # ── 5. Load radseg features, resample to BA grid, optional PCA, L2-norm
    feats_npz = np.load(feats_path, allow_pickle=False)
    features_key = getattr(args, "features_key", "lang_aligned_feats")
    if features_key not in feats_npz.files:
        print(f"[ERR] {feats_path} lacks '{features_key}'. Available: "
              f"{list(feats_npz.files)}", flush=True)
        return 2
    feats_native = torch.from_numpy(feats_npz[features_key]).float()    # (N_p, D, h_n, w_n)
    print(f"[step3a] features_key={features_key}", flush=True)
    N_p, D_raw, h_n, w_n = feats_native.shape
    print(f"[step3a] radseg: N_precomp={N_p}, D_raw={D_raw}, native=({h_n},{w_n})", flush=True)

    feats_ba = F.interpolate(feats_native, size=(H_ba, W_ba), mode="bilinear",
                             align_corners=False)                       # (N_p, D_raw, h_ba, w_ba)
    if args.pca_basis is not None:
        state = torch.load(str(args.pca_basis), map_location="cpu", weights_only=False)
        mean_t = state["mean"].float()
        comps_t = state["components"].float()
        print(f"[step3a] PCA: D_raw={D_raw} -> K={comps_t.shape[0]}, "
              f"var_explained={state.get('fit_variance_explained', 'n/a')}",
              flush=True)
        feats_ba = _pca_project(feats_ba, mean_t, comps_t)
    feats_ba = feats_ba / (feats_ba.norm(dim=1, keepdim=True) + 1e-8)

    # Map KF (DROID-W) -> precompute row using kf_global_indices when
    # available (schema v2, Plan-v2 §Step 3a Option A). Falls back to KF
    # position only if v1 (legacy identity) for backward compat.
    n_precomp = feats_ba.shape[0]
    schema = int(feats_npz.get("schema_version", np.int64(1)))
    if "kf_global_indices" in feats_npz.files and schema >= 2:
        kf_global = feats_npz["kf_global_indices"].astype(np.int64)
        frame_to_row = {int(f): r for r, f in enumerate(kf_global.tolist())}
        feat_row_per_kf = np.array(
            [frame_to_row.get(int(kf_to_gt[i]), -1) for i in range(N_kf)],
            dtype=np.int64,
        )
        mode = f"kf_global_indices (schema v{schema}) -- proper frame-idx lookup"
    else:
        # Legacy v1 fallback: position-based mapping (audit-only workaround;
        # only correct when precompute + audit share the same video.npz).
        feat_row_per_kf = np.arange(N_kf, dtype=np.int64)
        feat_row_per_kf = np.where(feat_row_per_kf < n_precomp, feat_row_per_kf, -1)
        mode = "position fallback (v1 npz)"
    n_missing = int((feat_row_per_kf < 0).sum())
    print(f"[step3a] feature coverage: {N_kf - n_missing}/{N_kf} KFs ({mode})",
          flush=True)

    # ── 6. Build edge graph and compute per-edge stats
    scale_x = W_ba / W_img
    scale_y = H_ba / H_img
    edges = []
    e_emb_list = []
    e_ate_list = []
    e_valid_frac_list = []
    e_rpe_list = []
    poses_T = [_qtvec_to_T(poses_qt[k]) for k in range(N_kf)]
    # DROID-W stores tum_poses in c2w (TUM convention); to get T_w2c for projection use inv.
    poses_w2c = [np.linalg.inv(T) for T in poses_T]

    for i in range(N_kf):
        if feat_row_per_kf[i] < 0:
            continue
        for delta in range(1, args.window + 1):
            j = i + delta
            if j >= N_kf or feat_row_per_kf[j] < 0:
                continue

            T_i = torch.from_numpy(poses_w2c[i]).float()
            T_j = torch.from_numpy(poses_w2c[j]).float()
            T_ij = T_j @ torch.linalg.inv(T_i)

            K_ba = torch.from_numpy(intrinsics_full[i]).float().clone()
            K_ba[0] *= scale_x; K_ba[1] *= scale_y
            K_ba[2] *= scale_x; K_ba[3] *= scale_y

            disp_i = torch.from_numpy(droid_disps[i]).float()
            mu_u, mu_v, valid = _project(disp_i, K_ba, T_ij, H_ba, W_ba)

            mu_x = 2.0 * mu_u / max(W_ba - 1, 1) - 1.0
            mu_y = 2.0 * mu_v / max(H_ba - 1, 1) - 1.0
            grid = torch.stack([mu_x, mu_y], dim=-1).unsqueeze(0)
            F_j_at = F.grid_sample(feats_ba[feat_row_per_kf[j]].unsqueeze(0), grid,
                                   mode="bilinear", padding_mode="border",
                                   align_corners=False).squeeze(0)
            F_j_at = F_j_at / (F_j_at.norm(dim=0, keepdim=True) + 1e-8)
            cs = (feats_ba[feat_row_per_kf[i]] * F_j_at).sum(dim=0)    # (h, w)
            r_sem = (1.0 - cs).clamp(min=0)
            r_sem2 = r_sem * r_sem

            valid_f = valid.float()
            if valid_f.sum() < 16:
                continue
            mean_r2 = float((r_sem2 * valid_f).sum() / valid_f.sum())
            ate_proxy = float(max(err_per_kf_mm[i], err_per_kf_mm[j]))

            # Per-edge RPE proxy (Plan-v2 review): T_ij relative transform
            # consistency between est and gt. Scale the est by the global
            # Sim3 scale factor so translations are commensurate.
            T_ij_est = np.linalg.inv(est_T_kf[j]) @ est_T_kf[i]         # cam_i -> cam_j (est)
            T_ij_gt  = np.linalg.inv(gt_T_kf[j])  @ gt_T_kf[i]          # cam_i -> cam_j (gt)
            t_est = T_ij_est[:3, 3] * scale                            # apply Sim3 scale
            t_gt  = T_ij_gt[:3, 3]
            rpe_trans_mm = float(np.linalg.norm(t_est - t_gt) * 1000.0)

            edges.append((i, j))
            e_emb_list.append(mean_r2)
            e_ate_list.append(ate_proxy)
            e_rpe_list.append(rpe_trans_mm)
            e_valid_frac_list.append(float(valid_f.mean()))

    e_emb = np.asarray(e_emb_list, dtype=np.float64)
    e_ate = np.asarray(e_ate_list, dtype=np.float64)
    e_valid = np.asarray(e_valid_frac_list, dtype=np.float64)
    e_rpe = np.asarray(e_rpe_list, dtype=np.float64) if e_rpe_list else None

    if len(e_emb) < 10:
        print(f"[ERR] only {len(e_emb)} valid edges -- not enough", flush=True)
        return 2

    rho, p = _spearman(e_emb, e_ate)
    pearson = float(np.corrcoef(e_emb, e_ate)[0, 1])

    print(f"\n[step3a] n_edges={len(e_emb)}, mean_valid_frac={e_valid.mean():.3f}",
          flush=True)
    print(f"[step3a] mean r_embed^2: median={np.median(e_emb):.5f}  IQR=({np.percentile(e_emb,25):.5f},{np.percentile(e_emb,75):.5f})",
          flush=True)
    print(f"[step3a] ATE proxy (Sim3-abs) [mm]: median={np.median(e_ate):.2f}  IQR=({np.percentile(e_ate,25):.2f},{np.percentile(e_ate,75):.2f})",
          flush=True)
    print(f"[step3a] Spearman rho vs ATE-abs ={rho:+.4f}  (p={p:.2e})" if p is not None
          else f"[step3a] Spearman rho vs ATE-abs ={rho:+.4f}", flush=True)
    print(f"[step3a] Pearson r vs ATE-abs    ={pearson:+.4f}", flush=True)

    # Per-edge RPE proxy (Plan-v2 review): more local than the absolute
    # Sim3-aligned ATE -- compares the relative transform i->j in est vs gt.
    if e_rpe is not None and len(e_rpe) == len(e_emb):
        rho_rpe, p_rpe = _spearman(e_emb, e_rpe)
        pearson_rpe = float(np.corrcoef(e_emb, e_rpe)[0, 1])
        print(f"[step3a] RPE proxy (per-edge trans) [mm]: median={np.median(e_rpe):.2f}  "
              f"IQR=({np.percentile(e_rpe,25):.2f},{np.percentile(e_rpe,75):.2f})",
              flush=True)
        print(f"[step3a] Spearman rho vs RPE     ={rho_rpe:+.4f}  (p={p_rpe:.2e})",
              flush=True)
        print(f"[step3a] Pearson r vs RPE        ={pearson_rpe:+.4f}", flush=True)
    else:
        rho_rpe = float("nan"); pearson_rpe = float("nan")
    print(f"[step3a] gate: rho >= {args.gate_rho} (against either proxy)", flush=True)
    passed = rho >= args.gate_rho or (e_rpe is not None and rho_rpe >= args.gate_rho)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out,
             edges=np.asarray(edges),
             mean_r_embed2=e_emb,
             ate_proxy_mm=e_ate,
             rpe_proxy_mm=e_rpe if e_rpe is not None else np.zeros(0),
             valid_frac=e_valid,
             spearman_rho=rho,
             spearman_p=p if p is not None else float("nan"),
             spearman_rho_rpe=rho_rpe,
             pearson_r=pearson,
             pearson_r_rpe=pearson_rpe,
             window=args.window,
             gate_rho=args.gate_rho,
             passed=passed)
    print(f"[step3a] dumped per-edge arrays -> {args.out}", flush=True)

    if passed:
        print(f"[step3a] PASS  ({rho:.4f} >= {args.gate_rho}) -- proceed to D.1b",
              flush=True)
        return 0
    print(f"[WARN] step3a: expected Spearman rho >= {args.gate_rho}, got {rho:.4f}, "
          f"fallback=STOP. D.1b will not help; revisit feature choice or gating "
          f"(plan-v2 §Step 3a STOP gate).", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
