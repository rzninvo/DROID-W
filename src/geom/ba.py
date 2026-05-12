# Copyright 2024 The GlORIE-SLAM Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
 
import lietorch
import torch
import torch.nn.functional as F
from .chol import block_solve, schur_solve
import src.geom.projective_ops as pops


# ──────────────────────────────────────────────────────────────────────────
# Plan-v2 §Step 3b — Goal D.1b gated semantic-prior residual.
#
# Per RADIO-ViPE Sec III-B Eqs.6-7:
#   r_embed(u) = lambda * sqrt(2 * (1 - cs_ij(u)))      [per-pixel residual]
#   E_embed    = sum_u  w_combined(u) * rho(r_embed(u))  [per-edge energy]
# where cs_ij(u) = <F_i(u), F_j(mu_ij(u))> on L2-normalised features (PCA-256
# of RADIO encoder output, NOT lang-aligned -- Report 23 §1.1).
#
# Plan-v2 §Step 3b applies a gated formulation:
#   w_combined = g_valid * g_unc * g_grad * w_photo * rho_huber'
# to limit influence to (a) projection-valid pixels, (b) low UDBA uncertainty,
# (c) regions with non-vanishing feature gradient (otherwise Hessian is rank-
# deficient on that pixel), (d) Goal C's static-region mask if available.
#
# Gauss-Newton: ALL Jacobians via chain rule on the existing projection
# Jacobians from `projective_ops.projective_transform()`:
#   dr/dT_i = (dr/dmu) @ Ji_proj      # (B,N,ht,wd,1,6)
#   dr/dT_j = (dr/dmu) @ Jj_proj
#   dr/dd_i = (dr/dmu) @ Jz_proj
# with dr/dmu = -lambda^2 / r * (Fi - cs * Fj_at_mu) @ dFj/dmu  /  |Fj_at_mu|
# (the projection factor (I - Fj_n Fj_n^T) is rewritten in terms of cs).
#
# The bilinear-interpolation spatial gradient dFj/dmu is computed analytically
# from finite differences on the integer-grid features (no autograd) then
# bilinearly resampled at mu (sub-pixel accurate to first order).
# ──────────────────────────────────────────────────────────────────────────


def _embed_residual_block(
    F_i, F_j_grid, coords, valid,
    Ji_proj, Jj_proj, Jz_proj,
    w_photo,
    *,
    lam_embed=2.0,
    huber_delta=0.5,
    gate_grad_min=0.01,
    eps_cs=1e-4,
    uncertainties_i=None,
    gate_unc_tau=None,
    goal_c_mask=None,
):
    """Build the D.1b embedding residual + Jacobian rows + combined weight.

    Args:
        F_i, F_j_grid: (B, N, K, ht, wd) PCA-256 L2-normalised features
            (RADIO encoder + frozen PCA per RADIO-ViPE Sec III-B).
        coords: (B, N, ht, wd, 2) projected pixel positions (mu_ij).
        valid: (B, N, ht, wd, 1) bool/float — projection in-bounds + min-depth.
        Ji_proj, Jj_proj: (B, N, ht, wd, 2, 6) projection Jacobians from
            `projective_transform(..., jacobian=True)`.
        Jz_proj: (B, N, ht, wd, 2, 1) depth Jacobian.
        w_photo: (B, N, ht, wd, 1) photometric per-pixel confidence
            (from the GRU update operator; reused per RADIO-ViPE Eq.7).
        lam_embed: residual amplitude (paper default 2.0).
        huber_delta: Huber threshold (plan-v2 §Step 3b default 0.5).
        gate_grad_min: minimum feature spatial-gradient norm to keep an edge.
        eps_cs: clip to avoid sqrt singularity at cs→1.
        uncertainties_i: optional (B, N, ht, wd) UDBA uncertainty for g_unc.
        gate_unc_tau: optional sigmoid threshold for g_unc. None → g_unc=1.
        goal_c_mask: optional (B, N, ht, wd) ∈ [0,1] from Goal C dynamic mask.

    Returns:
        r_embed_flat: (B, N, ht*wd, 1)
        Ji_embed_flat: (B, N, ht*wd, 6)
        Jj_embed_flat: (B, N, ht*wd, 6)
        Jz_embed_flat: (B, N, ht*wd, 1)
        w_combined_flat: (B, N, ht*wd, 1)   — all gates folded in
    """
    B, N, K, ht, wd = F_i.shape

    # ─── 1. Bilinear-sample F_j at mu_ij ───
    mu_u = coords[..., 0]
    mu_v = coords[..., 1]
    grid_x = 2.0 * mu_u / max(wd - 1, 1) - 1.0
    grid_y = 2.0 * mu_v / max(ht - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(B * N, ht, wd, 2)
    Fj_flat = F_j_grid.reshape(B * N, K, ht, wd)
    F_j_at = F.grid_sample(
        Fj_flat, grid, mode="bilinear",
        padding_mode="border", align_corners=False,
    ).view(B, N, K, ht, wd)

    # ─── 2. Cosine similarity ───
    Fi_n = F_i / (F_i.norm(dim=2, keepdim=True) + 1e-8)
    Fj_n = F_j_at / (F_j_at.norm(dim=2, keepdim=True) + 1e-8)
    cs = (Fi_n * Fj_n).sum(dim=2).clamp(min=-1.0, max=1.0 - eps_cs)  # (B,N,ht,wd)

    one_m_cs = (1.0 - cs).clamp(min=eps_cs)
    r_embed = lam_embed * torch.sqrt(2.0 * one_m_cs)                # (B,N,ht,wd)

    # ─── 3. Spatial gradient of F_j on integer grid (centred differences) ───
    # Pad with zero at boundaries — safe because g_valid will zero those pixels.
    Fj_dx = torch.zeros_like(F_j_grid)
    Fj_dx[..., :, 1:-1] = 0.5 * (F_j_grid[..., :, 2:] - F_j_grid[..., :, :-2])
    Fj_dy = torch.zeros_like(F_j_grid)
    Fj_dy[..., 1:-1, :] = 0.5 * (F_j_grid[..., 2:, :] - F_j_grid[..., :-2, :])

    Fj_dx_flat = Fj_dx.reshape(B * N, K, ht, wd)
    Fj_dy_flat = Fj_dy.reshape(B * N, K, ht, wd)
    dFj_du = F.grid_sample(
        Fj_dx_flat, grid, mode="bilinear",
        padding_mode="border", align_corners=False,
    ).view(B, N, K, ht, wd)
    dFj_dv = F.grid_sample(
        Fj_dy_flat, grid, mode="bilinear",
        padding_mode="border", align_corners=False,
    ).view(B, N, K, ht, wd)

    # ─── 4. dcs/dmu  via  d cos / d Fj_at  =  (I - Fj_n Fj_n^T) / |Fj_at| ───
    # dcs/dmu_u = Fi_n^T (I - Fj_n Fj_n^T) dFj/dmu_u / |Fj_at|
    #           = (Fi_n - cs Fj_n)^T dFj/dmu_u / |Fj_at|
    Fj_at_mag = (F_j_at.norm(dim=2, keepdim=True) + 1e-8)            # (B,N,1,ht,wd)
    proj_factor = Fi_n - cs.unsqueeze(2) * Fj_n                      # (B,N,K,ht,wd)
    dcs_du = (proj_factor * dFj_du).sum(dim=2) / Fj_at_mag.squeeze(2)  # (B,N,ht,wd)
    dcs_dv = (proj_factor * dFj_dv).sum(dim=2) / Fj_at_mag.squeeze(2)

    # ─── 5. dr/dcs = -lambda^2 / r  (from r = lambda*sqrt(2(1-cs))) ───
    dr_dcs = -lam_embed * lam_embed / r_embed.clamp(min=eps_cs)
    dr_du = dr_dcs * dcs_du
    dr_dv = dr_dcs * dcs_dv
    dr_dmu = torch.stack([dr_du, dr_dv], dim=-1).unsqueeze(-2)        # (B,N,ht,wd,1,2)

    # ─── 6. Chain to pose / depth via existing projection Jacobians ───
    Ji_embed = torch.matmul(dr_dmu, Ji_proj).squeeze(-2)              # (B,N,ht,wd,6)
    Jj_embed = torch.matmul(dr_dmu, Jj_proj).squeeze(-2)
    Jz_embed = torch.matmul(dr_dmu, Jz_proj).squeeze(-2).squeeze(-1)  # (B,N,ht,wd)

    # ─── 7. Gates ───
    valid_f = valid.squeeze(-1).float() if valid.dim() > 4 else valid.float()  # (B,N,ht,wd)
    g_grad = ((dFj_du.norm(dim=2) + dFj_dv.norm(dim=2)) > gate_grad_min).float()
    g = valid_f * g_grad
    if goal_c_mask is not None:
        g = g * goal_c_mask
    if uncertainties_i is not None and gate_unc_tau is not None:
        g_unc = torch.sigmoid(gate_unc_tau - uncertainties_i)
        g = g * g_unc

    # ─── 8. Robust kernel (Huber on r_embed) — IRLS form ───
    abs_r = r_embed.abs() + 1e-8
    w_rho = torch.where(
        abs_r <= huber_delta,
        torch.ones_like(r_embed),
        huber_delta / abs_r,
    )

    # ─── 9. Combined per-pixel weight: gate * photo-confidence * Huber-IRLS ───
    w_photo_pix = w_photo.squeeze(-1) if w_photo.dim() > 4 else w_photo  # (B,N,ht,wd)
    w_combined = g * w_photo_pix * w_rho

    # Flatten to (B, N, ht*wd, *)
    r_embed_flat = r_embed.view(B, N, ht * wd, 1)
    Ji_embed_flat = Ji_embed.view(B, N, ht * wd, 6)
    Jj_embed_flat = Jj_embed.view(B, N, ht * wd, 6)
    Jz_embed_flat = Jz_embed.view(B, N, ht * wd, 1)
    w_combined_flat = w_combined.view(B, N, ht * wd, 1)
    return (r_embed_flat, Ji_embed_flat, Jj_embed_flat, Jz_embed_flat,
            w_combined_flat)



def _scatter_sum(src, index, dim=0, dim_size=None):
    """Native PyTorch replacement for torch_scatter.scatter_sum.

    Fixes a latent bug in the old `index.unsqueeze(-1).expand_as(src)` form,
    which only happened to broadcast for src.dim() <= 3 with the right shape
    alignment. For src shapes like (B, N, D, D) (used by `safe_scatter_add_mat`
    on H-blocks) the old expand_as silently failed when N != D. This path was
    never hit in live SLAM because `DepthVideo.ba()` routes to the CUDA
    kernel; the bug only surfaces when the Python `BA()` is exercised (e.g.
    Plan-v2 §Step 3b D.1b synthetic 2-frame test).
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1
    shape = list(src.shape)
    shape[dim] = dim_size
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    # Build idx with same shape as src, where idx[..., dim=k, ...] == index[k].
    idx_shape = [1] * src.dim()
    idx_shape[dim] = index.shape[0]
    idx = index.view(idx_shape).expand_as(src)
    return out.scatter_add_(dim, idx, src)


# utility functions for scattering ops
def safe_scatter_add_mat(A, ii, jj, n, m):
    v = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < m)
    return _scatter_sum(A[:,v], ii[v]*m + jj[v], dim=1, dim_size=n*m)

def safe_scatter_add_vec(b, ii, n):
    v = (ii >= 0) & (ii < n)
    return _scatter_sum(b[:,v], ii[v], dim=1, dim_size=n)

# apply retraction operator to inv-depth maps
def disp_retr(disps, dz, ii):
    ii = ii.to(device=dz.device)
    return disps + _scatter_sum(dz, ii, dim=1, dim_size=disps.shape[1])

def wq_retr(wqs, dwq, ii):
    ii = ii.to(device=dwq.device)
    return wqs + _scatter_sum(dwq, ii, dim=1, dim_size=wqs.shape[1])

# apply retraction operator to poses
def pose_retr(poses, dx, ii):
    ii = ii.to(device=dx.device)
    return poses.retr(_scatter_sum(dx, ii, dim=1, dim_size=poses.shape[1]))

@torch.no_grad()
def BA(target, weight, eta, poses, disps, intrinsics, ii, jj,
       sensor_disps=None, lm=0.0001, ep=0.1, alpha=0.05, fixedp=1, rig=1,
       embed_features=None, gamma_embed=0.0, lam_embed=2.0,
       huber_delta=0.5, gate_grad_min=0.01, eps_cs=1e-4,
       uncertainties=None, gate_unc_tau=None, goal_c_mask=None):
    """ Full Bundle Adjustment.

    Optional Plan-v2 §Step 3b kwargs (when `gamma_embed > 0` AND
    `embed_features is not None`):
      - `embed_features`: (P, K, ht, wd) L2-normalised PCA-256 RADIO encoder
        features per KF buffer slot (matches `DepthVideo.radseg_feats_resize`).
      - `gamma_embed`: scalar amplitude on the residual (squared into the
        Hessian per Gauss-Newton). `0.0` => bit-identical bypass.
      - `lam_embed`, `huber_delta`, `gate_grad_min`, `eps_cs`: see
        `_embed_residual_block`.
      - `uncertainties`, `gate_unc_tau`: optional per-KF UDBA uncertainties +
        sigmoid threshold for `g_unc`.
      - `goal_c_mask`: optional (P, ht, wd) static-region mask from Goal C.
    """

    B, P, ht, wd = disps.shape
    N = ii.shape[0]
    D = poses.manifold_dim
    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
        poses, disps, intrinsics, ii, jj, jacobian=True)

    # Save un-flattened Jacobians for the optional D.1b embed block (it
    # chains through them via dr/dmu @ J_proj; needs (B,N,ht,wd,2,*) shapes).
    Ji_proj = Ji
    Jj_proj = Jj
    Jz_proj = Jz

    r = (target - coords).view(B, N, -1, 1) #[B,N,2*ht*wd,D]
    w = .001 * (valid * weight).view(B, N, -1, 1) #[B,N,2*ht*wd,D]


    ### 2: construct linear system ###
    Ji = Ji.reshape(B, N, -1, D)        #[B,N,2*ht*wd,D]
    Jj = Jj.reshape(B, N, -1, D)        #[B,N,2*ht*wd,D]
    wJiT = (w * Ji).transpose(2,3).contiguous()  #[B,N,D,2*ht*wd]
    wJjT = (w * Jj).transpose(2,3).contiguous()  #[B,N,D,2*ht*wd]

    Jz = Jz.reshape(B, N, ht*wd, -1)    #[B,N,ht*wd,2]

    Hii = torch.matmul(wJiT, Ji)        #[B,N,D,D]
    Hij = torch.matmul(wJiT, Jj)        #[B,N,D,D]
    Hji = torch.matmul(wJjT, Ji)        #[B,N,D,D]
    Hjj = torch.matmul(wJjT, Jj)        #[B,N,D,D]

    vi = torch.matmul(wJiT, r).squeeze(-1) #[B,N,D]
    vj = torch.matmul(wJjT, r).squeeze(-1) #[B,N,D]

    Ei = (wJiT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1) #[B,N,D,ht*wd]
    Ej = (wJjT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1) #[B,N,D,ht*wd]

    w = w.view(B, N, ht*wd, -1) #[B,N,ht*wd,2]
    r = r.view(B, N, ht*wd, -1) #[B,N,ht*wd,2]
    wk = torch.sum(w*r*Jz, dim=-1) #[B,N,ht*wd]
    Ck = torch.sum(w*Jz*Jz, dim=-1) #[B,N,ht*wd]

    # ── Plan-v2 §Step 3b — Goal D.1b gated semantic-prior residual ──
    # True branch bypass: when gamma_embed <= 0 or features absent, this entire
    # block is skipped, leaving the photometric BA bit-identical to vanilla.
    if gamma_embed > 0.0 and embed_features is not None:
        gamma2 = float(gamma_embed) * float(gamma_embed)
        # Gather per-edge features. embed_features is (P_buf, K, ht, wd).
        # ii, jj are (N,) edge indices.
        # The first batch dim B is typically 1 for DROID-W; expand.
        F_i_full = embed_features[ii].unsqueeze(0).expand(B, -1, -1, -1, -1).float()  # (B,N,K,ht,wd)
        F_j_full = embed_features[jj].unsqueeze(0).expand(B, -1, -1, -1, -1).float()

        # Per-pixel photo weight (1-channel). w view above is (B,N,ht*wd,2);
        # take the mean across the (u,v) channels and reshape to (B,N,ht,wd,1).
        w_photo_pix = w.mean(dim=-1).view(B, N, ht, wd, 1)

        unc_per_edge = None
        if uncertainties is not None:
            unc_per_edge = uncertainties[ii].unsqueeze(0).expand(B, -1, -1, -1)  # (B,N,ht,wd)
        goal_c_per_edge = None
        if goal_c_mask is not None:
            goal_c_per_edge = goal_c_mask[ii].unsqueeze(0).expand(B, -1, -1, -1)  # (B,N,ht,wd)

        r_em, Ji_em, Jj_em, Jz_em, w_em = _embed_residual_block(
            F_i_full, F_j_full, coords, valid,
            Ji_proj, Jj_proj, Jz_proj,
            w_photo_pix,
            lam_embed=lam_embed,
            huber_delta=huber_delta,
            gate_grad_min=gate_grad_min,
            eps_cs=eps_cs,
            uncertainties_i=unc_per_edge,
            gate_unc_tau=gate_unc_tau,
            goal_c_mask=goal_c_per_edge,
        )
        # Scale residual energy by gamma^2 by folding into the weight (so the
        # Hessian and gradient both pick up the factor consistently).
        w_em = w_em * gamma2

        wJiT_em = (w_em * Ji_em).transpose(2, 3).contiguous()        #[B,N,D,ht*wd]
        wJjT_em = (w_em * Jj_em).transpose(2, 3).contiguous()

        Hii = Hii + torch.matmul(wJiT_em, Ji_em)
        Hij = Hij + torch.matmul(wJiT_em, Jj_em)
        Hji = Hji + torch.matmul(wJjT_em, Ji_em)
        Hjj = Hjj + torch.matmul(wJjT_em, Jj_em)

        # SIGN CONVENTION: the photometric residual is r = (target - coords),
        # so the linear system solves H*dx = +J^T W r (the negative cancels
        # against the (-) in the coords change). The embedding residual is
        # r_em = lambda*sqrt(2(1-cs)) — strictly non-negative — so its
        # linearisation gives new_r_em ≈ r_em + J_em dx, and Gauss-Newton
        # minimisation yields H*dx = -J_em^T W r_em. We therefore SUBTRACT
        # the embed v / wk contributions while ADDING the (positive-definite)
        # H / C / E^T E contributions.
        vi = vi - torch.matmul(wJiT_em, r_em).squeeze(-1)
        vj = vj - torch.matmul(wJjT_em, r_em).squeeze(-1)

        # Embed has 1 residual channel: Jz_em is (B,N,ht*wd,1).
        Ei_em = (wJiT_em.view(B, N, D, ht*wd, 1) * Jz_em[:, :, None]).sum(dim=-1)
        Ej_em = (wJjT_em.view(B, N, D, ht*wd, 1) * Jz_em[:, :, None]).sum(dim=-1)
        Ei = Ei + Ei_em
        Ej = Ej + Ej_em

        wk_em = -(w_em * r_em * Jz_em).sum(dim=-1)                   #[B,N,ht*wd]  (sign per above)
        Ck_em = (w_em * Jz_em * Jz_em).sum(dim=-1)
        wk = wk + wk_em
        Ck = Ck + Ck_em


    kx, kk = torch.unique(ii, return_inverse=True)
    M = kx.shape[0]

    # only optimize keyframe poses
    P = torch.div(P,rig,rounding_mode="trunc")-fixedp
    ii = torch.div(ii,rig,rounding_mode="trunc")-fixedp
    jj = torch.div(jj,rig,rounding_mode="trunc")-fixedp

    H = safe_scatter_add_mat(Hii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hjj, jj, jj, P, P)            #[B,P*P,D,D]

    E = safe_scatter_add_mat(Ei, ii, kk, P, M) + \
        safe_scatter_add_mat(Ej, jj, kk, P, M)             #[B,P*M,D,ht*wd]

    v = safe_scatter_add_vec(vi, ii, P) + \
        safe_scatter_add_vec(vj, jj, P)                    #[B,P,D]

    C = safe_scatter_add_vec(Ck, kk, M)                    #[B,M,ht*wd]

    # C = C + eta.view(*C.shape) #+ 1e-7

    w = safe_scatter_add_vec(wk, kk, M)  #[B,M,ht*wd]
    if sensor_disps is None:
        C = C + eta.view(*C.shape) #+ 1e-7
    else:
        m = (sensor_disps[:,kx]>0).float().view(B,M,ht*wd)     #[B,M,ht*wd]
        C = C + m*alpha + (1-m)*eta.view(*C.shape)             #[B,M,ht*wd]
        w = w - m*alpha*(disps[:,kx]-sensor_disps[:,kx]).view(B,M,ht*wd)  #[B,M,ht*wd]


    H = H.view(B, P, P, D, D)
    E = E.view(B, P, M, D, ht*wd)

    ### 3: solve the system ###
    dx, dz = schur_solve(H, E, C, v, w, ep, lm)
    # dx [B,P,D]
    # dz [B,M,ht*wd]

    ### 4: apply retraction ###
    poses = pose_retr(poses, dx, torch.arange(P) + fixedp)
    disps = disp_retr(disps, dz.view(B,-1,ht,wd), kx)

    # disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
    disps = disps.clamp(min=0.0)

    return poses, disps





@torch.no_grad()
def BA_with_scale_shift(target, weight, eta, poses, disps, intrinsics, ii, jj, 
       mono_disps, scales=None, shifts=None, 
       valid_depth_mask=None, ignore_frames=0,
       lm=0.0001, ep=0.1, alpha=1.0, fixedp=1, rig=1):
    """ optimize disparities (disp), scales (w) and shifts (q) together, eq.17 in the paper,
        math details can be found in the supplementary
    """
    device = ii.device
    B, P, ht, wd = disps.shape
    N = ii.shape[0]
    D = poses.manifold_dim
    kx, kk = torch.unique(ii, return_inverse=True)
    M = kx.shape[0]
    sqrt_alpha = torch.tensor(alpha).sqrt().to(device)
    ll = torch.arange(M,device=device)
    wqs = torch.stack([scales,shifts],dim=2)         #[B,P,2]

    ignore_mask = kx<ignore_frames
    invalid_mask = (mono_disps[:,kx]<1e-6).view(B,M,ht*wd)    #[B,M,ht*wd]
    invalid_mask[:,ignore_mask] = True

    valid_depth_mask = valid_depth_mask[:,kx].view(B,M,ht*wd)
    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
        poses, disps, intrinsics, ii, jj, jacobian=True)

    r = (target - coords).view(B, N, -1, 1)           #[B,N,ht*wq*2,1]
    r_depth = sqrt_alpha * (disps[:,kx]-(scales[:,kx,None,None]*mono_disps[:,kx]+shifts[:,kx,None,None])).view(B,M,ht*wd,1)

    w = .001 * (valid * weight).view(B, N, -1, 1)

    sqrt_alpha = torch.ones(B,M,ht*wd,1).float().to(device) * sqrt_alpha
    sqrt_alpha[valid_depth_mask] *= 10

    J_d = torch.ones(B,M,ht*wd,1).float().to(device) * sqrt_alpha
    J_scale = -mono_disps[:,kx].clone().view(B,M,ht*wd,1) * sqrt_alpha#[B,M,ht*wd,1]
    J_shift = -torch.ones(B,M,ht*wd,1).float().to(device) * sqrt_alpha#[B,M,ht*wd,1]

    J_d[invalid_mask*valid_depth_mask] = 0
    J_scale[invalid_mask] = 0
    J_shift[invalid_mask] = 0

    J_wq = torch.cat([J_scale,J_shift],dim=3)         #[B,M,ht*wd,2]
    J_wq_T = J_wq.transpose(2,3).contiguous()  #[B,M,2,ht*wd]
    H_wq = torch.matmul(J_wq_T, J_wq)   #[B,M,2,2]
    u = - torch.matmul(J_wq_T, r_depth).squeeze(-1) #[B,M,2]
    ### 2: construct linear system ###

    Jz = Jz.reshape(B, N, ht*wd, -1)    #[B,N,ht*wd,2] 
    # here Jz does not contain the negative sign in the residual term 

    E_wq_d = (J_wq_T.view(B,M,2,ht*wd,-1) * J_d[:,:,None]).sum(dim=-1) #[B,M,2,ht*wd]

    w = w.view(B, N, ht*wd, -1) #[B,N,ht*wd,2]
    r = r.view(B, N, ht*wd, -1) #[B,N,ht*wd,2]
    wk = torch.sum(-w*r*Jz, dim=-1) #[B,N,ht*wd]
    Ck = torch.sum(w*(-Jz)*(-Jz), dim=-1) #[B,N,ht*wd]

    # only optimize keyframe poses
    P = torch.div(P,rig,rounding_mode="trunc")-fixedp
    ii = torch.div(ii,rig,rounding_mode="trunc")-fixedp
    jj = torch.div(jj,rig,rounding_mode="trunc")-fixedp

    H_wq = safe_scatter_add_mat(H_wq,ll,ll,M,M)       #[B,M*M,2,2]
    E_wq_d = safe_scatter_add_mat(E_wq_d,ll,ll,M,M)      #[B,M*M,2,ht*wd]
    C_proj = safe_scatter_add_vec(Ck, kk, M)                #[B,M,ht*wd]
    u = safe_scatter_add_vec(u, ll, M)                      #[B,M,2]

    # C = C + eta.view(*C.shape) #+ 1e-7
    C_depth = (J_d*J_d).view(B,M,ht*wd)
    # C = C_proj + C_depth + (1-C_depth)*eta.view(*C_proj.shape)             #[B,M,ht*wd]
    C = C_proj + C_depth + eta.view(*C_proj.shape) #+ 1e-7

    w_proj = safe_scatter_add_vec(wk, kk, M)                               #[B,M,ht*wd]
    w = -w_proj - (J_d*r_depth).view(B,M,ht*wd)  #[B,M,ht*wd]
    H = H_wq.view(B, M, M, 2, 2)
    E = E_wq_d.view(B, M, M, 2, ht*wd)
    ### 3: solve the system ###    
    dwq, dz = schur_solve(H, E, C, u, w, ep, lm)
    # dwq [B,M,2]
    # dz [B,M,ht*wd]
    ### 4: apply retraction ###
    # poses = pose_retr(poses, dx, torch.arange(P) + fixedp)
    disps = disp_retr(disps, dz.view(B,-1,ht,wd), kx)
    wqs = wq_retr(wqs,dwq,kx)
    # disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
    disps = disps.clamp(min=0.0)

    return poses, disps , wqs






def MoBA(target, weight, eta, poses, disps, intrinsics, ii, jj, fixedp=1, rig=1):
    """ Motion only bundle adjustment """

    B, P, ht, wd = disps.shape
    N = ii.shape[0]
    D = poses.manifold_dim

    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
        poses, disps, intrinsics, ii, jj, jacobian=True)

    r = (target - coords).view(B, N, -1, 1)
    w = .001 * (valid * weight).view(B, N, -1, 1)

    ### 2: construct linear system ###
    Ji = Ji.reshape(B, N, -1, D)
    Jj = Jj.reshape(B, N, -1, D)
    wJiT = (w * Ji).transpose(2,3).contiguous()
    wJjT = (w * Jj).transpose(2,3).contiguous()

    Hii = torch.matmul(wJiT, Ji)
    Hij = torch.matmul(wJiT, Jj)
    Hji = torch.matmul(wJjT, Ji)
    Hjj = torch.matmul(wJjT, Jj)

    vi = torch.matmul(wJiT, r).squeeze(-1)
    vj = torch.matmul(wJjT, r).squeeze(-1)

    # only optimize keyframe poses
    P = P // rig - fixedp
    ii = ii // rig - fixedp
    jj = jj // rig - fixedp

    H = safe_scatter_add_mat(Hii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hjj, jj, jj, P, P)

    v = safe_scatter_add_vec(vi, ii, P) + \
        safe_scatter_add_vec(vj, jj, P)
    
    H = H.view(B, P, P, D, D)

    ### 3: solve the system ###
    dx = block_solve(H, v)

    ### 4: apply retraction ###
    poses = pose_retr(poses, dx, torch.arange(P) + fixedp)
    return poses

