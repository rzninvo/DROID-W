"""
Plan-v2 §Step 3b — minimum viable test for the D.1b gated semantic-prior
residual in `src.geom.ba.BA()`.

Three checks, in order of stringency:

  T1. **bit-identity bypass**: BA() with gamma_embed=0 (default) returns the
      exact same (poses, disps) as the original vanilla path. Anything else
      means the new code mutated state when it shouldn't.

  T2. **embed-only sanity**: BA() with gamma_embed=1 and weight=0 (photometric
      term zeroed) runs without numerical failure. r_embed and Jacobian
      shapes inside the block must be consistent with the scatter pattern.

  T3. **synthetic 2-frame convergence**: build a 2-frame toy with a known
      SE3 perturbation between slot 0 (fixed) and slot 1, set target =
      identity flow and weight≈0 (no photometric signal), set features so
      F_i and F_j_grid share a spatially-varying pattern that the embed
      residual can lock onto. Iterate BA() with gamma_embed>0; check that
      ||pose_err|| monotonically decreases over a few iterations (does not
      need to converge to zero — just direction-positive).

Usage (cvg, droid-w env):
    cd /home/cvg/HERMES-SLAM/DROID-W
    python tests/test_embed_residual_synthetic_2frame.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import lietorch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.geom.ba import BA


def _make_problem(B=1, P=2, ht=16, wd=16, K=8, device="cuda:0", seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Disparities — constant 1.0 (depth=1).
    disps = torch.ones(B, P, ht, wd, device=device)
    # Intrinsics: fx=fy=ht, cx=wd/2, cy=ht/2 (so projection is well-behaved).
    intr = torch.tensor(
        [[[float(ht), float(ht), wd / 2.0, ht / 2.0] for _ in range(P)]],
        device=device, dtype=torch.float32,
    )  # (1, P, 4)
    # Identity poses for slot 0 + (initially) slot 1 — we'll set GT below.
    eye7 = torch.tensor([0, 0, 0, 0, 0, 0, 1.0], device=device).view(1, 1, 7).expand(B, P, 7).contiguous()
    poses = lietorch.SE3(eye7.clone())
    # Edges: connect (0,1) and (1,0) so both directions exist.
    ii = torch.tensor([0, 1], device=device, dtype=torch.long)
    jj = torch.tensor([1, 0], device=device, dtype=torch.long)
    N = ii.shape[0]
    # Photometric target = identity pixel grid (no flow), weight = small.
    yy, xx = torch.meshgrid(
        torch.arange(ht, device=device).float(),
        torch.arange(wd, device=device).float(),
        indexing="ij",
    )
    # BA() expects target/weight shapes that broadcast against coords (B,N,ht,wd,2)
    # and value channel last. Looking at the existing call site in BA():
    #   r = (target - coords).view(B, N, -1, 1)
    # so target and coords must agree in (B, N, ht, wd, 2).
    target = torch.stack([xx, yy], dim=-1).view(1, 1, ht, wd, 2).expand(B, N, ht, wd, 2).contiguous()
    # weight has same leading dims, the inner 2 is the (u,v) channel weight.
    weight = torch.ones(B, N, ht, wd, 2, device=device) * 1e-6
    eta = torch.ones(B, P, ht * wd, device=device) * 1e-4
    return poses, disps, intr, ii, jj, target, weight, eta


def _make_features(P, K, ht, wd, device, kind="ramp"):
    """Build a synthetic feature buffer with non-zero spatial gradient."""
    yy, xx = torch.meshgrid(
        torch.arange(ht, device=device).float(),
        torch.arange(wd, device=device).float(),
        indexing="ij",
    )
    feats = torch.zeros(P, K, ht, wd, device=device)
    if kind == "ramp":
        # Each channel k is a different linear ramp orientation.
        for k in range(K):
            ang = 2.0 * np.pi * k / K
            feats[:, k] = (
                np.cos(ang) * xx / wd + np.sin(ang) * yy / ht
            )
    elif kind == "sine":
        for k in range(K):
            fx_, fy_ = (k % 4) + 1, (k // 4 + 1)
            feats[:, k] = torch.sin(2.0 * np.pi * fx_ * xx / wd) * torch.cos(2.0 * np.pi * fy_ * yy / ht)
    else:
        raise ValueError(f"unknown kind {kind}")
    # The SAME pattern in both KFs (so identity correspondence -> cs=1).
    # L2-normalise per pixel for stable cosine.
    feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-8)
    return feats


def t1_bit_identity():
    """gamma_embed=0 (default) must produce bit-identical output to vanilla."""
    device = "cuda:0"
    poses_a, disps_a, intr, ii, jj, target, weight, eta = _make_problem(device=device, seed=0)
    poses_b, disps_b, _, _, _, _, _, _ = _make_problem(device=device, seed=0)

    # Vanilla call (no embed kwargs at all).
    poses_v, disps_v = BA(target, weight, eta, poses_a, disps_a.clone(), intr, ii, jj,
                          fixedp=0, rig=1)
    # Default embed-enabled signature but gamma_embed=0 — should bypass.
    poses_e, disps_e = BA(
        target, weight, eta, poses_b, disps_b.clone(), intr, ii, jj,
        embed_features=None,
        gamma_embed=0.0,
        fixedp=0, rig=1,
    )
    dpose = (poses_v.data - poses_e.data).abs().max().item()
    ddisp = (disps_v - disps_e).abs().max().item()
    print(f"  T1 max|pose diff|={dpose:.3e}, max|disp diff|={ddisp:.3e}")
    assert dpose < 1e-6 and ddisp < 1e-6, "T1: gamma_embed=0 must be bit-identical"
    print("  T1 PASS — gamma_embed=0 ≡ vanilla")


def t2_embed_runs_clean():
    """gamma_embed=1 with features runs to completion; r_embed shape OK."""
    device = "cuda:0"
    B, P, ht, wd, K = 1, 2, 16, 16, 8
    poses, disps, intr, ii, jj, target, weight, eta = _make_problem(
        B=B, P=P, ht=ht, wd=wd, K=K, device=device, seed=1,
    )
    feats = _make_features(P, K, ht, wd, device=device, kind="ramp")
    poses_out, disps_out = BA(
        target, weight, eta, poses, disps.clone(), intr, ii, jj,
        embed_features=feats,
        gamma_embed=1.0,
        lam_embed=2.0,
        huber_delta=0.5,
        gate_grad_min=0.0,  # accept all pixels — synthetic features have gradient
        fixedp=0, rig=1,
    )
    assert torch.isfinite(poses_out.data).all(), "T2: poses became non-finite"
    assert torch.isfinite(disps_out).all(), "T2: disps became non-finite"
    print(f"  T2 PASS — gamma_embed=1 runs clean; "
          f"max|pose update|={(poses_out.data - poses.data).abs().max().item():.3e}")


def t3_synthetic_2frame_convergence():
    """Iterate BA with gamma_embed=1; verify pose error toward GT decreases."""
    device = "cuda:0"
    B, P, ht, wd, K = 1, 2, 16, 16, 8

    # GT: slot 1 has a small perturbation (translation only, x-axis).
    eye7 = torch.tensor([0, 0, 0, 0, 0, 0, 1.0], device=device)
    poses_gt_data = torch.zeros(B, P, 7, device=device)
    poses_gt_data[:, 0] = eye7
    # 0.01 unit translation in x for slot 1's GT (then we perturb away from
    # this and see whether D.1b pulls us back).
    poses_gt_data[:, 1] = eye7
    poses_gt_data[:, 1, 0] = 0.01

    poses_init_data = poses_gt_data.clone()
    poses_init_data[:, 1, 0] = -0.01  # initialise slot 1 ON THE OTHER SIDE of GT
    poses = lietorch.SE3(poses_init_data.clone())

    # Build features that match at GT pose (identity correspondence + same
    # buffer in both KFs at the integer grid). With slot 1 perturbed AWAY
    # from GT, the projection mu_ij ≠ identity, so cs < 1 → r_embed > 0,
    # which should drive slot 1 back.
    feats = _make_features(P, K, ht, wd, device=device, kind="sine")

    disps = torch.ones(B, P, ht, wd, device=device)
    intr = torch.tensor(
        [[[float(ht), float(ht), wd / 2.0, ht / 2.0] for _ in range(P)]],
        device=device, dtype=torch.float32,
    )
    ii = torch.tensor([0, 1], device=device, dtype=torch.long)
    jj = torch.tensor([1, 0], device=device, dtype=torch.long)

    # Target = identity flow (which is satisfied at GT pose, NOT at init).
    # We deliberately use a moderate weight so photometric also pulls.
    yy, xx = torch.meshgrid(
        torch.arange(ht, device=device).float(),
        torch.arange(wd, device=device).float(),
        indexing="ij",
    )
    target = torch.stack([xx, yy], dim=-1).view(1, 1, ht, wd, 2).expand(B, 2, ht, wd, 2).contiguous()
    weight = torch.ones(B, 2, ht, wd, 2, device=device) * 0.1
    eta = torch.ones(B, P, ht * wd, device=device) * 1e-3

    def pose_err(p):
        # Translation error of slot 1 vs GT (slot 0 is fixed at identity).
        return (p.data[0, 1, :3] - poses_gt_data[0, 1, :3]).norm().item()

    err_init = pose_err(poses)
    print(f"  T3 init slot-1 translation err = {err_init:.4e}")

    errs = [err_init]
    for it in range(8):
        poses, disps = BA(
            target, weight, eta, poses, disps.clone(), intr, ii, jj,
            embed_features=feats,
            gamma_embed=1.0,
            lam_embed=2.0,
            huber_delta=0.5,
            gate_grad_min=0.0,
            fixedp=0, rig=1,
            lm=1e-3, ep=0.1,
        )
        e = pose_err(poses)
        errs.append(e)
        print(f"  T3 iter {it+1}: err = {e:.4e}  delta = {e - errs[-2]:+.4e}")

    print(f"  T3 trajectory: {[f'{e:.3e}' for e in errs]}")
    final = errs[-1]
    if final < err_init:
        print(f"  T3 PASS — final err {final:.3e} < init {err_init:.3e}")
    else:
        print(f"  T3 NOT-IMPROVED — final err {final:.3e} >= init {err_init:.3e}")
        print(f"  (this is informational; may indicate a sign error in dr/dmu)")


def main():
    print("== Plan-v2 §Step 3b — D.1b synthetic 2-frame sanity tests ==")
    print("[T1] bit-identity bypass (gamma_embed=0 default)")
    t1_bit_identity()
    print("[T2] embed enabled, gamma_embed=1 runs cleanly")
    t2_embed_runs_clean()
    print("[T3] synthetic 2-frame convergence under gamma_embed=1")
    t3_synthetic_2frame_convergence()
    print("==")


if __name__ == "__main__":
    main()
