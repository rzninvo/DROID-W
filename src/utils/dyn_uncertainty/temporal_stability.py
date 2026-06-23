"""HERMES variant: temporal-stability adaptive robust kernel (ARK) for BA weights.

Ports the KM-ViPE / RADIO-ViPE temporal stability field S(u) and adaptive Barron
robust kernel (Nasser et al., arXiv:2604.26067, Eqs. 5, 8-11; reference code
be2rlab/RADIO-ViPE@0773ef4, vipe/slam/ba/terms.py + kernel.py) onto DROID-W's
factor-graph BA. The weight produced here multiplies the update-net confidence
weight that enters droid_backends.ba — a single application in the normal
equations (H = J^T W J), matching the reference solver
(ConcreteTermEvalReturn.jtwj/nwjtr), whose kernel likewise returns sqrt-IRLS
weights into the same slot.

Intentional differences vs the reference (both documented):
- Features: DROID-W's per-keyframe DINOv2/FiT3D maps (video.dino_feats_resize,
  already at BA resolution) instead of RADSeg. The stability formulation is
  feature-agnostic (cross-view cosine consistency of dense features).
- Cadence: computed once per factor-graph update() call (DROID's weights are
  constant across the inner Gauss-Newton iterations), whereas the reference
  re-evaluates inside every solver iteration. Because of this, the kernel is
  additionally gated until tracking is past warmup (the hooks check
  video.counter > tracking.warmup): at bootstrap the residual equals the full
  GRU delta and a frozen kernel evaluated there could trap initialization.
- Graphs over feature-less frames (the trajectory filler stages non-keyframes
  with empty feature slots) must construct FactorGraph(use_stability=False);
  this module raises on all-zero feature rows rather than degrade silently.

Gated by cfg['tracking']['stability']['enable']; False = canonical DROID-W,
bit-identical (this module is never called).
"""

import torch
import torch.nn.functional as F

CHUNK = 16  # edges per chunk when gathering/sampling feature maps (bounds peak memory)


def barron_irls_sqrt_weight(r: torch.Tensor, alpha: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """IRLS weight of Barron's general loss; verbatim port of
    AdaptiveBarronRobustKernel.apply (RADIO-ViPE kernel.py:55-99).
    Returns sqrt(w) — the reference multiplies this into per-residual weights.
    """
    s = (r / c) ** 2
    eps = 1e-8
    alpha_is_2 = torch.isclose(alpha, torch.full_like(alpha, 2.0), atol=1e-6)
    alpha_is_0 = torch.isclose(alpha, torch.zeros_like(alpha), atol=1e-6)
    abs_am2 = torch.abs(alpha - 2.0)
    abs_am2_safe = torch.where(alpha_is_2 | alpha_is_0, torch.ones_like(abs_am2), abs_am2 + eps)
    general = (1.0 / c ** 2) * (s / abs_am2_safe + 1.0) ** (alpha / 2.0 - 1.0)
    weights = torch.where(alpha_is_2, torch.ones_like(general), general)
    weights = torch.where(alpha_is_0, 2.0 / (2.0 + s), weights)
    return torch.sqrt(weights)


def stability_to_alpha(S: torch.Tensor, thresh_movable: float, thresh_static: float,
                       alpha_dynamic: float, alpha_huber: float, alpha_static: float) -> torch.Tensor:
    """Three-regime piecewise-linear S -> alpha mapping; verbatim port of
    _stability_to_alpha (RADIO-ViPE terms.py:538-572).
    """
    t_lo, t_hi = thresh_movable, thresh_static
    t_mid = (t_lo + t_hi) * 0.5
    t_lower = ((S - t_lo) / max(t_mid - t_lo, 1e-6)).clamp(0.0, 1.0)
    t_upper = ((S - t_mid) / max(t_hi - t_mid, 1e-6)).clamp(0.0, 1.0)
    alpha = torch.lerp(torch.full_like(S, alpha_dynamic), torch.full_like(S, alpha_huber), t_lower)
    alpha = torch.lerp(alpha, torch.full_like(S, alpha_static), t_upper)
    return alpha


@torch.no_grad()
def modulate_weight(video, ii: torch.Tensor, jj: torch.Tensor,
                    target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute S(u) over the factor-graph edges and return the ARK-modulated weight.

    target/weight: (1, N, h, w, 2) as passed to video.ba. Also refreshes
    video.stability[f] for every source frame f present in `ii` (export buffer).
    """
    cfg = video.cfg['tracking']['stability']
    if not video.uncertainty_aware:
        raise RuntimeError(
            "[stability] tracking.stability.enable=True requires "
            "tracking.uncertainty_params.activate=True (video.dino_feats_resize is "
            "needed for the cross-view similarity); refusing to run without features.")

    coords, valid = video.reproject(ii, jj)   # (1,N,h,w,2), (1,N,h,w,1)
    feats = video.dino_feats_resize           # (buffer, C, h, w)
    N = ii.shape[0]
    h, w = coords.shape[2], coords.shape[3]

    # every frame this graph touches must have features staged; an all-zero row
    # means a caller bug (e.g. trajectory-filler-style staged frames) and would
    # silently yield S=0 / alpha_dynamic everywhere
    frames = torch.unique(torch.cat([ii, jj]))
    feat_mass = feats.abs().sum(dim=(1, 2, 3))[frames]
    if (feat_mass == 0).any():
        bad = frames[feat_mass == 0].tolist()
        raise RuntimeError(
            f"[stability] frames {bad} have no DINO features staged; S(u) would be "
            "garbage. Construct this FactorGraph with use_stability=False or stage features.")

    # ---- cs_ij(u): cross-view cosine similarity (terms.py:457-504) ----
    cs_all = torch.zeros(N, h, w, device=coords.device, dtype=torch.float32)
    wh = torch.tensor([w - 1.0, h - 1.0], device=coords.device)
    for c0 in range(0, N, CHUNK):
        c1 = min(c0 + CHUNK, N)
        src_n = F.normalize(feats[ii[c0:c1]].float(), dim=1, eps=1e-8)
        grid = coords[0, c0:c1] * 2.0 / wh - 1.0
        smp = F.grid_sample(feats[jj[c0:c1]].float(), grid, mode='bilinear',
                            padding_mode='border', align_corners=True)
        smp_n = smp / smp.norm(dim=1, keepdim=True).clamp_min(1e-8)
        cs_all[c0:c1] = (src_n * smp_n).sum(dim=1) * valid[0, c0:c1, ..., 0]

    # ---- S_i(u) = mean_cs * (1 - var_cs), per source frame (terms.py:506-532) ----
    S_fields = {}
    for f in torch.unique(ii):
        cs_f = cs_all[ii == f]
        if cs_f.shape[0] == 1:
            S = cs_f[0].clamp(0.0, 1.0)
        else:
            S = (cs_f.mean(0) * (1.0 - cs_f.var(0, unbiased=False))).clamp(0.0, 1.0)
        fidx = int(f)
        S_fields[fidx] = S
        video.stability[fidx] = S

    # ---- per-edge alpha from min(S_i, S_j) (terms.py:578-612) ----
    S_DEFAULT = 0.5  # neutral: frames never appearing as a source
    max_id = int(torch.cat([ii, jj]).max()) + 1
    S_table = torch.full((max_id, h, w), S_DEFAULT, device=coords.device)
    for fidx, S in S_fields.items():
        S_table[fidx] = S
    S_edge = torch.minimum(S_table[ii], S_table[jj])
    alpha = stability_to_alpha(S_edge, cfg['thresh_movable'], cfg['thresh_static'],
                               cfg['alpha_dynamic'], cfg['alpha_huber'], cfg['alpha_static'])

    # ---- Barron sqrt-IRLS weight on the flow residual (Eqs. 10-11) ----
    r = (target - coords)[0]                                       # (N, h, w, 2), pixels
    w_ark = barron_irls_sqrt_weight(r, alpha.unsqueeze(-1).expand_as(r), c=cfg['c'])
    return weight * w_ark.unsqueeze(0)
