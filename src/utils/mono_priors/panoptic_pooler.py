"""
B.1 — per-instance feature pooler for Plan B's panoptic pipeline.

`pool_lang_features_in_mask(F, mask, patch_size)` aggregates a Phase B'
language-aligned feature tensor over the spatial extent of an instance mask,
producing one (D,) vector per instance. Mirrors `pool_features_in_box` from
`src/utils/mono_priors/seg_model.py:390` (the existing DINO box-pool utility)
but accepts a per-pixel mask AND operates on (D, h, w) axis-order tensors
rather than (h, w, D) row-major DINO maps.

Visibility-aware weighting (OVI-MAP §3.2 Eq. 2): each feature-grid cell
contributes proportional to the area of the image-resolution mask falling
inside its `patch_size × patch_size` covered block. Cells with zero overlap
are skipped. Pool is renormalized to L2.

Caller filters by `min_area` BEFORE calling this — empty/near-empty masks
return a zero vector and `is_empty=True`.

Tests (run via `pytest scripts/test_panoptic_pooler.py` or `python -m doctest`):

    >>> import torch, numpy as np
    >>> # Synthetic: F is (D=2, h=3, w=3); cell values encode (row, col).
    >>> F = torch.zeros(2, 3, 3)
    >>> for i in range(3):
    ...     for j in range(3):
    ...         F[0, i, j] = float(i)
    ...         F[1, i, j] = float(j)
    >>> # Mask covering only the center cell of the 9×9 image (patch=3).
    >>> mask = np.zeros((9, 9), dtype=bool)
    >>> mask[3:6, 3:6] = True
    >>> result = pool_lang_features_in_mask(F, mask, patch_size=3)
    >>> result.is_empty
    False
    >>> # Center cell is (1, 1) → unnormalized = (1, 1) → L2-normalized = (0.707, 0.707)
    >>> abs(float(result.feature[0]) - 0.7071) < 0.01
    True
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PoolResult:
    feature: torch.Tensor       # (D,) L2-normalized pooled feature, fp32
    weight_sum: float           # total mask area (pixels) used for the pool
    n_active_cells: int         # how many feature-grid cells contributed
    is_empty: bool              # True if mask was below the active-cell floor


def pool_lang_features_in_mask(
    F_grid: torch.Tensor,
    mask: np.ndarray,
    patch_size: int = 0,
    min_active_cells: int = 1,
    cell_overlap_thresh: float = 0.10,
    min_weight_sum: float = 0.5,
) -> PoolResult:
    """Visibility-weighted average pool of `F_grid` over `mask`.

    Args:
        F_grid: torch.Tensor of shape (D, h, w), any dtype. Phase B'
            lang-aligned feature for one keyframe.
        mask: np.ndarray of shape (H, W) bool / uint8. Image-resolution
            instance mask. (H, W) and (h, w) need NOT be related by a fixed
            integer patch_size — RADSeg's sliding-window aggregation produces
            a feature grid whose stride is set by SCRA/SCGA, not the RADIO
            patch_size. We use `F.adaptive_avg_pool2d` to compute the
            mask-fraction in each feature cell for any aspect.
        patch_size: kept for API stability only — not used. Pass anything.
        min_active_cells: minimum number of feature-grid cells whose overlap
            with `mask` exceeds `cell_overlap_thresh`. Below this → empty pool.
        cell_overlap_thresh: a feature cell "contributes" to the pool iff at
            least this fraction of its image-area block lies in `mask`.

    Returns:
        PoolResult — the L2-normalized (D,) feature plus diagnostic counts.
        On `is_empty=True`, `feature` is a zero vector.

    The visibility-weighted average is:
        weight_ij = (mask-fraction inside the (i, j) feature cell)
        pooled    = sum_ij(weight_ij * F[:, i, j]) / sum_ij(weight_ij)
        feature   = pooled / ||pooled||₂
    """
    _ = patch_size  # silence linter; intentionally unused, see docstring
    if not torch.is_tensor(F_grid):
        raise TypeError(f"F_grid must be torch.Tensor, got {type(F_grid)}")
    if F_grid.ndim != 3:
        raise ValueError(f"F_grid shape must be (D, h, w), got {tuple(F_grid.shape)}")
    D, h, w = F_grid.shape
    H, W = int(mask.shape[0]), int(mask.shape[1])
    if H < h or W < w:
        raise AssertionError(
            f"mask shape {(H, W)} smaller than feature grid {(h, w)} — "
            f"adaptive_avg_pool2d only supports downsampling, not upsampling"
        )

    device = F_grid.device
    if mask.dtype != bool:
        mask = mask.astype(bool)
    mask_t = torch.from_numpy(mask.astype(np.float32))[None, None].to(device)

    # Adaptive-average-pool the mask down to the exact feature grid — handles
    # any RADIO sliding-window stride / aspect-ratio mismatch correctly.
    weights_grid = F.adaptive_avg_pool2d(mask_t, output_size=(h, w))[0, 0]  # (h, w)
    weights_flat = weights_grid.flatten().to(F_grid.dtype)   # (h*w,)
    active = weights_flat >= cell_overlap_thresh
    n_active = int(active.sum().item())

    if n_active < min_active_cells:
        return PoolResult(
            feature=torch.zeros(D, dtype=torch.float32, device=device),
            weight_sum=float(weights_flat[active].sum().item()) if n_active else 0.0,
            n_active_cells=n_active,
            is_empty=True,
        )

    F_flat = F_grid.reshape(D, h * w).to(torch.float32)      # (D, h*w)
    w_active = weights_flat[active].to(torch.float32)
    F_active = F_flat[:, active]                             # (D, n_active)
    weight_sum = float(w_active.sum().item())
    # Reviewer 1 audit #7: tiny instances covering ~0.1 cell on the feat-grid
    # pool from a single noisy feature; require min_weight_sum >= 0.5 (half a
    # full feature cell of mask coverage) to commit to the pool.
    if weight_sum < min_weight_sum:
        return PoolResult(
            feature=torch.zeros(D, dtype=torch.float32, device=device),
            weight_sum=weight_sum, n_active_cells=n_active, is_empty=True,
        )
    pooled = (F_active * w_active.unsqueeze(0)).sum(dim=1) / weight_sum   # (D,)
    pooled_norm = F.normalize(pooled, dim=0, eps=1e-8)
    return PoolResult(
        feature=pooled_norm,
        weight_sum=weight_sum,
        n_active_cells=n_active,
        is_empty=False,
    )


def encode_pca256(feature_1536: torch.Tensor, mean: np.ndarray,
                  components: np.ndarray) -> np.ndarray:
    """Project a (D=1536,) torch tensor through the frozen PCA basis to (256,)."""
    if feature_1536.ndim != 1:
        raise ValueError(f"expected (D,), got {feature_1536.shape}")
    f = feature_1536.detach().cpu().numpy().astype(np.float32)  # (D,)
    centered = f - mean                                          # (D,)
    return (centered @ components.T).astype(np.float32)          # (target_dim,)
