"""
Plan-v2 §Step 4 / B.1.5 — depth-discontinuity refinement of CropFormer
entity masks per MaskFusion (Rünz et al., ISMAR 2018, arXiv:1804.09194)
"geometric edges" rule.

Failure mode B.1.5 targets: CropFormer occasionally merges entities that
share a 2D boundary but lie at very different depths -- e.g. a person and
the door behind them, or a chair and the wall. Depth-edge thresholding on
the mono_disps map provides a geometry-only second opinion: if a 2D mask
is bisected by a strong depth jump, the two halves are two physical
objects.

Algorithm (per KF, per CropFormer mask):

  1. Convert disparity to depth on the BA grid:  D = 1 / disp.clamp(min=eps).
  2. Sobel gradient of depth:  |grad D| (m / pixel).
  3. Depth-edge map:  E_D = |grad D| > tau_edge (default 0.05 m/pixel).
  4. mask_minus_edges = mask AND NOT E_D.
  5. Connected components on mask_minus_edges.
  6. If there are >=2 CCs each above min_cc_size pixels AND the depth
     ranges of the CCs do NOT overlap (i.e. they are at different depths),
     split the mask into K new masks (one per qualifying CC).
  7. Otherwise keep the mask as-is.

Output `<scene>/panoptic_v7_cropformer_refined.npz` mirrors the v7 schema
but with possibly MORE masks per KF. seg_kf_offsets updates accordingly.

Usage (cvg, droid-w env):
    python scripts/refine_panoptic_with_depth_v7.py \\
        --scene Outputs/TUM_RGBD/freiburg3_walking_static \\
        --in-name panoptic_v7_cropformer.npz \\
        --tau-edge 0.05 \\
        --min-cc-size 50 \\
        --viz-kfs 9 40 75
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from PIL import Image
from scipy.ndimage import sobel, label as cc_label

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _depth_edge_map(disp_full: np.ndarray, eps_disp: float, tau_edge: float,
                    smooth_sigma: float = 1.5) -> np.ndarray:
    """disp_full: (H, W) fp32 monocular disparity.
    Returns depth-edge mask (H, W) bool — True where local depth gradient
    magnitude exceeds tau_edge (meters per pixel). Gaussian-smoothed first
    to suppress monocular-depth pixel noise."""
    from scipy.ndimage import gaussian_filter
    d = 1.0 / np.clip(disp_full, eps_disp, None)
    # Clip absurd depths from monocular noise (e.g. disp ~ 0 -> depth = 1/eps).
    d = np.clip(d, 0.1, 10.0)
    d_smooth = gaussian_filter(d, sigma=smooth_sigma)
    dx = sobel(d_smooth, axis=1, mode="reflect")
    dy = sobel(d_smooth, axis=0, mode="reflect")
    grad = np.sqrt(dx * dx + dy * dy)
    return grad > tau_edge


def _refine_one_mask(mask: np.ndarray, depth: np.ndarray,
                     depth_edge: np.ndarray, min_cc_size: int,
                     depth_overlap_thresh: float) -> list[np.ndarray]:
    """Try to split a single 2D mask at depth discontinuities.

    Returns a list of (H, W) bool masks. If no split is warranted, returns
    [mask] unchanged.
    """
    mask_kept = mask & ~depth_edge
    if mask_kept.sum() < min_cc_size:
        return [mask]  # too small after edge removal — don't split

    labels, n_cc = cc_label(mask_kept, structure=np.ones((3, 3), dtype=int))
    if n_cc <= 1:
        return [mask]

    # Filter components by size + collect their depth ranges.
    cc_list = []
    for k in range(1, n_cc + 1):
        comp_mask = labels == k
        if comp_mask.sum() < min_cc_size:
            continue
        comp_depth = depth[comp_mask]
        cc_list.append((comp_mask, float(comp_depth.min()), float(comp_depth.max())))

    if len(cc_list) <= 1:
        return [mask]

    # Determine whether the components are at *different* depths. Pairwise
    # depth-range non-overlap heuristic.
    cc_list.sort(key=lambda x: x[1])
    split_groups = [[cc_list[0]]]
    for cc, dmin, dmax in cc_list[1:]:
        prev_dmax = max(c[2] for c in split_groups[-1])
        prev_dmin = min(c[1] for c in split_groups[-1])
        # If this CC's depth range is well separated from the previous group:
        gap = dmin - prev_dmax
        prev_extent = prev_dmax - prev_dmin
        if gap > max(depth_overlap_thresh, 0.5 * prev_extent + 0.05):
            split_groups.append([(cc, dmin, dmax)])
        else:
            split_groups[-1].append((cc, dmin, dmax))

    if len(split_groups) <= 1:
        return [mask]

    # Materialise one new mask per group.
    new_masks = []
    for group in split_groups:
        m = np.zeros_like(mask, dtype=bool)
        for cc, _, _ in group:
            m |= cc
        # Restore the edge pixels closest to each new mask (so we don't lose
        # boundary pixels). For simplicity, dilate the new mask back to its
        # original support intersected with the (mask AND depth_edge) belt.
        # NOTE: this is the MaskFusion convention — the edge pixels go with
        # whichever side they connect to.
        # Approximation: add back the original mask's pixels NOT in any
        # other group's component support. Cheap proxy: just add the edge
        # pixels in this mask that are touching this group's CCs.
        # Defer: keep clean, no edge pixel restoration; B.2 spatial voting
        # is robust to ~1-pixel-thick mask shrinkage.
        new_masks.append(m)
    return new_masks


def _triptych(rgb_np, masks_before, masks_after, depth_edge,
              split_input_indices=None, split_output_groups=None):
    """If `split_input_indices` and `split_output_groups` are provided,
    highlight ONLY the masks that got split (rest are dim). Otherwise show
    all masks coloured normally."""
    H, W, _ = rgb_np.shape
    rng = np.random.default_rng(0)

    def _colour_overlay(masks, only_indices=None, highlight_pal=None):
        out = np.zeros_like(rgb_np)
        if only_indices is None:
            pal = rng.integers(64, 256, size=(max(1, len(masks) + 1), 3)).astype(np.uint8)
            for i, m in enumerate(masks):
                out[m] = pal[(i % (len(pal) - 1)) + 1]
        else:
            for k, i in enumerate(only_indices):
                if i >= len(masks):
                    continue
                col = highlight_pal[k % len(highlight_pal)] if highlight_pal is not None else (
                    rng.integers(64, 256, size=3).astype(np.uint8))
                out[masks[i]] = col
        return out

    if split_input_indices is not None:
        # Focused viz: each split-input gets a distinct hue; before shows the
        # one merged mask, after shows the K split children in the SAME hue
        # family (slightly different saturations).
        n_splits = len(split_input_indices)
        if n_splits > 0:
            hues = rng.integers(64, 256, size=(n_splits, 3)).astype(np.uint8)
            before_overlay = _colour_overlay(masks_before,
                                             only_indices=split_input_indices,
                                             highlight_pal=hues)
            # For "after", colour each new child mask with a slight variation of its parent's hue.
            after_overlay = np.zeros_like(rgb_np)
            for k, group in enumerate(split_output_groups):
                base = hues[k]
                for j, child_idx in enumerate(group):
                    if child_idx >= len(masks_after):
                        continue
                    shade = (base * (0.6 + 0.4 * (j % 3 + 1) / 3)).clip(0, 255).astype(np.uint8)
                    after_overlay[masks_after[child_idx]] = shade
        else:
            before_overlay = np.zeros_like(rgb_np)
            after_overlay = np.zeros_like(rgb_np)
    else:
        before_overlay = _colour_overlay(masks_before)
        after_overlay = _colour_overlay(masks_after)

    blend_b = (rgb_np.astype(np.float32) * 0.6 + before_overlay.astype(np.float32) * 0.4).astype(np.uint8)
    blend_a = (rgb_np.astype(np.float32) * 0.6 + after_overlay.astype(np.float32) * 0.4).astype(np.uint8)
    edges_view = rgb_np.copy()
    edges_view[depth_edge] = (220, 40, 40)
    sep = np.full((H, 8, 3), 255, dtype=np.uint8)
    return np.concatenate([blend_b, sep, edges_view, sep, blend_a], axis=1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, type=Path)
    p.add_argument("--rgb-dir", default=None, type=Path)
    p.add_argument("--in-name", default="panoptic_v7_cropformer.npz", type=str)
    p.add_argument("--out-name", default="panoptic_v7_cropformer_refined.npz", type=str)
    p.add_argument("--tau-edge", default=0.05, type=float,
                   help="Depth-gradient threshold in meters/pixel (Sobel magnitude). "
                        "0.05 m/pix = 5cm depth jump per pixel = strong discontinuity.")
    p.add_argument("--min-cc-size", default=50, type=int,
                   help="Discard split components below this pixel count.")
    p.add_argument("--depth-overlap-thresh", default=0.20, type=float,
                   help="Minimum depth-gap (m) between groups to keep them split.")
    p.add_argument("--eps-disp", default=1e-3, type=float)
    p.add_argument("--viz-kfs", nargs="*", type=int, default=[])
    p.add_argument("--viz-dir", default=None, type=Path)
    args = p.parse_args()

    scene = args.scene if args.scene.is_absolute() else REPO_ROOT / args.scene
    v = np.load(scene / args.in_name, allow_pickle=False)
    masks_in = v["masks"]                                              # (N_total, H_mask, W_mask) bool
    seg_kf_offsets_in = v["seg_kf_offsets"]                            # (N_kf+1,) int64
    seg_scores_in = v["seg_scores"]                                    # (N_total,) fp32
    kf_global_indices = v["kf_global_indices"]
    H_mask, W_mask = masks_in.shape[1:3] if len(masks_in) > 0 else (0, 0)
    N_kf = int(v["n_keyframes"])
    print(f"[setup] in={args.in_name}  N_kf={N_kf}  N_masks_in={len(masks_in)}  "
          f"mask_HW=({H_mask},{W_mask})", flush=True)

    video = np.load(scene / "video.npz", allow_pickle=False)
    # video.npz schema (Step 0 inspection):
    #   mono_disps      : (N_kf, H, W) fp32  -- monocular disp at image res
    #   droid_disps_up  : (N_kf, H, W) fp32  -- BA-optimised disp upsampled
    #   droid_disps     : (N_kf, h_ba, w_ba)
    # MaskFusion's geometric edges want the most metric-faithful depth.
    # mono_disps comes from the monocular depth network (DPT/DepthAnything-
    # style) and is scaled to the SLAM frame; droid_disps_up is the BA-
    # optimised version. For B.1.5 we use mono_disps as the geometry input
    # (matches the OVI-MAP recipe of using monocular depth for boundary
    # detection independent of SLAM-state convergence).
    mono_disps = video["mono_disps"]                                  # (N_kf, H_img, W_img) fp32
    H_img, W_img = mono_disps.shape[1:3]
    print(f"[setup] mono_disps shape={mono_disps.shape}, "
          f"range=[{mono_disps.min():.3f}, {mono_disps.max():.3f}]", flush=True)
    if (H_mask, W_mask) != (H_img, W_img):
        print(f"[setup] mask grid ({H_mask},{W_mask}) != disp grid "
              f"({H_img},{W_img}) -- masks will be resized down for the "
              f"depth-refine pass (nearest-neighbour).", flush=True)

    rgb_dir = args.rgb_dir if (args.rgb_dir and args.rgb_dir.is_absolute()) else (
        REPO_ROOT / args.rgb_dir) if args.rgb_dir else None
    rgb_files = sorted(f for f in rgb_dir.iterdir() if f.suffix.lower() == ".png") if rgb_dir else None

    viz_dir = args.viz_dir or (scene / "viz_v7_refined")
    viz_kfs = set(args.viz_kfs)
    if viz_kfs:
        viz_dir.mkdir(parents=True, exist_ok=True)

    all_masks_out = []
    all_scores_out = []
    seg_kf_offsets_out = [0]
    n_in_total = 0
    n_out_total = 0
    n_split_total = 0
    t0 = time.time()

    for k in range(N_kf):
        m_start = int(seg_kf_offsets_in[k])
        m_end = int(seg_kf_offsets_in[k + 1])
        kf_masks = masks_in[m_start:m_end]
        kf_scores = seg_scores_in[m_start:m_end]
        n_in_total += len(kf_masks)

        disp_k = mono_disps[k].astype(np.float32)
        depth_k = 1.0 / np.clip(disp_k, args.eps_disp, None)
        depth_edge = _depth_edge_map(disp_k, args.eps_disp, args.tau_edge)

        # If masks are at a different resolution than disps, downsample masks
        # (nearest) to the disp grid; ALL bookkeeping then happens at the
        # disp grid. The output `masks` will also be at the disp grid which
        # is the canonical grid for B.2 tracking + B.3 best-view sampling.
        if (H_mask, W_mask) != (H_img, W_img) and len(kf_masks) > 0:
            import cv2 as _cv2
            kf_masks_resized = np.stack([
                _cv2.resize(m.astype(np.uint8), (W_img, H_img),
                            interpolation=_cv2.INTER_NEAREST).astype(bool)
                for m in kf_masks
            ], axis=0)
        else:
            kf_masks_resized = kf_masks

        masks_after = []
        scores_after = []
        split_in_idx = []                 # indices into kf_masks_resized that got split
        split_out_groups = []             # for each split, list of indices into masks_after for its children
        for in_i, (m, s) in enumerate(zip(kf_masks_resized, kf_scores)):
            new_ms = _refine_one_mask(m, depth_k, depth_edge,
                                       args.min_cc_size, args.depth_overlap_thresh)
            if len(new_ms) > 1:
                n_split_total += 1
                start_idx = len(masks_after)
                split_in_idx.append(in_i)
                split_out_groups.append(list(range(start_idx, start_idx + len(new_ms))))
            for nm in new_ms:
                masks_after.append(nm)
                scores_after.append(float(s))

        all_masks_out.extend(masks_after)
        all_scores_out.extend(scores_after)
        n_out_total += len(masks_after)
        seg_kf_offsets_out.append(n_out_total)

        if k in viz_kfs and rgb_files:
            frame_idx = int(kf_global_indices[k])
            if frame_idx < len(rgb_files):
                import cv2
                image_bgr = cv2.imread(str(rgb_files[frame_idx]))
                image_bgr = cv2.resize(image_bgr, (W_img, H_img))
                rgb_np = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                # All-masks viz
                tri = _triptych(rgb_np, list(kf_masks_resized), masks_after, depth_edge)
                Image.fromarray(tri).save(viz_dir / f"kf{k:03d}_v7_refined_all.png")
                # Focused viz: only the masks that got split, highlighted
                tri_focus = _triptych(rgb_np, list(kf_masks_resized), masks_after, depth_edge,
                                       split_input_indices=split_in_idx,
                                       split_output_groups=split_out_groups)
                Image.fromarray(tri_focus).save(viz_dir / f"kf{k:03d}_v7_refined_focus.png")

                # ── Single-example viz: pick the split with the MOST children ──
                if split_in_idx:
                    best = max(range(len(split_in_idx)),
                                key=lambda j: len(split_out_groups[j]))
                    parent_idx = split_in_idx[best]
                    parent_mask = kf_masks_resized[parent_idx]
                    children = [masks_after[ci] for ci in split_out_groups[best]
                                 if ci < len(masks_after)]
                    n_ch = len(children)
                    # Build a 3-panel single-example figure:
                    #   RGB + parent (red) | depth-edges inside parent (red on grey) | parent split children (colors)
                    rgb_dim = (rgb_np.astype(np.float32) * 0.5).astype(np.uint8)
                    p_view = rgb_np.copy()
                    p_view[parent_mask] = (0.4 * p_view[parent_mask].astype(np.float32)
                                            + 0.6 * np.array([220, 40, 40])).astype(np.uint8)

                    edge_in_parent = depth_edge & parent_mask
                    e_view = rgb_dim.copy()
                    e_view[edge_in_parent] = (220, 40, 40)

                    c_view = rgb_np.copy()
                    pal_c = np.array([
                        (220, 60, 60), (60, 180, 60), (60, 60, 220),
                        (220, 180, 60), (180, 60, 220), (60, 220, 220),
                    ], dtype=np.uint8)
                    for j, child in enumerate(children):
                        col = pal_c[j % len(pal_c)]
                        c_view[child] = (0.35 * c_view[child].astype(np.float32)
                                          + 0.65 * col).astype(np.uint8)

                    sep = np.full((H_img, 8, 3), 255, dtype=np.uint8)
                    panel = np.concatenate([p_view, sep, e_view, sep, c_view], axis=1)
                    Image.fromarray(panel).save(viz_dir / f"kf{k:03d}_v7_example_split.png")
                    print(f"[viz] kf{k}: {len(kf_masks)} -> {len(masks_after)} entities; "
                          f"{len(split_in_idx)} merged-masks split into "
                          f"{sum(len(g) for g in split_out_groups)} children; "
                          f"example: 1 parent (idx {parent_idx}, area={int(parent_mask.sum())} px) "
                          f"-> {n_ch} depth-coherent children", flush=True)
                else:
                    print(f"[viz] kf{k}: 0 splits", flush=True)

        if (k + 1) % 10 == 0 or k == N_kf - 1 or k == 0:
            elapsed = time.time() - t0
            eta = elapsed / max(1, k + 1) * (N_kf - k - 1)
            print(f"  [{k+1:4d}/{N_kf}]  ({elapsed:5.1f}s, ~{eta:5.1f}s ETA, "
                  f"{len(kf_masks)} -> {len(masks_after)})", flush=True)

    walltime = time.time() - t0

    masks_flat = np.stack(all_masks_out, axis=0) if all_masks_out else np.zeros((0, H_img, W_img), dtype=bool)
    scores_flat = np.asarray(all_scores_out, dtype=np.float32)
    seg_kf_offsets_out = np.array(seg_kf_offsets_out, dtype=np.int64)

    out_path = scene / args.out_name
    np.savez(
        out_path,
        schema_version=np.int64(2),
        proposer=np.array(str(v["proposer"]) + "+B1.5_depth_refine"),
        n_keyframes=np.int64(N_kf),
        image_hw=np.array([H_img, W_img], dtype=np.int64),
        kf_global_indices=kf_global_indices,
        masks=masks_flat,
        seg_kf_offsets=seg_kf_offsets_out,
        seg_scores=scores_flat,
        tau_edge=np.float32(args.tau_edge),
        min_cc_size=np.int64(args.min_cc_size),
        depth_overlap_thresh=np.float32(args.depth_overlap_thresh),
        walltime_seconds=np.float32(walltime),
    )
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\n[save] {out_path}  ({size_mb:.1f} MB)", flush=True)
    print(f"[summary] N_kf={N_kf}", flush=True)
    print(f"  entities BEFORE: {n_in_total}  (avg {n_in_total/N_kf:.1f}/kf)", flush=True)
    print(f"  entities AFTER : {n_out_total}  (avg {n_out_total/N_kf:.1f}/kf)", flush=True)
    print(f"  masks split    : {n_split_total}  ({100*n_split_total/max(1, n_in_total):.1f}% of input)", flush=True)
    print(f"[time] {walltime:.1f}s total  ({1000*walltime/N_kf:.0f} ms/kf)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
