"""
Step 0.b visualiser (plan-v2): PCA-RGB feature overlay vs source RGB.

Loads <scene>/radseg_features.npz, picks a keyframe, PCA-reduces the
language-aligned features (D=1536) to 3 channels per pixel, upsamples to
image resolution, and saves a triptych (RGB | PCA-RGB | 50/50 blend) so the
spatial alignment between feature regions and image regions can be eyeballed.

Eyeball gate per plan-v2 Step 0.b: object silhouettes (especially the visible
person on freiburg3_walking_static KF 9) align between PCA-RGB and source RGB
to within ~2 px. Catches crop/resize/padding misalignment that a cosine
round-trip cannot.

Usage (cvg, droid-w conda env):
    python scripts/viz_feature_pca_rgb_v0.py \\
        --features Outputs/TUM_RGBD/freiburg3_walking_static/radseg_features.npz \\
        --rgb-dir datasets/TUM_RGBD/rgbd_dataset_freiburg3_walking_static/rgb \\
        --kf-index 9 \\
        --out Outputs/TUM_RGBD/freiburg3_walking_static/step0b_kf9_triptych.png
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
from PIL import Image

torch.backends.cudnn.deterministic = True
torch.manual_seed(0)


def _pca_rgb(feats: torch.Tensor) -> torch.Tensor:
    """(D, H, W) -> (3, H, W) in [0,1] via top-3 PCA components."""
    D, H, W = feats.shape
    X = feats.reshape(D, -1).T.float()                                 # (HW, D)
    X = X - X.mean(dim=0, keepdim=True)
    _, _, V = torch.pca_lowrank(X, q=3, center=False, niter=4)
    Y = X @ V                                                          # (HW, 3)
    lo = Y.quantile(0.02, dim=0)
    hi = Y.quantile(0.98, dim=0)
    Y = (Y - lo) / (hi - lo + 1e-8)
    return Y.clamp(0, 1).T.reshape(3, H, W)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--rgb-dir", required=True, type=Path)
    p.add_argument("--kf-index", default=9, type=int)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    d = np.load(args.features, allow_pickle=False)
    feats_all = d["lang_aligned_feats"]
    kf_indices = d["kf_indices"]
    H_img, W_img = (int(x) for x in d["image_hw"])

    kf = args.kf_index
    frame_idx = int(kf_indices[kf])
    print(f"[step0b] KF {kf} -> frame_idx={frame_idx}, image_hw=({H_img},{W_img})",
          flush=True)

    rgb_files = sorted([f for f in args.rgb_dir.iterdir() if f.suffix == ".png"])
    if frame_idx >= len(rgb_files):
        print(f"[ERR] frame_idx={frame_idx} >= {len(rgb_files)} RGB files",
              flush=True)
        return 2
    rgb_path = rgb_files[frame_idx]
    print(f"[step0b] rgb={rgb_path.name}", flush=True)
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))

    F_native = torch.from_numpy(feats_all[kf]).float()                 # (D, h, w)
    pca3 = _pca_rgb(F_native)                                          # (3, h, w)
    pca3_img = F.interpolate(pca3.unsqueeze(0), size=(H_img, W_img),
                             mode="bilinear", align_corners=False).squeeze(0)
    pca_rgb = (pca3_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    if rgb.shape[:2] != (H_img, W_img):
        rgb = np.asarray(Image.fromarray(rgb).resize((W_img, H_img),
                                                     Image.Resampling.BILINEAR))

    blend = (rgb.astype(np.float32) * 0.5
             + pca_rgb.astype(np.float32) * 0.5).astype(np.uint8)

    sep = np.full((H_img, 8, 3), 255, dtype=np.uint8)
    triptych = np.concatenate([rgb, sep, pca_rgb, sep, blend], axis=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(triptych).save(args.out)
    print(f"[step0b] saved triptych shape={triptych.shape} -> {args.out}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
