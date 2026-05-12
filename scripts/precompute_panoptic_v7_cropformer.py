"""
Plan-v2 §Step 4 / B.1 — CropFormer entity-segmentation precompute (TRUE
OVI-MAP recipe, supersedes the earlier Mask2Former probe).

Per OVI-MAP §4 ("we adopt CropFormer for 2D entity segmentation"), this
script runs `qqlu/Entity/Entityv2/CropFormer` with the Swin-L w7 3x
checkpoint on every KF of a scene and saves a v2-schema entity-mask
artefact for downstream B.2 (cross-KF tracking + TSDF) and B.3 (per-
instance SigLIP-L labeling).

Output `<scene>/panoptic_v7_cropformer.npz`:

    schema_version       : int64 = 2
    proposer             : "cropformer_swin_large_w7_3x"
    n_keyframes          : int64
    image_hw             : (2,) int64
    kf_global_indices    : (N_kf,) int64    dataset frame indices (Report 23 v2)
    masks                : (N_total, H, W) bool   per-entity mask, flat-list
    seg_kf_offsets       : (N_kf + 1,) int64  per-KF slicing into masks/scores
    seg_scores           : (N_total,) float32  CropFormer instance score
    walltime_seconds     : float32

NOTE: CropFormer is class-agnostic — there are NO label_ids or names.
Class labels are assigned downstream by SigLIP per-instance (B.3).

Storage cost on 384x512 for 83 KFs at ~50 entities each:
    50 * 83 * 384 * 512 * 1 byte = ~800 MB raw. Compressed via np.savez:
    typically ~50-100 MB on RAM-backed data.

Usage (cvg, droid-w env):
    python scripts/precompute_panoptic_v7_cropformer.py \\
        --scene Outputs/TUM_RGBD/freiburg3_walking_static \\
        --rgb-dir datasets/TUM_RGBD/rgbd_dataset_freiburg3_walking_static/rgb \\
        --viz-kfs 9 40 75
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Make the qqlu/Entity CropFormer dir importable.
CROPFORMER_DIR = Path("/home/cvg/HERMES-SLAM/DROID-W/thirdparty/Entity/Entityv2/CropFormer")
sys.path.insert(0, str(CROPFORMER_DIR))
sys.path.insert(0, str(CROPFORMER_DIR / "demo_cropformer"))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import cv2
from PIL import Image

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from predictor import CropFormerPredictor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def setup_cfg(config_file: str, weights: str, device: str, score_thresh: float):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = device
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.freeze()
    return cfg


def _build_palette(n, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.integers(64, 256, size=(max(1, n), 3)).astype(np.uint8)
    p[0] = (0, 0, 0)
    return p


def _triptych(rgb_np, masks_kept):
    H, W, _ = rgb_np.shape
    palette = _build_palette(max(1, len(masks_kept) + 1))
    inst = np.zeros_like(rgb_np)
    for i, m in enumerate(masks_kept):
        inst[m] = palette[(i % (len(palette) - 1)) + 1]
    blend = (rgb_np.astype(np.float32) * 0.5 + inst.astype(np.float32) * 0.5).astype(np.uint8)
    sep = np.full((H, 8, 3), 255, dtype=np.uint8)
    return np.concatenate([rgb_np, sep, blend, sep, inst], axis=1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, type=Path)
    p.add_argument("--rgb-dir", default=None, type=Path)
    p.add_argument("--config", default=str(CROPFORMER_DIR /
                   "configs/entityv2/entity_segmentation/cropformer_swin_large_3x.yaml"),
                   type=str)
    p.add_argument("--weights", default="/home/cvg/HERMES-SLAM/DROID-W/weights/"
                   "CropFormer_model/Entity_Segmentation/CropFormer_swin_large_w7_3x/"
                   "CropFormer_swin_large_w7_3x_6843ef.pth", type=str)
    p.add_argument("--out-name", default="panoptic_v7_cropformer.npz", type=str)
    p.add_argument("--device", default="cuda:0", type=str)
    p.add_argument("--score-thresh", default=0.5, type=float)
    p.add_argument("--viz-kfs", nargs="*", type=int, default=[])
    p.add_argument("--viz-dir", default=None, type=Path)
    args = p.parse_args()

    scene = args.scene if args.scene.is_absolute() else REPO_ROOT / args.scene
    video = np.load(scene / "video.npz", allow_pickle=False)
    kf_global_indices = video["timestamps"].astype(np.int64)
    N_kf = len(kf_global_indices)
    _, _, H_img, W_img = video["images"].shape
    print(f"[setup] {scene.name}: N_kf={N_kf}  HxW={H_img}x{W_img}", flush=True)

    rgb_dir = args.rgb_dir if (args.rgb_dir and args.rgb_dir.is_absolute()) else (
        REPO_ROOT / args.rgb_dir) if args.rgb_dir else None
    rgb_files = sorted(f for f in rgb_dir.iterdir() if f.suffix.lower() == ".png") if rgb_dir else None

    print(f"[load] CropFormer Swin-L w7 3x  (config={Path(args.config).name})", flush=True)
    t0 = time.time()
    cfg = setup_cfg(args.config, args.weights, args.device, args.score_thresh)
    predictor = CropFormerPredictor(cfg)
    print(f"[load] done in {time.time()-t0:.1f}s", flush=True)

    viz_kfs = set(args.viz_kfs)
    viz_dir = args.viz_dir or (scene / "viz_v7_cropformer")
    if viz_kfs:
        viz_dir.mkdir(parents=True, exist_ok=True)

    all_masks = []
    all_scores = []
    seg_kf_offsets = [0]

    t_loop = time.time()
    n_total = 0
    for k in range(N_kf):
        if rgb_files:
            frame_idx = int(kf_global_indices[k])
            if frame_idx >= len(rgb_files):
                print(f"[WARN] kf{k}: frame_idx {frame_idx} OOB; skipping", flush=True)
                seg_kf_offsets.append(n_total)
                continue
            image_bgr = cv2.imread(str(rgb_files[frame_idx]))
        else:
            arr = (video["images"][k].transpose(1, 2, 0).clip(0, 255)).astype(np.uint8)
            image_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        predictions = predictor(image_bgr)
        instances = predictions["instances"]
        masks = instances.pred_masks.to("cpu").numpy().astype(bool)
        scores = instances.scores.to("cpu").numpy()
        keep = scores >= args.score_thresh
        masks = masks[keep]
        scores = scores[keep]

        all_masks.append(masks)
        all_scores.append(scores)
        n_total += len(masks)
        seg_kf_offsets.append(n_total)

        if k in viz_kfs:
            rgb_np = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            tri = _triptych(rgb_np, masks)
            out = viz_dir / f"kf{k:03d}_v7_cropformer_triptych.png"
            Image.fromarray(tri).save(out)
            print(f"[viz] kf{k}: {len(masks)} entities, saved {out.name}", flush=True)

        if (k + 1) % 10 == 0 or k == N_kf - 1 or k == 0:
            elapsed = time.time() - t_loop
            eta = elapsed / max(1, k + 1) * (N_kf - k - 1)
            print(f"  [{k+1:4d}/{N_kf}]  ({elapsed:6.1f}s, ~{eta:6.1f}s ETA, "
                  f"{len(masks)} entities)", flush=True)

    walltime = time.time() - t_loop

    # Pack into flat-list. Storage cost ~ N_total * H * W bytes -- expect 50-200 MB.
    masks_flat = np.concatenate(all_masks, axis=0) if all_masks else np.zeros((0, H_img, W_img), dtype=bool)
    scores_flat = np.concatenate(all_scores, axis=0) if all_scores else np.zeros((0,), dtype=np.float32)
    seg_kf_offsets = np.array(seg_kf_offsets, dtype=np.int64)

    out_path = scene / args.out_name
    np.savez(
        out_path,
        schema_version=np.int64(2),
        proposer=np.array("cropformer_swin_large_w7_3x"),
        n_keyframes=np.int64(N_kf),
        image_hw=np.array([H_img, W_img], dtype=np.int64),
        kf_global_indices=kf_global_indices,
        masks=masks_flat,
        seg_kf_offsets=seg_kf_offsets,
        seg_scores=scores_flat.astype(np.float32),
        walltime_seconds=np.float32(walltime),
    )
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\n[save] {out_path}  ({size_mb:.1f} MB)", flush=True)
    print(f"[summary] N_kf={N_kf}, total entities={n_total}, "
          f"avg entities/kf={n_total/N_kf:.1f}", flush=True)
    print(f"[time] {walltime:.1f}s total  ({1000*walltime/N_kf:.0f} ms/kf)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
