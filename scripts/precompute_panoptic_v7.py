"""
Plan-v2 §Step 4 / B.1 — entity-level panoptic precompute for the v7 chain.

Replaces the v5 SAM-1 AMG path (which fragments objects into 5-8 part-masks
per person, Report 19 blocker) with `facebook/mask2former-swin-large-coco-
panoptic` (Apache 2.0, plan-v2 §Step 4 fallback). The qqlu/CropFormer model
is the paper-faithful choice but needs Detectron2; HF Mask2Former-Swin-L is
the acceptable substitute that ships in transformers and is what the smoke
test in `tests/smoke_cropformer_kf9.py` already validated (KF 9 of
walking_static -> 2 person masks, target <=2).

Output `<scene>/panoptic_v7.npz` schema (compatible-by-name with the v5
panoptic_amg.npz; the consumer in `build_panoptic_tracks_v7.py` reads the
v2-style fields):

    schema_version       : int64 = 2
    model_version        : U                facebook/mask2former-swin-large-coco-panoptic
    n_keyframes          : int64
    image_hw             : (2,) int64
    kf_global_indices    : (N_kf,) int64    dataset frame indices (matches
                                            radseg_features.npz v2 convention)
    segmentation         : (N_kf, H, W) uint16   per-pixel segment id (1..K_kf)
    seg_kf_offsets       : (N_kf + 1,) int64  flat-list offsets into the
                                              per-segment metadata arrays
    seg_ids              : (N_segs_total,) int64   segment id within its KF
    seg_label_ids        : (N_segs_total,) int64   COCO panoptic class id
    seg_label_names      : (N_segs_total,) U64     string class name
    seg_scores           : (N_segs_total,) float32
    seg_is_person        : (N_segs_total,) bool

Storage cost on 384x512: ~32 MB for 83 KFs (segmentation map alone). Per-
segment metadata is small.

Usage (cvg, droid-w env):
    python scripts/precompute_panoptic_v7.py \\
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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Re-use the viz palette logic from the smoke test for consistency.
def _build_palette(n, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.integers(64, 256, size=(max(1, n), 3)).astype(np.uint8)
    p[0] = (0, 0, 0)  # background id 0
    return p


def _triptych(rgb_np, seg_map, seg_info, id2label, palette):
    H, W, _ = rgb_np.shape
    all_inst = np.zeros_like(rgb_np)
    for s in seg_info:
        m = (seg_map == s["id"])
        idx = (s["id"] % (len(palette) - 1)) + 1
        all_inst[m] = palette[idx]
    blend = (rgb_np.astype(np.float32) * 0.5 + all_inst.astype(np.float32) * 0.5).astype(np.uint8)
    person_mask = np.zeros((H, W), dtype=bool)
    for s in seg_info:
        if "person" in id2label.get(s["label_id"], "").lower():
            person_mask |= (seg_map == s["id"])
    person_overlay = rgb_np.copy()
    if person_mask.any():
        person_overlay[person_mask] = (
            0.45 * rgb_np[person_mask].astype(np.float32)
            + 0.55 * np.array([220, 40, 40], dtype=np.float32)
        ).astype(np.uint8)
    sep = np.full((H, 8, 3), 255, dtype=np.uint8)
    return np.concatenate([rgb_np, sep, blend, sep, person_overlay], axis=1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, type=Path,
                   help="Scene Outputs dir; reads <scene>/video.npz for kf indices")
    p.add_argument("--rgb-dir", default=None, type=Path,
                   help="If omitted: read RGB from <scene>/video.npz['images']")
    p.add_argument("--model", default="facebook/mask2former-swin-large-coco-panoptic", type=str)
    p.add_argument("--device", default="cuda:0", type=str)
    p.add_argument("--score-thresh", default=0.5, type=float)
    p.add_argument("--out-name", default="panoptic_v7.npz", type=str)
    p.add_argument("--viz-kfs", nargs="*", type=int, default=[],
                   help="KF indices (in video.npz order) to also save triptych viz for")
    p.add_argument("--viz-dir", default=None, type=Path,
                   help="Where to write viz PNGs (default: <scene>/viz_v7/)")
    args = p.parse_args()

    scene = args.scene if args.scene.is_absolute() else REPO_ROOT / args.scene
    video_path = scene / "video.npz"
    if not video_path.exists():
        print(f"[ERR] missing {video_path}", flush=True)
        return 2
    video = np.load(video_path, allow_pickle=False)
    kf_global_indices = video["timestamps"].astype(np.int64)
    images_full = video["images"]                                 # (N_kf, 3, H, W) fp32
    N_kf = len(kf_global_indices)
    _, _, H_img, W_img = images_full.shape
    print(f"[setup] {scene.name}: N_kf={N_kf}  HxW={H_img}x{W_img}", flush=True)
    print(f"[setup] kf_global_indices[:10]: {kf_global_indices[:10].tolist()}", flush=True)

    # Optional: ground-truth dataset RGB images (sharper than video.npz's
    # downscaled-and-fp32 versions). For Mask2Former we use video.npz directly
    # to ensure pixel-alignment with downstream BA grid.
    rgb_dir = None
    if args.rgb_dir is not None:
        rgb_dir = args.rgb_dir if args.rgb_dir.is_absolute() else REPO_ROOT / args.rgb_dir
        if not rgb_dir.exists():
            print(f"[WARN] rgb-dir {rgb_dir} not found; falling back to video.npz", flush=True)
            rgb_dir = None
    rgb_files = sorted(rgb_dir.iterdir()) if rgb_dir else None
    if rgb_files:
        rgb_files = [f for f in rgb_files if f.suffix.lower() == ".png"]
        print(f"[setup] using {len(rgb_files)} RGB files from {rgb_dir}", flush=True)

    from transformers import (
        Mask2FormerForUniversalSegmentation,
        Mask2FormerImageProcessor,
    )
    print(f"[load] {args.model}", flush=True)
    t0 = time.time()
    processor = Mask2FormerImageProcessor.from_pretrained(args.model)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model).to(args.device).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    print(f"[load] done in {time.time()-t0:.1f}s ({len(id2label)} COCO panoptic classes)", flush=True)

    seg_maps = np.zeros((N_kf, H_img, W_img), dtype=np.uint16)
    all_seg_ids, all_label_ids, all_label_names, all_scores, all_is_person = [], [], [], [], []
    seg_kf_offsets = [0]
    viz_kfs = set(args.viz_kfs)
    viz_dir = args.viz_dir or (scene / "viz_v7")
    if viz_kfs:
        viz_dir.mkdir(parents=True, exist_ok=True)

    palette = _build_palette(512)

    t_loop = time.time()
    n_person_total = 0
    for k in range(N_kf):
        # Source RGB for inference: prefer the dataset PNG if available.
        if rgb_files:
            frame_idx = int(kf_global_indices[k])
            if frame_idx >= len(rgb_files):
                print(f"[WARN] kf{k}: frame_idx {frame_idx} > rgb_files; skipping", flush=True)
                continue
            image = Image.open(rgb_files[frame_idx]).convert("RGB")
            # Resize to video.npz image size for downstream alignment.
            if (image.height, image.width) != (H_img, W_img):
                image = image.resize((W_img, H_img), Image.Resampling.BILINEAR)
        else:
            arr = (images_full[k].transpose(1, 2, 0).clip(0, 255)).astype(np.uint8)
            image = Image.fromarray(arr)

        inputs = processor(images=image, return_tensors="pt").to(args.device)
        with torch.no_grad():
            outputs = model(**inputs)
        post = processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[(H_img, W_img)],
            threshold=args.score_thresh,
            mask_threshold=0.5,
        )[0]
        seg = post["segmentation"].cpu().numpy().astype(np.uint16)
        info = post["segments_info"]
        seg_maps[k] = seg

        n_p = 0
        for s in info:
            name = id2label.get(s["label_id"], "?")
            is_person = "person" in name.lower()
            n_p += int(is_person)
            all_seg_ids.append(int(s["id"]))
            all_label_ids.append(int(s["label_id"]))
            all_label_names.append(name)
            all_scores.append(float(s["score"]))
            all_is_person.append(bool(is_person))
        n_person_total += n_p
        seg_kf_offsets.append(len(all_seg_ids))

        if k in viz_kfs:
            rgb_np = np.asarray(image)
            tri = _triptych(rgb_np, seg, info, id2label, palette)
            out = viz_dir / f"kf{k:03d}_v7_triptych.png"
            Image.fromarray(tri).save(out)
            print(f"[viz] kf{k}: {len(info)} segments ({n_p} person), saved {out.name}", flush=True)

        if (k + 1) % 10 == 0 or k == N_kf - 1 or k == 0:
            elapsed = time.time() - t_loop
            eta = elapsed / max(1, k + 1) * (N_kf - k - 1)
            print(f"  [{k+1:4d}/{N_kf}]  ({elapsed:5.1f}s, ~{eta:5.1f}s ETA, "
                  f"{len(info)} seg, n_person={n_p})", flush=True)

    walltime = time.time() - t_loop
    out_path = scene / args.out_name
    np.savez(
        out_path,
        schema_version=np.int64(2),
        model_version=np.array(args.model),
        n_keyframes=np.int64(N_kf),
        image_hw=np.array([H_img, W_img], dtype=np.int64),
        kf_global_indices=kf_global_indices,
        segmentation=seg_maps,
        seg_kf_offsets=np.array(seg_kf_offsets, dtype=np.int64),
        seg_ids=np.array(all_seg_ids, dtype=np.int64),
        seg_label_ids=np.array(all_label_ids, dtype=np.int64),
        seg_label_names=np.array(all_label_names),
        seg_scores=np.array(all_scores, dtype=np.float32),
        seg_is_person=np.array(all_is_person, dtype=bool),
        walltime_seconds=np.float32(walltime),
    )
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\n[save] {out_path}  ({size_mb:.1f} MB)", flush=True)
    print(f"[summary] N_kf={N_kf}, total segments={len(all_seg_ids)}, "
          f"total person masks={n_person_total}, "
          f"avg segs/kf={len(all_seg_ids)/N_kf:.1f}, "
          f"avg person/kf={n_person_total/N_kf:.2f}", flush=True)
    print(f"[time] {walltime:.1f}s total  ({1000*walltime/N_kf:.0f} ms/kf)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
