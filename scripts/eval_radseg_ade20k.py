"""
ADE20K-150 mIoU eval for our RADSeg port (RADIO + SigLIP-2 + 80-template OVSS).

Protocol mirrors the RADSeg paper (arXiv 2511.19704) Table 2 — unrefined
RADSeg (no SAM mask refinement) on the 2000-image ADE20K validation
split. We extend the published recipe to the v4-h + siglip2-g substrate
that HERMES-SLAM defaults to (commit 1847d06), and report the v3-b +
siglip2 baseline alongside for direct comparison.

Per CLAUDE.md §6 (no silent fallbacks): ignore-class handling and
sliding-window aggregation are logged explicitly per image.

Refs:
- ADE20K-150 dataset: http://data.csail.mit.edu/places/ADEchallenge/
  ADEChallengeData2016.zip — class indices 1..150, 0 == void/unlabeled.
- objectInfo150.txt — TSV (Idx, Ratio, Train, Val, Name); names use
  comma-separated synonyms; we take the first synonym per RADSeg paper §4.
- mIoU formula: per-class IoU = TP/(TP+FP+FN) over the val split,
  averaged over the 150 classes (excluding any class with zero GT pixels);
  pixels with gt==0 are excluded from all per-class accumulators per
  ADE20K convention.

Usage (cvg):
    python scripts/eval_radseg_ade20k.py \\
        --ade-root /home/cvg/HERMES-SLAM/DROID-W/datasets/ADE20K/ADEChallengeData2016 \\
        --radio-version c-radio_v4-h \\
        --no-sam-refinement \\
        --output Outputs/eval/ade20k_v4h_nosam.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

# Allow `python scripts/...` from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.mono_priors.radio_grounding import RadioGrounder


def _load_class_names(obj_info_path: Path) -> list[str]:
    """Parse `objectInfo150.txt`. Returns list of 150 names (the FIRST
    comma-separated synonym for each class, lowercased, stripped). Order is
    by the file's `Idx` column (1..150)."""
    names: list[str] = []
    with open(obj_info_path) as f:
        next(f)  # header: Idx Ratio Train Val Name
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            idx = int(parts[0])
            full_name = parts[4].strip()
            # First synonym only — RADSeg paper §4 convention.
            primary = full_name.split(",")[0].strip().lower()
            assert idx == len(names) + 1, (
                f"objectInfo150.txt index mismatch: got {idx}, expected "
                f"{len(names) + 1}; format unexpected."
            )
            names.append(primary)
    if len(names) != 150:
        raise RuntimeError(
            f"Expected 150 ADE20K classes, parsed {len(names)} from "
            f"{obj_info_path}"
        )
    return names


def _load_validation_list(ade_root: Path) -> list[tuple[Path, Path]]:
    img_dir = ade_root / "images" / "validation"
    ann_dir = ade_root / "annotations" / "validation"
    images = sorted(img_dir.glob("*.jpg"))
    pairs = []
    for ip in images:
        ap = ann_dir / (ip.stem + ".png")
        if not ap.exists():
            print(f"[WARN] ade20k: missing annotation for {ip.name}, "
                  f"expected={ap} fallback=skipping", flush=True)
            continue
        pairs.append((ip, ap))
    if not pairs:
        raise RuntimeError(f"No (image, annotation) pairs found under {ade_root}")
    return pairs


def _accumulate_iou(
    pred: np.ndarray,           # (H, W) int in [-1, 149], -1 == "no class"
    gt: np.ndarray,             # (H, W) int in [0, 150],   0  == void
    n_classes: int,
    tp: np.ndarray, fp: np.ndarray, fn: np.ndarray,
    counts: np.ndarray,
) -> tuple[int, int, int]:
    """Update TP/FP/FN per class using ADE20K convention:
        - pixels with gt==0 are excluded entirely
        - pixels with pred==-1 are 'predicted void' → contribute as FN to
          whatever class GT assigns there (counts as a miss).
    Returns (n_valid_pixels, n_void_in_gt, n_void_in_pred) for logging."""
    valid = gt > 0
    n_void_gt = int((~valid).sum())
    n_void_pred = int((pred == -1).sum())
    if not valid.any():
        return 0, n_void_gt, n_void_pred

    g = gt[valid].astype(np.int64) - 1   # (n_valid,) ∈ [0, 149]
    p = pred[valid].astype(np.int64)     # (n_valid,) ∈ [-1, 149]

    # Per-class: count pixels by GT class (always valid)
    np.add.at(counts, g, 1)

    # FN contribution from pred==-1: these GT pixels are missed by every
    # class → bump fn[g] for those.
    miss = p < 0
    np.add.at(fn, g[miss], 1)

    # For pixels with valid prediction, do per-class TP/FP/FN.
    keep = ~miss
    if keep.any():
        gv = g[keep]; pv = p[keep]
        match = pv == gv
        # TP for the matched class
        np.add.at(tp, gv[match], 1)
        # FN for GT class where pred missed
        miss_pred = ~match
        if miss_pred.any():
            np.add.at(fn, gv[miss_pred], 1)
            # FP for predicted class where it didn't match GT
            np.add.at(fp, pv[miss_pred], 1)

    return int(valid.sum()), n_void_gt, n_void_pred


def _compute_metrics(
    tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, counts: np.ndarray,
) -> dict:
    n_classes = len(tp)
    denom = (tp + fp + fn).astype(np.float64)
    per_class_iou = np.where(denom > 0, tp.astype(np.float64) / denom, np.nan)

    seen = counts > 0
    n_seen = int(seen.sum())
    miou = float(np.nanmean(per_class_iou[seen])) if n_seen > 0 else float("nan")

    total = int(counts.sum())
    if total > 0 and n_seen > 0:
        freq = counts.astype(np.float64) / total
        # FW-mIoU: sum over classes of freq * IoU; classes with iou==nan
        # contribute 0 (no GT pixels → freq is 0 too, so contribution is 0).
        fw_iou = float(np.nansum(freq * per_class_iou))
    else:
        fw_iou = float("nan")

    pixel_acc = float(tp.sum() / max(1, total))

    return {
        "mIoU": miou,
        "fwIoU": fw_iou,
        "pixel_accuracy": pixel_acc,
        "n_classes_seen": n_seen,
        "n_classes_total": n_classes,
        "per_class_iou": per_class_iou.tolist(),
        "per_class_pixel_count": counts.tolist(),
        "per_class_tp": tp.tolist(),
        "per_class_fp": fp.tolist(),
        "per_class_fn": fn.tolist(),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="ADE20K-150 mIoU eval for our RADSeg port.")
    p.add_argument("--ade-root", required=True, type=Path,
                   help="Path to ADEChallengeData2016/ root.")
    p.add_argument("--radio-version", required=True, type=str,
                   help="e.g. c-radio_v4-h, c-radio_v3-b")
    p.add_argument("--lang-adaptor", default=None, type=str,
                   help="Override; auto-detected from version when omitted.")
    p.add_argument("--no-sam-refinement", action="store_true",
                   help="Disable SAM refinement to match RADSeg paper Table 2 protocol.")
    p.add_argument("--output", required=True, type=Path,
                   help="JSON output path (per-class IoU + summary).")
    p.add_argument("--device", default="cuda:0", type=str)
    p.add_argument("--max-images", default=None, type=int,
                   help="Optional cap on validation images (debug).")
    p.add_argument("--print-every", default=50, type=int)
    p.add_argument("--ckpt-every", default=200, type=int,
                   help="Save resume-checkpoint every N images.")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore any existing checkpoint and start from image 0.")
    p.add_argument("--max-side", default=1536, type=int,
                   help="If image's long side exceeds this (in px), retry on "
                        "OOM at downsampled resolution (long_side<=max_side).")
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[setup] ade_root={args.ade_root}", flush=True)
    print(f"[setup] radio_version={args.radio_version}  "
          f"sam_refinement={not args.no_sam_refinement}", flush=True)
    print(f"[setup] output={args.output}", flush=True)

    obj_info = args.ade_root / "objectInfo150.txt"
    class_names = _load_class_names(obj_info)
    print(f"[setup] loaded {len(class_names)} class names. "
          f"first 5 = {class_names[:5]}", flush=True)

    pairs = _load_validation_list(args.ade_root)
    if args.max_images is not None:
        pairs = pairs[: args.max_images]
    print(f"[setup] {len(pairs)} (image, annotation) pairs to process", flush=True)

    t0 = time.time()
    grounder = RadioGrounder(
        device=args.device,
        radio_version=args.radio_version,
        lang_adaptor=args.lang_adaptor,
        sam_refinement=(not args.no_sam_refinement),
    )
    grounder.set_queries(class_names)
    print(f"[load] grounder ready in {time.time()-t0:.1f}s "
          f"(patch={grounder.patch_size}, lang={grounder.lang_adaptor_name})",
          flush=True)

    n_classes = len(class_names)
    tp = np.zeros(n_classes, dtype=np.int64)
    fp = np.zeros(n_classes, dtype=np.int64)
    fn = np.zeros(n_classes, dtype=np.int64)
    counts = np.zeros(n_classes, dtype=np.int64)

    n_total_valid_px = 0
    n_total_void_gt = 0
    n_total_void_pred = 0

    # Resume from checkpoint if present (per CLAUDE.md §6 — log explicitly).
    ckpt_path = args.output.with_suffix(".ckpt.npz")
    start_i = 0
    if ckpt_path.exists() and not args.no_resume:
        ckpt = np.load(ckpt_path)
        if (int(ckpt["n_classes"]) == n_classes
                and int(ckpt["n_images"]) == len(pairs)
                and str(ckpt["radio_version"]) == args.radio_version):
            tp[:] = ckpt["tp"]; fp[:] = ckpt["fp"]; fn[:] = ckpt["fn"]
            counts[:] = ckpt["counts"]
            start_i = int(ckpt["next_i"])
            n_total_valid_px = int(ckpt["n_total_valid_px"])
            n_total_void_gt = int(ckpt["n_total_void_gt"])
            n_total_void_pred = int(ckpt["n_total_void_pred"])
            print(f"[resume] {ckpt_path} -> resuming at image {start_i}/{len(pairs)}",
                  flush=True)
        else:
            print(f"[WARN] ade20k: checkpoint {ckpt_path} does not match current "
                  f"config (n_classes/radio_version/n_images differ); "
                  f"fallback=ignoring checkpoint, restarting from image 0",
                  flush=True)

    n_oom_resized = 0
    t_loop = time.time()
    for i, (ip, ap) in enumerate(pairs):
        if i < start_i:
            continue
        gt = np.asarray(Image.open(ap), dtype=np.int64)  # (H, W) ∈ [0, 150]

        # SCGA's self-correlation matrix is O(num_tokens²); on ~4 MP ADE
        # images that's >10 GB of float32 and OOMs even with empty_cache().
        # Strategy: try native resolution; on OOM, resize so the longer
        # side is ≤ args.max_side and retry. Predictions get NN-resized back
        # to GT resolution before IoU. Per CLAUDE.md §6 we [WARN] every
        # fallback so reviewers can see how many images dropped resolution.
        rgb_full = np.asarray(Image.open(ip).convert("RGB"), dtype=np.uint8)
        rgb = rgb_full
        retried_at_lower_res = False
        try:
            result = grounder.segment(rgb)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            # Downsample so the long side is at most args.max_side.
            H0, W0 = rgb_full.shape[:2]
            scale = args.max_side / max(H0, W0)
            if scale >= 1.0:
                # We're already small; OOM came from elsewhere — skip image.
                print(f"[WARN] ade20k: OOM on image {i} ({ip.name}) at native "
                      f"{H0}x{W0} but max_side={args.max_side} >= max(H,W); "
                      f"fallback=skip image, accumulator unchanged",
                      flush=True)
                n_oom_resized += 1
                continue
            new_h = int(round(H0 * scale))
            new_w = int(round(W0 * scale))
            rgb = np.asarray(
                Image.fromarray(rgb_full).resize((new_w, new_h), Image.BILINEAR),
                dtype=np.uint8,
            )
            print(f"[WARN] ade20k: OOM on image {i} ({ip.name}) at native "
                  f"{H0}x{W0}, fallback=resize to {new_h}x{new_w} (long-side "
                  f"<= {args.max_side}px) and retry", flush=True)
            torch.cuda.empty_cache()
            result = grounder.segment(rgb)
            retried_at_lower_res = True
            n_oom_resized += 1

        pred = result["best_query"]                    # (H, W) ∈ [-1, n_classes-1]
        if pred.shape != gt.shape:
            # Resize-by-nearest if mismatch (always the case after a retry).
            pred_pil = Image.fromarray((pred + 1).astype(np.int32), mode="I")
            pred_pil = pred_pil.resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
            pred = np.asarray(pred_pil, dtype=np.int64) - 1

        n_valid, n_void_gt, n_void_pred = _accumulate_iou(
            pred, gt, n_classes, tp, fp, fn, counts,
        )
        n_total_valid_px += n_valid
        n_total_void_gt += n_void_gt
        n_total_void_pred += n_void_pred

        # Drop GPU caches every image — these eat the SCGA temporaries that
        # leak otherwise (observed OOM at image ~1100 on v4-h with another
        # 4-GB tenant on the GPU).
        del result, pred
        torch.cuda.empty_cache()

        if (i + 1) % args.print_every == 0 or i == 0 or i == len(pairs) - 1:
            elapsed = time.time() - t_loop
            eta = elapsed / max(1, i + 1 - start_i) * (len(pairs) - i - 1)
            running = _compute_metrics(tp, fp, fn, counts)
            print(
                f"  [{i+1:4d}/{len(pairs)}] "
                f"running mIoU={running['mIoU']*100:5.2f}  "
                f"fwIoU={running['fwIoU']*100:5.2f}  "
                f"acc={running['pixel_accuracy']*100:5.2f}  "
                f"void_gt={100*n_void_gt/(n_void_gt+n_valid):4.1f}%  "
                f"({elapsed:5.1f}s, ~{eta:5.1f}s ETA)",
                flush=True,
            )

        # Persist checkpoint every N images so we can resume after OOMs.
        if (i + 1) % args.ckpt_every == 0:
            np.savez(
                ckpt_path, tp=tp, fp=fp, fn=fn, counts=counts,
                next_i=i + 1,
                n_total_valid_px=n_total_valid_px,
                n_total_void_gt=n_total_void_gt,
                n_total_void_pred=n_total_void_pred,
                n_classes=n_classes,
                n_images=len(pairs),
                radio_version=args.radio_version,
            )

    final = _compute_metrics(tp, fp, fn, counts)
    final["radio_version"] = args.radio_version
    final["lang_adaptor"] = grounder.lang_adaptor_name
    final["sam_refinement"] = not args.no_sam_refinement
    final["n_images"] = len(pairs)
    final["n_total_valid_px"] = n_total_valid_px
    final["n_total_void_gt"] = n_total_void_gt
    final["n_total_void_pred"] = n_total_void_pred
    final["class_names"] = class_names
    final["walltime_seconds"] = time.time() - t_loop
    final["n_oom_resized"] = n_oom_resized

    with open(args.output, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\n=== ADE20K-150 final ({args.radio_version}, "
          f"SAM={'on' if not args.no_sam_refinement else 'off'}) ===", flush=True)
    print(f"  mIoU             = {final['mIoU']*100:.3f}", flush=True)
    print(f"  fwIoU            = {final['fwIoU']*100:.3f}", flush=True)
    print(f"  pixel_accuracy   = {final['pixel_accuracy']*100:.3f}", flush=True)
    print(f"  classes seen     = {final['n_classes_seen']}/{n_classes}", flush=True)
    print(f"  void_pixels (gt) = {n_total_void_gt}/{n_total_void_gt + n_total_valid_px} "
          f"({100*n_total_void_gt/max(1, n_total_void_gt + n_total_valid_px):.2f}%)",
          flush=True)
    print(f"[save] {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
