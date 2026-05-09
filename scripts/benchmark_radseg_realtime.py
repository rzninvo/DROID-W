"""
Phase A — measure real-time feasibility of our radseg open-vocab segmenter.

Decides whether HERMES-SLAM can run RadioGrounder INLINE per DROID-W
keyframe at the SLAM keyframe rate (5-10 Hz target on the 5090).

Sweeps four configs, all mirroring designs found in the literature:
    1. v4-h + siglip2-g + SAM-3      — our current published default
    2. v4-h + siglip2-g + no SAM     — drop SAM-3 refinement
    3. v3-b + siglip2 + SAM-1        — RADIO-ViPE's vendored default with refinement
    4. v3-b + siglip2 + no SAM       — RADIO-ViPE's RADSeg-paper Table-2 recipe

Per config, on N keyframes from `<scene>/video.npz`:
    - mean / median / p99 wall-clock latency per `grounder.segment(rgb)`
    - peak GPU memory
    - per-step breakdown (RADIO encode, lang-align, SAM refine) when isolable
    - mask coverage: mean fraction of pixels with `best_score >= threshold`,
      averaged across the queried classes

Runtime: ~5-10 min total at N=50 (cold start dominates the first call).

Usage (on cvg):
    python scripts/benchmark_radseg_realtime.py \
        --scene Outputs/TUM_RGBD/freiburg3_walking_static \
        --queries person monitor "office chair" radiator door floor \
        --threshold 0.50 \
        --n-keyframes 50 \
        --output Outputs/eval/radseg_realtime_freiburg3.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.mono_priors.radio_grounding import RadioGrounder


def _percentile(xs: List[float], q: float) -> float:
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))


def _bench_one_config(
    images: np.ndarray,         # (N, 3, H, W) uint8
    queries: List[str],
    radio_version: str,
    sam_refinement: bool,
    threshold: float,
    device: str,
    skip_warmup: int = 3,
) -> dict:
    """Run grounder.segment() over all images; return latency + coverage stats."""
    print(f"\n=== bench: radio={radio_version}  sam={sam_refinement} ===", flush=True)

    # Free any prior CUDA state — keep configs from polluting each other.
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t_load = time.time()
    grounder = RadioGrounder(
        device=device,
        radio_version=radio_version,
        sam_refinement=sam_refinement,
    )
    grounder.set_queries(queries)
    load_seconds = time.time() - t_load
    load_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"  load: {load_seconds:.1f}s  GPU peak after load: {load_mem_gb:.2f} GB",
          flush=True)

    latencies = []
    cov_per_q = np.zeros(len(queries), dtype=np.float64)
    cov_n = 0

    torch.cuda.reset_peak_memory_stats()
    for i in range(len(images)):
        rgb = images[i].transpose(1, 2, 0)  # (H, W, 3) uint8
        torch.cuda.synchronize()
        t0 = time.time()
        result = grounder.segment(rgb)
        torch.cuda.synchronize()
        dt = time.time() - t0
        if i >= skip_warmup:
            latencies.append(dt)
            best_score = result["best_score"]
            best_q = result["best_query"]
            for qi in range(len(queries)):
                m = (best_score >= threshold) & (best_q == qi)
                cov_per_q[qi] += float(m.mean())
            cov_n += 1
        if (i + 1) % 10 == 0 or i == 0:
            print(f"    [{i+1:3d}/{len(images)}] dt={dt:.3f}s", flush=True)
        del result
        # NOTE: deliberately NOT calling torch.cuda.empty_cache() here — it
        # adds 1-5 ms of cudaFree overhead per iter and disturbs the allocator,
        # uniformly inflating measured latency. The per-iteration `del result`
        # is enough to release intermediate tensors. Empty-cache happens once
        # between configs in the outer loop. (Reviewer A audit, Report 14.)

    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    cov_mean = (cov_per_q / max(1, cov_n)).tolist()

    out = {
        "radio_version": radio_version,
        "sam_refinement": bool(sam_refinement),
        "lang_adaptor": grounder.lang_adaptor_name,
        "queries": queries,
        "threshold": threshold,
        "n_warmup": skip_warmup,
        "n_timed": len(latencies),
        "load_seconds": load_seconds,
        "load_mem_gb": load_mem_gb,
        "peak_mem_gb": peak_mem_gb,
        "latency_seconds": {
            "mean": float(np.mean(latencies)) if latencies else None,
            "median": float(np.median(latencies)) if latencies else None,
            "p90": _percentile(latencies, 90) if latencies else None,
            "p99": _percentile(latencies, 99) if latencies else None,
            "min": float(min(latencies)) if latencies else None,
            "max": float(max(latencies)) if latencies else None,
        },
        "fps": {
            "mean": (1.0 / float(np.mean(latencies))) if latencies else None,
            "median": (1.0 / float(np.median(latencies))) if latencies else None,
        },
        "coverage_per_query": dict(zip(queries, cov_mean)),
    }

    # Drop the model to free memory before the next config.
    del grounder
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, type=Path,
                   help="Scene dir with video.npz (e.g. Outputs/TUM_RGBD/freiburg3_walking_static)")
    p.add_argument("--queries", required=True, nargs="+",
                   help="Text queries; pass the same list across all configs.")
    p.add_argument("--threshold", type=float, default=0.50,
                   help="Softmax threshold for coverage measurement (matches test_radio_segmentation.py).")
    p.add_argument("--n-keyframes", type=int, default=50,
                   help="Subsample first N keyframes from video.npz.")
    p.add_argument("--output", required=True, type=Path,
                   help="JSON output path.")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--skip", nargs="+", default=[],
                   help="Optionally skip specific configs by tag, e.g. --skip v3b_sam.")
    args = p.parse_args()

    # Load images.
    npz_path = args.scene.resolve() / "video.npz"
    if not npz_path.exists():
        print(f"[ERR] {npz_path} not found", flush=True)
        return 1
    z = np.load(npz_path)
    images = z["images"]
    if images.dtype != np.uint8:
        images = (images * 255.0).clip(0, 255).astype(np.uint8)
    if args.n_keyframes is not None and args.n_keyframes < len(images):
        images = images[: args.n_keyframes]
    print(f"[setup] scene={args.scene}  N_kf_used={len(images)}  HxW={images.shape[2]}x{images.shape[3]}",
          flush=True)
    print(f"[setup] queries={args.queries}  threshold={args.threshold}", flush=True)

    skip = set(args.skip)
    configs = [
        ("v4h_sam3",    "c-radio_v4-h", True),
        ("v4h_no_sam",  "c-radio_v4-h", False),
        ("v3b_sam1",    "c-radio_v3-b", True),
        ("v3b_no_sam",  "c-radio_v3-b", False),
    ]

    results = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for tag, version, sam in configs:
        if tag in skip:
            print(f"\n[skip] {tag}", flush=True)
            continue
        try:
            r = _bench_one_config(
                images, args.queries, version, sam, args.threshold, args.device,
            )
            results[tag] = r
            # Save incrementally so a late OOM doesn't lose prior results.
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  [save] {args.output}", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            results[tag] = {"error": "CUDA OOM", "details": str(e)[:200]}
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            torch.cuda.empty_cache()
            gc.collect()
            print(f"  [WARN] {tag}: OOM — recorded and continuing", flush=True)

    # Final summary table.
    print(f"\n{'='*84}", flush=True)
    print(f"{'config':<15} {'fps_mean':>9} {'fps_med':>9} {'lat_mean':>10} {'lat_p99':>10} {'mem_gb':>7}",
          flush=True)
    print(f"{'-'*84}", flush=True)
    for tag in [c[0] for c in configs]:
        r = results.get(tag)
        if r is None or "error" in r:
            note = (r.get("error", "skipped") if r else "skipped")
            print(f"{tag:<15} {note}", flush=True)
            continue
        lat = r["latency_seconds"]
        fps = r["fps"]
        print(f"{tag:<15} {fps['mean']:>9.2f} {fps['median']:>9.2f} {lat['mean']*1000:>9.1f}ms "
              f"{lat['p99']*1000:>9.1f}ms {r['peak_mem_gb']:>7.2f}",
              flush=True)
    print(f"{'='*84}\n", flush=True)
    print(f"[done] full results at {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
