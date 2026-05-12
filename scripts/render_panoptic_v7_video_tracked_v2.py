"""Re-render to match kf9_cropformer_actual.png exactly (480x640 native
TUM-RGBD resolution, full triptych width 1936x480) using B.2 global
track IDs as the stable colour palette.

Differences from v1:
  - upsample B.1.5 refined masks 384x512 -> 480x640 via nearest-neighbour
    so the visualisation matches the user's chosen B.1 style exactly.
  - NO --min-track-kf filter. Every CropFormer-refined mask is drawn:
      * gid > 0: stable palette[gid] (cross-KF consistent colour)
      * gid = -1 (skipped during tracking <10 valid 3D pts):
        per-mask-deterministic dim grey colour so the right panel is
        full like the B.1 raw video, but the eye can still tell them
        from the tracked entities.
  - Same 8-px white separator, same triptych layout, same overlay blend.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--refined-npz", required=True, type=str)
    p.add_argument("--tracks-npz", required=True, type=str)
    p.add_argument("--rgb-dir", required=True, type=str)
    p.add_argument("--out", required=True, type=str)
    p.add_argument("--fps", default=6, type=int)
    p.add_argument("--sep-px", default=8, type=int)
    args = p.parse_args()

    refined = np.load(args.refined_npz, allow_pickle=True)
    tracks = np.load(args.tracks_npz, allow_pickle=True)

    masks_ref = refined["masks"]               # (N, 384, 512) bool
    offsets = refined["seg_kf_offsets"]
    kf_gi = refined["kf_global_indices"]
    n_kf = int(refined["n_keyframes"])
    Hm, Wm = int(masks_ref.shape[1]), int(masks_ref.shape[2])

    global_ids = tracks["global_track_ids"]
    n_global_tracks = int(tracks["n_global_tracks"])
    print(f"[load] {masks_ref.shape[0]} refined masks at {Hm}x{Wm}, "
          f"{n_global_tracks} global tracks, "
          f"{int((global_ids == -1).sum())} skipped",
          flush=True)

    # Determine native RGB resolution from disk.
    rgb_dir = Path(args.rgb_dir)
    rgb_files = sorted(f for f in rgb_dir.iterdir() if f.suffix == ".png")
    first = cv2.imread(str(rgb_files[0]))
    if first is None:
        print(f"[ERR] cannot read {rgb_files[0]}", flush=True)
        return 1
    H, W = first.shape[:2]
    print(f"[setup] RGB native res = {H}x{W}; upsampling masks {Hm}x{Wm} -> {H}x{W}",
          flush=True)

    # HSV golden-ratio palette: each label id gets a hue offset of phi
    # times the previous, so adjacent ids land in maximally-different hues.
    # High saturation and value -> distinct colours even for many labels.
    n_lab = int(global_ids.max()) + 1
    phi = 0.6180339887498949
    hues = ((np.arange(max(n_lab, 1)) * phi) % 1.0).astype(np.float32)
    # Vary saturation/value slightly per id for extra contrast.
    sats = 0.55 + 0.45 * ((np.arange(max(n_lab, 1)) * 0.7) % 1.0)
    vals = 0.75 + 0.25 * ((np.arange(max(n_lab, 1)) * 0.3 + 0.5) % 1.0)
    hsv = np.stack([hues * 179.0, sats * 255.0, vals * 255.0], axis=-1)
    hsv = hsv.reshape(-1, 1, 3).astype(np.uint8)
    palette = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape(-1, 3)
    palette[0] = (40, 40, 40)  # label 0 (never used) -> dark grey

    # Pre-build a "skipped" greyscale palette indexed by per-mask hash
    # (deterministic and dim, so skipped fragments are visible but quieter).
    # We map each skipped mask to a grey shade in [80, 140] via hash on
    # its global index, so different skipped fragments get distinct greys.
    skipped_grey = lambda m_global: int(80 + (m_global * 37) % 60)

    sep = np.full((H, args.sep_px, 3), 255, dtype=np.uint8)
    triptych_w = W * 3 + args.sep_px * 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(args.fps),
                             (triptych_w, H))
    if not writer.isOpened():
        print(f"[ERR] could not open VideoWriter for {out_path}", flush=True)
        return 1

    for k in range(n_kf):
        gi = int(kf_gi[k])
        bgr = cv2.imread(str(rgb_files[gi]))
        if bgr is None:
            print(f"[WARN] kf {k}: no RGB at {rgb_files[gi]}", flush=True)
            continue
        if bgr.shape[:2] != (H, W):
            bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        s, e = int(offsets[k]), int(offsets[k + 1])
        inst = np.zeros_like(rgb)
        n_tracked, n_skipped = 0, 0
        for m_global in range(s, e):
            mask_low = masks_ref[m_global]                          # (384, 512) bool
            mask = cv2.resize(mask_low.astype(np.uint8), (W, H),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
            gid = int(global_ids[m_global])
            if gid > 0:
                inst[mask] = palette[gid]
                n_tracked += 1
            else:
                g = skipped_grey(m_global)
                inst[mask] = (g, g, g)
                n_skipped += 1

        blend = (rgb.astype(np.float32) * 0.5
                 + inst.astype(np.float32) * 0.5).astype(np.uint8)

        triptych = np.concatenate([rgb, sep, blend, sep, inst], axis=1)

        label = (f"KF {k:03d} / {n_kf-1}  gi={gi:03d}  "
                 f"tracked={n_tracked} skipped(grey)={n_skipped}  "
                 f"global_tracks={n_global_tracks}")
        cv2.putText(triptych, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(triptych, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(cv2.cvtColor(triptych, cv2.COLOR_RGB2BGR))
        if (k + 1) % 10 == 0 or k == n_kf - 1:
            print(f"[render] kf {k+1}/{n_kf}  tracked={n_tracked} "
                  f"skipped={n_skipped}", flush=True)

    writer.release()
    print(f"[done] wrote {out_path}  ({triptych_w}x{H} @ {args.fps} fps, "
          f"{n_kf} frames)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
