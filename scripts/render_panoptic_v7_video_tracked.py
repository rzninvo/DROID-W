"""Re-render the B.1.5 refined CropFormer masks colored by the
B.2 GLOBAL track IDs (not per-KF random palette).

Same triptych layout as render_panoptic_v7_video.py:
  RGB | RGB+overlay (0.5 blend) | pure mask-colour
with 8-px white separators.

Stable-colour invariant: mask index i in any KF gets colour
palette[global_track_ids[mask_idx_in_flat]], so the same global track
keeps the same colour across all KFs it appears in.

Per-KF random palette (the previous video) flickers because B.1 has no
cross-KF identity. This version replaces the random palette with the
B.2 OVI-MAP §3.1 identity.
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
    p.add_argument("--min-track-kf", default=1, type=int,
                   help="hide tracks visible in fewer KFs than this")
    args = p.parse_args()

    refined = np.load(args.refined_npz, allow_pickle=True)
    tracks = np.load(args.tracks_npz, allow_pickle=True)

    masks_all = refined["masks"]              # (N, H, W) bool
    offsets = refined["seg_kf_offsets"]       # (n_kf+1,)
    kf_gi = refined["kf_global_indices"]      # (n_kf,)
    n_kf = int(refined["n_keyframes"])
    H, W = int(masks_all.shape[1]), int(masks_all.shape[2])

    global_ids = tracks["global_track_ids"]   # (N,) int (-1 = skipped)
    track_kf_counts = tracks["track_kf_counts"]  # (n_global+1,)
    n_global_tracks = int(tracks["n_global_tracks"])
    voxel_size = float(tracks["voxel_size"])
    theta_assoc = float(tracks["theta_assoc"])
    print(f"[load] {masks_all.shape[0]} masks across {n_kf} KFs; "
          f"{n_global_tracks} global tracks; "
          f"voxel_size={voxel_size} theta_assoc={theta_assoc}",
          flush=True)

    # Build stable palette indexed by GLOBAL track id.
    rng = np.random.default_rng(0)
    n_lab = int(global_ids.max()) + 1
    palette = rng.integers(64, 256, size=(n_lab, 3)).astype(np.uint8)
    palette[0] = (40, 40, 40)  # label 0 = unassigned (dark grey)

    # Per-track visibility filter: dim tracks that only appear in <K KFs.
    track_visible = (track_kf_counts >= args.min_track_kf)

    rgb_dir = Path(args.rgb_dir)
    rgb_files = sorted(f for f in rgb_dir.iterdir() if f.suffix == ".png")

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
        kf_masks = masks_all[s:e]
        kf_global_ids = global_ids[s:e]

        inst = np.zeros_like(rgb)
        kept_tracks_in_kf = set()
        for m, gid in zip(kf_masks, kf_global_ids):
            if gid <= 0:                       # skipped or unassigned
                continue
            if not track_visible[gid]:
                continue
            inst[m] = palette[gid]
            kept_tracks_in_kf.add(int(gid))
        blend = (rgb.astype(np.float32) * 0.5 + inst.astype(np.float32) * 0.5).astype(np.uint8)

        triptych = np.concatenate([rgb, sep, blend, sep, inst], axis=1)

        label = (f"KF {k:03d} / {n_kf-1}  gi={gi:03d}  "
                 f"tracks={len(kept_tracks_in_kf)}  global_total={n_global_tracks}")
        cv2.putText(triptych, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(triptych, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(cv2.cvtColor(triptych, cv2.COLOR_RGB2BGR))
        if (k + 1) % 10 == 0 or k == n_kf - 1:
            print(f"[render] kf {k+1}/{n_kf}  visible tracks={len(kept_tracks_in_kf)}",
                  flush=True)

    writer.release()
    print(f"[done] wrote {out_path}  ({triptych_w}x{H} @ {args.fps} fps, {n_kf} frames)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
