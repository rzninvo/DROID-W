"""
B.5 — Replica per-vertex mIoU eval (OVO-SLAM-compatible).

Predicts a class label per mesh vertex by:
  1. Encoding the 51-class Replica taxonomy via SigLIP-2 (RADSeg 80-template
     ensemble; same text-feature space as B.0.5 / B.4).
  2. Per global track: argmax cosine vs the 51 class embeddings → class id.
  3. Per mesh vertex: nearest neighbor in the union of all track point clouds
     → assign the vertex to that track's predicted class.
     Vertices farther than `--vertex-track-max-dist` from any track point
     get class id 51 (ignore / unannotated).
  4. mIoU = mean over present classes of TP / (TP + FP + FN), where the
     per-class TP/FP/FN are accumulated using `map_to_reduced` to convert
     raw class ids on both sides into the 51-class reduced index space.
  5. Optional `--ignore-background` drops {wall, floor, ceiling, door, window}
     from the mIoU (5 classes) — same flag OVO-SLAM offers.

POSE / FRAME NOTE: Replica mesh vertices are in OpenGL world-frame (Replica
native). Our track points (B.3) are also in OpenGL c2w world-frame (per
datasets.py:Replica which deliberately skips NICE-SLAM's Y/Z flip). So the
KD-tree is on directly-comparable points — no flip required.

Usage (cvg):
    python scripts/eval_panoptic_replica.py \\
        --tracks Outputs/Replica/room0/tracks.npz \\
        --volumes Outputs/Replica/room0/instance_volumes.npz \\
        --pca-basis weights/pca_basis.pt \\
        --eval-info configs/RGBD/Replica/eval_info.yaml \\
        --mesh /home/cvg/datasets/Replica/Replica/room0_mesh.ply \\
        --gt /home/cvg/datasets/Replica/semantic_gt/room0.txt \\
        --output Outputs/Replica/room0/replica_miou.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml

torch.backends.cudnn.deterministic = True

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_ply_vertices(mesh_path: Path) -> np.ndarray:
    """Read binary PLY vertex positions. Returns (V, 3) float32.
    No external deps — handles the standard Replica binary_little_endian
    PLY (vertex element with x, y, z, nx, ny, nz, red, green, blue, alpha)."""
    with open(mesh_path, "rb") as f:
        # Header is ASCII; read until 'end_header\n'.
        header = b""
        while True:
            line = f.readline()
            header += line
            if line.strip() == b"end_header":
                break
        text = header.decode("utf-8", errors="replace")
        # Find 'element vertex N'.
        n_vertex = None
        for ln in text.splitlines():
            if ln.startswith("element vertex"):
                n_vertex = int(ln.split()[2])
                break
        if n_vertex is None:
            raise RuntimeError("PLY header missing 'element vertex N'")
        # Replica room0_mesh.ply layout: 6 floats (x, y, z, nx, ny, nz) +
        # 3 bytes (uchar red, green, blue). NO alpha channel. Stride = 27.
        # (Other Replica scenes use the same layout — the original FAIR
        # release ships meshes without alpha; vMAP/NICE-SLAM didn't repackage.)
        rec_dtype = np.dtype([("xyz", np.float32, (3,)),
                              ("normal", np.float32, (3,)),
                              ("rgb", np.uint8, (3,))])
        if rec_dtype.itemsize != 27:
            raise RuntimeError(f"unexpected struct size {rec_dtype.itemsize}; "
                               f"PLY may have alpha. Re-read the header.")
        data = np.frombuffer(f.read(n_vertex * 27), dtype=rec_dtype)
        if data.shape[0] != n_vertex:
            raise RuntimeError(f"got {data.shape[0]} vertices, expected {n_vertex}")
        return data["xyz"].copy()


def _load_gt_labels(gt_path: Path) -> np.ndarray:
    """OVO-SLAM-format vertex labels. One int per line. Returns (V,) int64."""
    arr = np.loadtxt(gt_path, dtype=np.int64)
    return arr


def _load_pca_basis(path: Path):
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    return (state["mean"].numpy().astype(np.float32),
            state["components"].numpy().astype(np.float32),
            int(state["feature_dim"]),
            int(state["target_dim"]))


def _build_text_only_encoder(radio_version: str, lang_adaptor: str, device: str):
    from src.utils.mono_priors.radseg.radseg_encoder import RADSegEncoder
    is_v4 = "v4" in radio_version.lower()
    return RADSegEncoder(
        device=device, model_version=radio_version, lang_model=lang_adaptor,
        return_radio_features=True, compile=False, amp=False, predict=False,
        sam_refinement=False, sam3=is_v4, sam_ckpt="",
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", required=True, type=str)
    p.add_argument("--volumes", required=True, type=str)
    p.add_argument("--pca-basis", default="weights/pca_basis.pt", type=str)
    p.add_argument("--eval-info", required=True, type=str)
    p.add_argument("--mesh", required=True, type=str)
    p.add_argument("--gt", required=True, type=str)
    p.add_argument("--output", default=None, type=str)
    p.add_argument("--vertex-track-max-dist", default=0.20, type=float,
                   help="Max distance (m) from a vertex to its nearest track "
                        "point. Beyond this, vertex → ignore.")
    p.add_argument("--ignore-background", action="store_true",
                   help="Drop {wall, floor, ceiling, door, window} from mIoU "
                        "(matches OVO-SLAM's --ignore-background flag).")
    p.add_argument("--radio-version", default="c-radio_v4-h", type=str)
    p.add_argument("--device", default="cuda:0", type=str)
    args = p.parse_args()

    trk_path = Path(args.tracks);    vol_path = Path(args.volumes)
    bas_path = Path(args.pca_basis); inf_path = Path(args.eval_info)
    msh_path = Path(args.mesh);      gt_path = Path(args.gt)
    if not trk_path.is_absolute(): trk_path = REPO_ROOT / trk_path
    if not vol_path.is_absolute(): vol_path = REPO_ROOT / vol_path
    if not bas_path.is_absolute(): bas_path = REPO_ROOT / bas_path
    if not inf_path.is_absolute(): inf_path = REPO_ROOT / inf_path
    if not msh_path.is_absolute(): msh_path = REPO_ROOT / msh_path
    if not gt_path.is_absolute(): gt_path = REPO_ROOT / gt_path
    out_path = Path(args.output) if args.output else trk_path.parent / "replica_miou.json"
    if not out_path.is_absolute(): out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[setup] tracks={trk_path}", flush=True)
    print(f"[setup] mesh={msh_path}\n[setup] gt={gt_path}", flush=True)

    # ── Load eval taxonomy ─────────────────────────────────────────────────
    info = yaml.safe_load(inf_path.read_text())
    valid_ids: List[int] = info["valid_class_ids"]
    num_cls: int = info["num_classes"]            # 51
    map_to_reduced: Dict[int, int] = {int(k): int(v) for k, v in info["map_to_reduced"].items()}
    ignore_list: List[int] = info.get("ignore", [num_cls]) or [num_cls]
    class_names: List[str] = info["class_names_reduced"]
    bg_ids: List[int] = info.get("background_reduced_ids", [])
    print(f"[taxonomy] num_classes={num_cls}  ignore_index={ignore_list}", flush=True)

    # ── Load tracks + decode embeddings ────────────────────────────────────
    trk = np.load(trk_path)
    track_emb_pca = trk["track_emb_pca"].astype(np.float32)
    G = int(trk["n_tracks_total"])
    pca_mean, pca_components, D, target_dim = _load_pca_basis(bas_path)
    decoded = track_emb_pca @ pca_components + pca_mean             # (G, D)
    decoded = decoded / (np.linalg.norm(decoded, axis=1, keepdims=True) + 1e-8)
    print(f"[tracks] G={G}, decoded shape={decoded.shape}", flush=True)

    # ── Encode 51-class names via SigLIP-2 ─────────────────────────────────
    is_v4 = "v4" in args.radio_version.lower()
    lang_adaptor = "siglip2-g" if is_v4 else "siglip2"
    t0 = time.time()
    encoder = _build_text_only_encoder(args.radio_version, lang_adaptor, args.device)
    with torch.no_grad():
        cls_emb = encoder.encode_labels(class_names, onehot=False)   # (51, D)
    cls_emb = cls_emb.cpu().numpy().astype(np.float32)
    cls_emb = cls_emb / (np.linalg.norm(cls_emb, axis=1, keepdims=True) + 1e-8)
    print(f"[text] {len(class_names)} class embeddings in {time.time()-t0:.1f}s", flush=True)
    del encoder
    torch.cuda.empty_cache()

    # ── Per-track predicted class (argmax over reduced classes) ────────────
    cos_track_class = decoded @ cls_emb.T                            # (G, num_cls)
    track_pred_reduced = cos_track_class.argmax(axis=1)              # (G,) reduced ids ∈ [0, num_cls)
    print(f"[predict] track class assignment done", flush=True)

    # ── Vertex → nearest track-point → track id ────────────────────────────
    vol = np.load(vol_path)
    points = vol["points"].astype(np.float32)                        # (P, 3)
    track_offsets = vol["track_offsets"].astype(np.int64)
    P = points.shape[0]
    print(f"[volumes] P_total={P}", flush=True)

    vertices = _load_ply_vertices(msh_path)                          # (V, 3)
    V = vertices.shape[0]
    gt = _load_gt_labels(gt_path)                                    # (V,)
    if gt.shape[0] != V:
        print(f"[WARN] GT labels n={gt.shape[0]} != mesh vertices n={V}; truncating", flush=True)
        n = min(gt.shape[0], V)
        gt = gt[:n]; vertices = vertices[:n]; V = n
    print(f"[mesh] V={V}", flush=True)

    # KD-tree.
    print(f"[knn] building KD-tree over {P} track points ...", flush=True)
    t0 = time.time()
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    print(f"[knn] built in {time.time()-t0:.1f}s", flush=True)

    # Map points → tracks via track_offsets.
    point_track_id = np.zeros(P, dtype=np.int32)
    for g in range(G):
        s, e = int(track_offsets[g]), int(track_offsets[g + 1])
        point_track_id[s:e] = g

    # Replica meshes contain FLT_MAX sentinels (~3.4e38) for invalid /
    # unfilled vertices. np.isfinite() doesn't catch these — they're not
    # NaN/Inf, just impossibly large. Filter on absolute magnitude.
    # Indoor rooms fit in <100 m bbox; FLT_MAX is 3.4e38.
    SENTINEL_MAG = 1e6
    finite_v = (np.isfinite(vertices).all(axis=1)
                & (np.abs(vertices) < SENTINEL_MAG).all(axis=1))
    n_finite = int(finite_v.sum())
    print(f"[mesh] {n_finite}/{V} vertices valid "
          f"(rest = NaN/Inf or |coord| > {SENTINEL_MAG:.0e} sentinel → ignore)",
          flush=True)

    # Query nearest point per vertex (only finite ones).
    # workers=-1 parallelizes across CPU cores; default single-thread is
    # ~10x slower on 871k×2.66M cKDTree query. Pin a chunk size so progress
    # is visible.
    dist = np.full(V, np.inf, dtype=np.float64)
    nn_idx = np.zeros(V, dtype=np.int64)
    if n_finite > 0:
        finite_idx = np.where(finite_v)[0]
        chunk = 100_000
        t_query = time.time()
        for ci in range(0, n_finite, chunk):
            sl = finite_idx[ci: ci + chunk]
            d_f, n_f = tree.query(vertices[sl], k=1, workers=-1)
            dist[sl] = d_f
            nn_idx[sl] = n_f
            done = min(ci + chunk, n_finite)
            print(f"  knn {done}/{n_finite} ({time.time()-t_query:.1f}s)", flush=True)
    vertex_track_id = point_track_id[nn_idx]                          # (V,)
    too_far = (dist > args.vertex_track_max_dist) | (~finite_v)
    print(f"[knn] median dist (finite) {np.median(dist[finite_v]):.3f} m, {int(too_far.sum())}/{V} "
          f"vertices > {args.vertex_track_max_dist} m or non-finite → ignore", flush=True)

    # Predicted reduced class per vertex.
    vertex_pred = np.full(V, num_cls, dtype=np.int64)                 # default = ignore
    valid_v = ~too_far
    vertex_pred[valid_v] = track_pred_reduced[vertex_track_id[valid_v]]

    # ── GT raw → reduced ───────────────────────────────────────────────────
    # map_to_reduced maps every class id (incl -2, 256) to a reduced index;
    # any unmapped class -> num_cls (ignore).
    gt_reduced = np.full_like(gt, num_cls)
    for raw, red in map_to_reduced.items():
        gt_reduced[gt == raw] = red
    print(f"[gt] mapped {(gt_reduced != num_cls).sum()}/{V} vertices to {num_cls} valid classes",
          flush=True)

    # ── mIoU ───────────────────────────────────────────────────────────────
    # Confusion matrix — but for memory-efficient large V, just per-class TP/FP/FN.
    classes_to_eval = list(range(num_cls))
    if args.ignore_background:
        classes_to_eval = [c for c in classes_to_eval if c not in bg_ids]
    iou_per_class = np.zeros(len(classes_to_eval), dtype=np.float64)
    present = np.zeros(len(classes_to_eval), dtype=bool)
    for k_idx, k in enumerate(classes_to_eval):
        tp = int(((vertex_pred == k) & (gt_reduced == k)).sum())
        fp = int(((vertex_pred == k) & (gt_reduced != k) & (gt_reduced != num_cls)).sum())
        fn = int(((vertex_pred != k) & (gt_reduced == k)).sum())
        denom = tp + fp + fn
        if denom > 0:
            iou_per_class[k_idx] = tp / denom
            present[k_idx] = True

    miou = float(iou_per_class[present].mean()) if present.any() else 0.0
    pa = float(((vertex_pred == gt_reduced) & (gt_reduced != num_cls)).sum()
               / max(1, (gt_reduced != num_cls).sum()))
    print(f"\n=== Replica room0 results ===", flush=True)
    print(f"  num_classes={num_cls}  classes_present={int(present.sum())}/{len(classes_to_eval)}",
          flush=True)
    print(f"  mIoU      = {miou*100:.2f}%   (mean over present classes)", flush=True)
    print(f"  PixelAcc  = {pa*100:.2f}%     (vertex-level accuracy)", flush=True)
    print(f"  background_excluded={args.ignore_background}", flush=True)

    # Per-class breakdown (top + bottom).
    print(f"\n  per-class IoU (sorted desc):", flush=True)
    pairs = sorted([(iou_per_class[i], classes_to_eval[i]) for i in range(len(classes_to_eval))
                    if present[i]], reverse=True)
    for iou, k in pairs[:10]:
        print(f"    {class_names[k]:<20s} IoU = {iou*100:.2f}%", flush=True)
    if len(pairs) > 10:
        print(f"    ... ({len(pairs)-20} omitted) ...", flush=True)
        for iou, k in pairs[-10:]:
            print(f"    {class_names[k]:<20s} IoU = {iou*100:.2f}%", flush=True)

    out = {
        "scene": str(trk_path.parent.name),
        "miou": miou,
        "pixel_accuracy": pa,
        "n_classes_present": int(present.sum()),
        "n_classes_evaluated": len(classes_to_eval),
        "ignore_background": bool(args.ignore_background),
        "vertex_track_max_dist_m": float(args.vertex_track_max_dist),
        "n_vertices": int(V),
        "n_track_points": int(P),
        "n_tracks": int(G),
        "per_class_iou": {class_names[classes_to_eval[i]]: float(iou_per_class[i])
                          for i in range(len(classes_to_eval)) if present[i]},
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[save] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
