"""Compute mIoU of our open-vocab pipeline on Replica vs GT semantic labels.

Method (vertex-projection eval, comparable to OVI-MAP Table 2 metric):

  1. Load Replica mesh vertices + GT native sem labels (per-vertex).
  2. Map native → REPLICA_51 index (semantic_const.py:Replica_map_to_reduced).
  3. For each track in our pipeline, run b4_query_labelset.py logic:
     argmax over REPLICA_51 (LERF canonical-phrase relevancy).
  4. For each Replica vertex:
       project to camera frame for each KF;
       check visibility (in image bounds + depth match GT depth ±5cm);
       look up our predicted track at the pixel, get its REPLICA_51 label;
       majority vote across visible KFs;
  5. Compute per-class IoU vs GT vertex labels.
  6. Report mIoU + per-class breakdown + count of predicted classes.

OVI-MAP Table 2 reports mIoU ≈ 26.9 with SigLIP-large on Replica office0.
We're monocular DROID-W substituted with Replica GT depth/poses (so the
encoding-side comparison is fair); track-level pipeline is the only delta.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch


# OVI-MAP semantic_const.py:161
REPLICA_51 = [
    'wall', 'ceiling', 'floor', 'chair', 'blinds', 'sofa', 'table',
    'rug', 'window', 'lamp', 'door', 'pillow', 'bench', 'tv-screen',
    'cabinet', 'pillar', 'blanket', 'tv-stand', 'cushion', 'bin',
    'vent', 'bed', 'stool', 'picture', 'indoor-plant', 'desk',
    'comforter', 'nightstand', 'shelf', 'vase', 'plant-stand',
    'basket', 'plate', 'monitor', 'pipe', 'panel', 'desk-organizer',
    'wall-plug', 'book', 'box', 'clock', 'sculpture', 'tissue-paper',
    'camera', 'tablet', 'pot', 'bottle', 'candle', 'bowl', 'cloth',
    'switch',
]

# OVI-MAP semantic_const.py Replica_map_to_reduced
REPLICA_MAP_TO_REDUCED = {
    93:0, 31:1, 40:2, 20:3, 12:4, 76:5, 80:6, 98:7, 97:8, 47:9,
    37:10, 61:11, 8:12, 87:13, 18:14, 60:15, 11:16, 88:17, 29:18, 10:19,
    92:20, 7:21, 78:22, 59:23, 44:24, 34:25, 26:26, 54:27, 71:28, 91:29,
    63:30, 3:31, 64:32, 52:33, 62:34, 56:35, 35:36, 95:37, 13:38, 15:39,
    22:40, 70:41, 83:42, 17:43, 82:44, 65:45, 14:46, 19:47, 16:48, 23:49,
    79:50,
}

CANONICAL_PHRASES = ['object', 'things', 'stuff', 'texture']


def _to_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    for a in ("pooler_output", "text_embeds", "image_embeds", "last_hidden_state"):
        if hasattr(obj, a) and getattr(obj, a) is not None:
            t = getattr(obj, a)
            if t.ndim == 3: t = t.mean(dim=1)
            return t
    raise RuntimeError(f"Unknown SigLIP output: {type(obj)}")


def encode_text(model, processor, device, texts):
    with torch.no_grad():
        inp = processor(text=texts, return_tensors="pt",
                        padding="max_length", max_length=64).to(device)
        out = _to_tensor(model.get_text_features(**inp))
        out = out / (out.norm(dim=-1, keepdim=True) + 1e-9)
    return out.cpu().float().numpy()


def lerf_per_label_argmax(phi, label_emb, canon_emb, T=10.0):
    """Per-track argmax over labelset via LERF canonical-phrase relevancy."""
    s_lab = phi @ label_emb.T
    s_can = phi @ canon_emb.T
    diff = s_lab[:, :, None] - s_can[:, None, :]
    rela = (1.0 / (1.0 + np.exp(-T * diff))).min(axis=-1)
    return rela.argmax(axis=1), rela.max(axis=1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings-npz", required=True)
    p.add_argument("--refined-npz",   required=True)
    p.add_argument("--tracks-npz",    required=True)
    p.add_argument("--video-npz",     required=True)
    p.add_argument("--mesh-ply",      required=True)
    p.add_argument("--gt-sem-txt",    required=True)
    p.add_argument("--out-json",      required=True)
    p.add_argument("--siglip-model",  default="google/siglip2-large-patch16-384")
    p.add_argument("--depth-tol",     default=0.05, type=float,
                   help="metres tolerance for visibility check")
    args = p.parse_args()

    from plyfile import PlyData
    print(f"[load] mesh {args.mesh_ply}", flush=True)
    mesh = PlyData.read(args.mesh_ply)
    V = np.stack([mesh["vertex"]["x"],
                  mesh["vertex"]["y"],
                  mesh["vertex"]["z"]], axis=-1).astype(np.float32)  # (Nv, 3)
    Nv = V.shape[0]
    print(f"[load] {Nv} vertices", flush=True)

    gt_native = np.loadtxt(args.gt_sem_txt, dtype=np.int32)
    assert gt_native.shape[0] == Nv, "GT label count must match vertex count"
    gt_reduced = np.full(Nv, -1, dtype=np.int32)
    for native, reduced in REPLICA_MAP_TO_REDUCED.items():
        gt_reduced[gt_native == native] = reduced
    print(f"[load] GT classes present (REPLICA_51 reduced indices): "
          f"{np.unique(gt_reduced[gt_reduced >= 0]).tolist()}", flush=True)

    print(f"[load] embeddings {args.embeddings_npz}", flush=True)
    emb_blob = np.load(args.embeddings_npz, allow_pickle=True)
    track_emb = emb_blob["embeddings_l2"]
    valid_track_ids = np.where(np.linalg.norm(track_emb, axis=1) > 0)[0]
    valid_track_ids = valid_track_ids[valid_track_ids > 0]
    phi = track_emb[valid_track_ids]
    print(f"[setup] {len(valid_track_ids)} tracks searchable", flush=True)

    print(f"[load] {args.siglip_model}", flush=True)
    from transformers import AutoModel, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.siglip_model).to(device).eval()
    proc = AutoProcessor.from_pretrained(args.siglip_model)
    label_emb = encode_text(model, proc, device, REPLICA_51)
    canon_emb = encode_text(model, proc, device, CANONICAL_PHRASES)
    track_argmax, track_argmax_rela = lerf_per_label_argmax(phi, label_emb, canon_emb)
    track_label_by_id = {int(valid_track_ids[n]): int(track_argmax[n])
                          for n in range(len(valid_track_ids))}
    print(f"[label] per-track argmax over REPLICA_51 done", flush=True)
    # Print track-class distribution
    from collections import Counter
    pred_class_hist = Counter([REPLICA_51[track_argmax[n]]
                                for n in range(len(valid_track_ids))])
    print(f"[label] predicted-class histogram (top-15):")
    for cls, n in pred_class_hist.most_common(15):
        print(f"    {cls:20s} : {n} tracks", flush=True)

    print(f"[load] {args.refined_npz}", flush=True)
    refined = np.load(args.refined_npz, allow_pickle=True)
    masks = refined["masks"]
    offsets = refined["seg_kf_offsets"]
    kf_gi = refined["kf_global_indices"]
    n_kf = int(refined["n_keyframes"])
    H_m, W_m = int(masks.shape[1]), int(masks.shape[2])
    tracks_npz = np.load(args.tracks_npz, allow_pickle=True)
    gids = tracks_npz["global_track_ids"]
    video = np.load(args.video_npz, allow_pickle=True)
    poses = video["poses"]                                       # (n_kf, 4, 4) cam→world
    droid_up = video["droid_disps_up"]                           # (n_kf, H, W)
    intr_ba = video["intrinsics"]                                # (n_kf, 4) at /8 of full
    scale = float(video["scale"])
    Hd, Wd = int(droid_up.shape[1]), int(droid_up.shape[2])
    intrinsics_full = intr_ba * (Hd / 48.0)                      # back to full res

    print(f"[setup] {n_kf} KFs at {H_m}x{W_m}", flush=True)
    # Per-KF per-pixel predicted label map (REPLICA_51 index, -1 = unmapped)
    # We don't materialise this -- compute on the fly per vertex.

    # PASS: project each vertex into every KF, vote
    vertex_votes = np.zeros((Nv, len(REPLICA_51)), dtype=np.int32)
    t0 = time.time()
    for k in range(n_kf):
        T_cw = poses[k].astype(np.float32)
        T_wc = np.linalg.inv(T_cw)                                # world→cam
        fx, fy, cx, cy = intrinsics_full[k].astype(np.float32)
        # Project all vertices
        Vh = np.concatenate([V, np.ones((Nv, 1), dtype=np.float32)], axis=1)  # (Nv, 4)
        Pc = (T_wc @ Vh.T).T                                       # (Nv, 4) cam coords
        Z = Pc[:, 2]
        valid_z = Z > 1e-3
        u = fx * Pc[:, 0] / np.maximum(Z, 1e-3) + cx
        v = fy * Pc[:, 1] / np.maximum(Z, 1e-3) + cy
        ui = u.astype(np.int32)
        vi = v.astype(np.int32)
        in_img = (ui >= 0) & (ui < W_m) & (vi >= 0) & (vi < H_m)
        valid = valid_z & in_img

        if not valid.any():
            continue

        # Visibility: check depth match with GT depth at the projected pixel
        depth_pred = np.zeros(Nv, dtype=np.float32)
        disp_k = droid_up[k]
        # Sample disparity at projected pixel for valid vertices
        u_valid = ui[valid]; v_valid = vi[valid]
        disp_at_pix = disp_k[v_valid, u_valid]
        depth_at_pix = scale / np.maximum(disp_at_pix, 1e-3)
        z_valid = Z[valid]
        # Vertex must match observed depth (otherwise it's occluded)
        occl_ok = np.abs(z_valid - depth_at_pix) < args.depth_tol
        visible = np.where(valid)[0][occl_ok]

        if visible.size == 0:
            continue

        # Look up which track owns the pixel
        # For each visible vertex, find which track's mask covers its pixel
        s_off, e_off = int(offsets[k]), int(offsets[k + 1])
        for m_global in range(s_off, e_off):
            tid = int(gids[m_global])
            if tid <= 0 or tid not in track_label_by_id:
                continue
            mask = masks[m_global]                                # (H, W) bool
            pixel_in_mask = mask[vi[visible], ui[visible]]
            verts_in_mask = visible[pixel_in_mask]
            lbl = track_label_by_id[tid]
            np.add.at(vertex_votes, (verts_in_mask, lbl), 1)
        if (k + 1) % 20 == 0:
            print(f"[vote] kf {k+1}/{n_kf}  elapsed={time.time()-t0:.1f}s",
                  flush=True)

    # Majority vote per vertex
    has_vote = vertex_votes.sum(axis=1) > 0
    pred = np.full(Nv, -1, dtype=np.int32)
    pred[has_vote] = vertex_votes[has_vote].argmax(axis=1)
    print(f"[vote] {has_vote.sum()} / {Nv} vertices have a vote", flush=True)

    # Per-class IoU on union of GT classes present
    classes_present = sorted(np.unique(gt_reduced[gt_reduced >= 0]).tolist())
    print(f"[eval] computing IoU for {len(classes_present)} GT classes", flush=True)
    per_class = {}
    ious = []
    for c in classes_present:
        gt_c = gt_reduced == c
        pred_c = pred == c
        inter = int((gt_c & pred_c).sum())
        union = int((gt_c | pred_c).sum())
        iou = inter / union if union > 0 else 0.0
        per_class[REPLICA_51[c]] = {
            "iou": iou,
            "n_gt_vertices": int(gt_c.sum()),
            "n_pred_vertices": int(pred_c.sum()),
        }
        ious.append(iou)
    miou = float(np.mean(ious)) if ious else 0.0
    print(f"\n{'='*60}", flush=True)
    print(f"mIoU over {len(ious)} GT classes: {miou:.4f}", flush=True)
    print(f"{'='*60}", flush=True)
    for c in classes_present:
        info = per_class[REPLICA_51[c]]
        print(f"  {REPLICA_51[c]:22s}  IoU={info['iou']:.4f}  "
              f"gt={info['n_gt_vertices']:>6d}  pred={info['n_pred_vertices']:>6d}",
              flush=True)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "miou": miou,
        "n_classes": len(ious),
        "per_class": per_class,
        "n_vertices": int(Nv),
        "n_voted_vertices": int(has_vote.sum()),
        "predicted_class_hist": dict(pred_class_hist),
    }, open(out_path, "w"), indent=2)
    print(f"\n[save] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
