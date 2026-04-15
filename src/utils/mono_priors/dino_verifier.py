"""
DINOv2-feature verifier for mask-first open-vocab classification.

The mask-first pipeline assigns each FastSAM mask a CLIP-text label. Visually
ambiguous regions (a small dry bush, a textured rock) sometimes match a
high-movability class (giraffe, person) by coincidence and slip through the
CLIP-margin gate. The fix: pool DINOv2 features (already extracted by DROID-W
during SLAM) inside each mask and use them as a second, scale-invariant
appearance signal. ConceptGraphs / OVO-SLAM use the same pattern.

Two complementary scores per track:
  • Track DINO variance — real giraffe is consistent across frames, bush
    is not (different bush patches in each keyframe).
  • Class prototype distance — distance to the average DINO embedding of
    the top-K most-CLIP-confident detections of the same class. A real
    giraffe is close to the giraffe prototype; a bush is far.

Both fold into a single `dino_trust ∈ [0, 1]` that is consumed by
`classify_tracks_logodds` as a third input to `prior_gate` alongside CLIP
margin and motion evidence.

Requires DROID-W to have run with `tracking.uncertainty_params.activate: True`
which causes `dino_feats` of shape `(N, H/14, W/14, C)` to be saved into
`video.npz` (see depth_video.py:822).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2


def pool_dino_features(
    mask: np.ndarray,
    dino_frame: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Mean-pool DINOv2 features inside a binary FastSAM mask.

    Args:
        mask:       (H, W) uint8 binary (image-resolution).
        dino_frame: (h, w, C) float32 — DINOv2 features for this keyframe
                    at feature resolution (typically H/14, W/14).

    Returns:
        L2-normalized (C,) float32 embedding, or None if the mask is empty
        at feature resolution.
    """
    if mask is None or dino_frame is None or mask.sum() == 0:
        return None
    h, w, _ = dino_frame.shape
    H, W = mask.shape
    if (h, w) != (H, W):
        m = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        m = mask.astype(np.uint8)
    if m.sum() == 0:
        return None
    feats = dino_frame[m > 0]                # (n_pixels, C)
    pooled = feats.mean(axis=0).astype(np.float32)
    n = float(np.linalg.norm(pooled))
    if n < 1e-8:
        return None
    return pooled / n


def attach_dino_to_detections(
    all_detections: List[List[Dict]],
    dino_feats: np.ndarray,
) -> None:
    """In-place: add `dino_emb` to every detection that has a `mask`.

    Args:
        all_detections: per-frame detection lists. Detections must carry
                        `mask` (image-res binary) and `frame_idx`.
        dino_feats:     (N, h, w, C) float32 — full sequence of DINO features.
    """
    if dino_feats is None:
        return
    N = len(dino_feats)
    for frame_dets in all_detections:
        for det in frame_dets:
            f_idx = int(det.get("frame_idx", -1))
            if not (0 <= f_idx < N):
                continue
            emb = pool_dino_features(det.get("mask"), dino_feats[f_idx])
            if emb is not None:
                det["dino_emb"] = emb


def compute_track_dino_consistency(
    tracks: Dict[int, Dict],
    all_detections: List[List[Dict]],
) -> Dict[int, Dict]:
    """
    For each track, compute the mean and variance of its per-frame DINO
    embeddings.

    Variance metric: 1 - mean(cos(per_frame_emb, track_mean_emb)). For a real
    object this stays low (~0.05). For a bush track that re-fires on
    different bush patches each keyframe, it climbs (~0.25-0.40).

    Returns: {track_id: {"dino_mean": (C,), "dino_var": float, "n_emb": int}}
    """
    track_embs: Dict[int, List[np.ndarray]] = {}
    for frame_dets in all_detections:
        for det in frame_dets:
            tid = det.get("track_id", -1)
            emb = det.get("dino_emb")
            if tid < 0 or emb is None:
                continue
            track_embs.setdefault(tid, []).append(emb)

    out: Dict[int, Dict] = {}
    for tid in tracks:
        embs = track_embs.get(tid, [])
        if not embs:
            out[tid] = {"dino_mean": None, "dino_var": 1.0, "n_emb": 0}
            continue
        stacked = np.stack(embs, axis=0)
        mean = stacked.mean(axis=0)
        n = float(np.linalg.norm(mean))
        mean = mean / n if n > 1e-8 else mean
        if len(embs) == 1:
            var = 0.0
        else:
            sims = stacked @ mean
            var = float(1.0 - sims.mean())
        out[tid] = {"dino_mean": mean, "dino_var": var, "n_emb": len(embs)}
    return out


def build_class_prototypes(
    tracks: Dict[int, Dict],
    track_dino: Dict[int, Dict],
    top_k: int = 5,
) -> Dict[str, np.ndarray]:
    """
    Per class, pick the K tracks with the highest mean CLIP confidence
    that have a valid DINO mean, average their DINO embeddings, L2-norm.

    Real-giraffe prototypes are distinctive; new detections of "giraffe"
    that are far from this prototype are likely mis-labeled.
    """
    by_label: Dict[str, List[Tuple[float, np.ndarray]]] = {}
    for tid, t in tracks.items():
        if track_dino.get(tid, {}).get("dino_mean") is None:
            continue
        conf = float(t.get("mean_confidence", 0.0))
        by_label.setdefault(t["label"], []).append((conf, track_dino[tid]["dino_mean"]))

    prototypes: Dict[str, np.ndarray] = {}
    for lab, items in by_label.items():
        items.sort(reverse=True, key=lambda x: x[0])
        chosen = [emb for _, emb in items[:top_k]]
        if not chosen:
            continue
        proto = np.stack(chosen, axis=0).mean(axis=0)
        n = float(np.linalg.norm(proto))
        prototypes[lab] = proto / n if n > 1e-8 else proto
    return prototypes


def compute_dino_trust(
    tracks: Dict[int, Dict],
    track_dino: Dict[int, Dict],
    prototypes: Dict[str, np.ndarray],
    var_floor: float = 0.15,
    proto_dist_floor: float = 0.40,
    proto_weight: float = 0.0,
) -> Dict[int, Dict]:
    """
    Per-track `dino_trust ∈ [0, 1]`.

    Empirically, **per-track DINO variance is the cleaner signal** than
    distance-to-class-prototype: a real giraffe has tight variance (~0.06)
    even when there are bush mis-labels in the same class, while bushes
    relabeled "giraffe" each frame are different patches with high variance
    (>0.15). The class prototype, by contrast, is unreliable when only
    a handful of tracks share the label (prototype averages bush-mis-labels
    in with the real animal) and ends up REJECTING the real giraffe whose
    appearance is far from the contaminated mean.

    Default `proto_weight = 0.0` disables the prototype contribution. Set
    it positive (~0.3) only on scenes with many tracks of each class.

    Returns: {track_id: {"dino_trust": float, "dino_proto_dist": float}}
    """
    out: Dict[int, Dict] = {}
    for tid, t in tracks.items():
        td = track_dino.get(tid, {})
        emb = td.get("dino_mean")
        var = td.get("dino_var", 1.0)
        if emb is None:
            out[tid] = {"dino_trust": 0.0, "dino_proto_dist": 1.0}
            continue
        proto = prototypes.get(t["label"])
        if proto is None:
            proto_dist = 0.0
        else:
            proto_dist = float(1.0 - np.dot(emb, proto))
        var_trust = max(0.0, 1.0 - var / max(1e-6, var_floor))
        if proto_weight > 0 and proto is not None:
            proto_trust = max(0.0, 1.0 - proto_dist / max(1e-6, proto_dist_floor))
            trust = (1.0 - proto_weight) * var_trust + proto_weight * proto_trust
        else:
            trust = var_trust
        out[tid] = {"dino_trust": float(trust), "dino_proto_dist": proto_dist}
    return out


def annotate_tracks_with_dino(
    tracks: Dict[int, Dict],
    all_detections: List[List[Dict]],
    dino_feats: np.ndarray,
    var_floor: float = 0.15,
    proto_dist_floor: float = 0.40,
    proto_top_k: int = 5,
    proto_weight: float = 0.0,
) -> None:
    """
    One-shot helper: pool features, compute per-track variance + class
    prototypes + dino_trust, and stamp the trust onto each track dict in-place
    under keys `dino_trust`, `dino_var`, `dino_proto_dist`.

    Intended to be called after `build_object_tracks` and before
    `classify_tracks_logodds`. If `dino_feats` is None (SLAM ran without
    save_feature), every track gets `dino_trust = None` and the log-odds
    fusion falls back to its previous behavior.
    """
    if dino_feats is None:
        for t in tracks.values():
            t["dino_trust"] = None
            t["dino_var"] = -1.0
            t["dino_proto_dist"] = -1.0
        return

    attach_dino_to_detections(all_detections, dino_feats)
    track_dino = compute_track_dino_consistency(tracks, all_detections)
    prototypes = build_class_prototypes(tracks, track_dino, top_k=proto_top_k)
    trust = compute_dino_trust(tracks, track_dino, prototypes,
                                var_floor=var_floor, proto_dist_floor=proto_dist_floor,
                                proto_weight=proto_weight)
    for tid, t in tracks.items():
        t["dino_trust"] = trust[tid]["dino_trust"]
        t["dino_proto_dist"] = trust[tid]["dino_proto_dist"]
        t["dino_var"] = track_dino.get(tid, {}).get("dino_var", -1.0)
        t["dino_n_emb"] = track_dino.get(tid, {}).get("n_emb", 0)
