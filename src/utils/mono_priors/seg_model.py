"""
Object detection and tracking module for scene graph construction.

Runs on DROID-W keyframes and produces per-frame detections with persistent
object IDs across frames. Static/dynamic classification is done exclusively
via DROID-W's uncertainty map — no hardcoded class-based separation.

Data flow:
    DROID-W saves keyframes (RGB, pose, depth, FiT3D features, uncertainty)
    → This module detects and tracks all objects via YOLO-World + BoT-SORT
    → classify_detections_by_uncertainty() tags each detection using the
      uncertainty map (75th percentile inside each box)
    → Scene graph module consumes labeled, tracked, tagged detections
"""

from typing import Dict, List, Optional
import os
import json

import numpy as np
import cv2
import torch


# Comprehensive prompt list for YOLO-World open-vocabulary detection.
# NOT split into static/dynamic — DROID-W's uncertainty map handles that.
DEFAULT_CLASSES = [
    # People
    "person",
    # Vehicles
    "car", "bus", "truck", "motorcycle", "bicycle", "scooter",
    "van", "taxi", "ambulance", "train",
    # Animals
    "dog", "cat", "bird", "horse", "cow", "sheep",
    # Personal transport & carried objects
    "skateboard", "stroller", "wheelchair", "shopping cart",
    "umbrella", "bag", "backpack", "suitcase",
    # Indoor furniture & objects
    "chair", "office chair", "table", "desk", "couch", "bed",
    "door", "cabinet", "drawer", "shelf", "bookcase",
    # Indoor small objects
    "laptop", "monitor", "phone", "keyboard", "mouse",
    "cup", "bottle", "plate", "bowl", "book",
    "lamp", "clock", "vase", "plant", "potted plant",
    # Outdoor objects
    "bench", "trash can", "fire hydrant", "mailbox",
    "traffic light", "stop sign", "parking meter",
    "flag", "sign", "pole", "cone",
    # Appliances & misc — note: "tv" intentionally omitted because YOLO-World
    # confuses computer monitors with TVs in indoor scenes. The VLM scout
    # will add "tv"/"television" if a real one is present.
    "microwave", "oven", "refrigerator", "sink",
    "toilet", "fan", "robot vacuum", "toy", "ball",
]


def _is_yoloe(model_name: str) -> bool:
    """Detect a YOLOE checkpoint by name (yoloe-v8s-seg.pt, yoloe-11l-seg.pt, ...)."""
    return "yoloe" in str(model_name).lower()


def _is_yoloe_prompt_free(model_name: str) -> bool:
    """YOLOE prompt-free variants (…-seg-pf.pt) have a fixed 4585-class vocab
    and do NOT accept set_classes — skipping it prevents a crash."""
    n = str(model_name).lower()
    return "yoloe" in n and "pf" in n


def _set_classes_unified(model, classes: List[str], is_yoloe: bool, is_prompt_free: bool = False):
    """
    Apply class vocabulary using whichever API the loaded model expects.

    YOLO-World:        model.set_classes(classes)
    YOLOE (prompted):  model.set_classes(classes, model.get_text_pe(classes))
    YOLOE (-pf):       no-op — the prompt-free variant has a fixed vocabulary

    YOLOE (prompted) requires the text-prompt embeddings explicitly because
    RepRTA folds them into the model weights at inference time (zero-overhead
    detection).
    """
    if is_prompt_free:
        return
    if not isinstance(classes, list) or len(classes) == 0:
        return
    if is_yoloe:
        text_pe = model.get_text_pe(classes)
        model.set_classes(classes, text_pe)
    else:
        model.set_classes(classes)


def get_detector(
    model_name: str = "yolov8s-worldv2.pt",
    classes: List[str] = None,
    device: str = "cuda:0",
    output_dir: str = None,
):
    """
    Load an open-vocab detection model and set its vocabulary.

    Supports two backends, selected by the model_name string:
      - "yolov8s-worldv2.pt" (YOLO-World v2) — boxes only; pair with FastSAM
      - "yoloe-v8s-seg.pt"   (YOLOE, ICCV 2025) — boxes + per-instance masks
                                                  in one forward pass

    Class priority:
        1. Explicit `classes` argument (if provided)
        2. VLM-discovered classes from `output_dir/vlm_classes.json` (if exists)
        3. DEFAULT_CLASSES fallback (only when VLM is not used)

    Returns:
        Loaded model with vocabulary set. Has an attribute `_is_yoloe` (bool)
        to let downstream callers know which backend they're talking to.
    """
    is_yoloe = _is_yoloe(model_name)
    is_prompt_free = _is_yoloe_prompt_free(model_name)
    if is_yoloe:
        from ultralytics import YOLOE
        model = YOLOE(model_name)
    else:
        from ultralytics import YOLO
        model = YOLO(model_name)
    model.to(device)
    model._is_yoloe = is_yoloe  # tag the instance so callers don't need to re-check the name
    model._is_prompt_free = is_prompt_free

    # Resolve class list: explicit > VLM-discovered > fallback
    if classes is None and output_dir is not None:
        from src.utils.mono_priors.vlm_scene_scout import VLMSceneScout
        vlm_classes = VLMSceneScout.load_classes(output_dir)
        if vlm_classes is not None:
            classes = vlm_classes

    if classes is None:
        classes = DEFAULT_CLASSES

    _set_classes_unified(model, classes, is_yoloe, is_prompt_free)
    return model


def update_detector_classes(model, classes: List[str]):
    """
    Update the detector's vocabulary in-place.

    Called when the VLM scene scout discovers new object types. Re-encodes
    text prompts into CLIP embeddings, so only call when the list actually
    changes. No-op for YOLOE prompt-free models (fixed vocab).
    """
    is_yoloe = bool(getattr(model, "_is_yoloe", False))
    is_prompt_free = bool(getattr(model, "_is_prompt_free", False))
    _set_classes_unified(model, classes, is_yoloe, is_prompt_free)


def _extract_masks_at_image_res(result, H: int, W: int) -> Optional[np.ndarray]:
    """
    Pull per-instance binary masks from an Ultralytics result, reshaped to
    (N, H, W) uint8 in {0, 1}. Returns None if the result has no masks.

    Handles both `retina_masks=True` (full-res) and the default down-sampled
    case by resizing per-mask via nearest-neighbor.
    """
    if getattr(result, "masks", None) is None or result.masks is None:
        return None
    md = result.masks.data
    if md is None or len(md) == 0:
        return None
    arr = md.detach().cpu().numpy()
    if arr.ndim == 2:
        arr = arr[None]
    arr = (arr > 0.5).astype(np.uint8)
    if arr.shape[1] != H or arr.shape[2] != W:
        out = np.zeros((arr.shape[0], H, W), dtype=np.uint8)
        for i in range(arr.shape[0]):
            out[i] = cv2.resize(arr[i], (W, H), interpolation=cv2.INTER_NEAREST)
        arr = out
    return arr


@torch.no_grad()
def detect_objects(
    model,
    image_np: np.ndarray,
    conf_thresh: float = 0.15,
) -> List[Dict]:
    """
    Run open-vocabulary detection on a single frame (no tracking).

    For YOLOE-seg models, also extracts a per-instance pixel mask and attaches
    it to each detection under the 'mask' key. For YOLO-World, the 'mask'
    field is omitted (callers can fill it via FastSAM).

    Returns:
        List of detections, each a dict with:
            box, label, confidence, class_id, img_h, img_w, [mask]
    """
    H, W = image_np.shape[:2]
    is_yoloe = bool(getattr(model, "_is_yoloe", False))
    predict_kwargs = dict(conf=conf_thresh, imgsz=1280, verbose=False)
    if is_yoloe:
        predict_kwargs["retina_masks"] = True
    results = model.predict(image_np, **predict_kwargs)

    detections = []
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            masks = _extract_masks_at_image_res(result, H, W) if is_yoloe else None

            for i in range(len(classes)):
                label = result.names.get(classes[i], str(classes[i])) if result.names else str(classes[i])
                det = {
                    "box": boxes[i].tolist(),
                    "label": label,
                    "confidence": float(confs[i]),
                    "class_id": int(classes[i]),
                    "img_h": H,
                    "img_w": W,
                }
                if masks is not None and i < len(masks):
                    det["mask"] = masks[i]
                detections.append(det)

    return detections


@torch.no_grad()
def track_objects(
    model,
    image_np: np.ndarray,
    frame_idx: int,
    conf_thresh: float = 0.15,
    tracker: str = "botsort.yaml",
) -> List[Dict]:
    """
    Run open-vocabulary detection with persistent object tracking.

    Uses BoT-SORT (default) or ByteTrack to maintain consistent object IDs
    across sequential frames. Must be called on frames in order — the tracker
    maintains internal state between calls via persist=True.

    Note: DROID-W keyframes are not consecutive video frames (gaps of 5-30
    frames due to motion-based keyframe selection). BoT-SORT handles this
    reasonably well due to its re-identification features, but expect some
    ID switches on objects that move significantly between keyframes.

    Args:
        model: YOLO-World model from get_detector().
        image_np: uint8 numpy array (H, W, 3) in RGB.
        frame_idx: Keyframe index (used for first_seen/last_seen metadata).
        conf_thresh: Minimum detection confidence.
        tracker: Tracker config ("botsort.yaml" or "bytetrack.yaml").

    Returns:
        List of detections, each a dict with:
            box: [x1, y1, x2, y2] in pixel coords
            label: str class name
            confidence: float
            class_id: int (model-internal)
            track_id: int — persistent ID across frames (-1 if tracking lost)
            frame_idx: int — which keyframe this detection is from
            img_h: int
            img_w: int
    """
    H, W = image_np.shape[:2]
    is_yoloe = bool(getattr(model, "_is_yoloe", False))
    track_kwargs = dict(conf=conf_thresh, imgsz=1280, persist=True, tracker=tracker, verbose=False)
    if is_yoloe:
        track_kwargs["retina_masks"] = True
    results = model.track(image_np, **track_kwargs)

    detections = []
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            masks = _extract_masks_at_image_res(result, H, W) if is_yoloe else None

            # track IDs may not exist if tracker loses the object
            if result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = np.full(len(classes), -1, dtype=int)

            for i in range(len(classes)):
                label = result.names.get(classes[i], str(classes[i])) if result.names else str(classes[i])
                det = {
                    "box": boxes[i].tolist(),
                    "label": label,
                    "confidence": float(confs[i]),
                    "class_id": int(classes[i]),
                    "track_id": int(track_ids[i]),
                    "frame_idx": frame_idx,
                    "img_h": H,
                    "img_w": W,
                }
                if masks is not None and i < len(masks):
                    det["mask"] = masks[i]
                detections.append(det)

    return detections


def build_object_tracks(all_detections: List[List[Dict]]) -> Dict[int, Dict]:
    """
    Aggregate per-frame tracked detections into object-level summaries.

    Args:
        all_detections: List of per-frame detection lists from track_objects().

    Returns:
        Dict mapping track_id → object summary:
            track_id: int
            label: str (most frequent label for this track)
            first_seen: int (first keyframe index)
            last_seen: int (last keyframe index)
            num_frames: int (how many keyframes this object appears in)
            boxes: list of [frame_idx, [x1,y1,x2,y2]] pairs (trajectory)
            mean_confidence: float
    """
    tracks = {}

    for frame_dets in all_detections:
        for det in frame_dets:
            tid = det.get("track_id", -1)
            if tid < 0:
                continue

            if tid not in tracks:
                tracks[tid] = {
                    "track_id": tid,
                    "labels": [],
                    "first_seen": det["frame_idx"],
                    "last_seen": det["frame_idx"],
                    "num_frames": 0,
                    "boxes": [],
                    "confidences": [],
                }

            t = tracks[tid]
            t["labels"].append(det["label"])
            t["last_seen"] = max(t["last_seen"], det["frame_idx"])
            t["first_seen"] = min(t["first_seen"], det["frame_idx"])
            t["num_frames"] += 1
            t["boxes"].append([det["frame_idx"], det["box"]])
            t["confidences"].append(det["confidence"])

    # Finalize: pick most frequent label, compute mean confidence
    for tid, t in tracks.items():
        from collections import Counter
        t["label"] = Counter(t["labels"]).most_common(1)[0][0]
        t["mean_confidence"] = float(np.mean(t["confidences"]))
        del t["labels"], t["confidences"]

    return tracks


def save_detections(detections: List[Dict], output_dir: str, idx: int):
    """Save detections to JSON for scene graph construction."""
    det_dir = os.path.join(output_dir, "detections")
    os.makedirs(det_dir, exist_ok=True)
    with open(os.path.join(det_dir, f"{idx:05d}.json"), "w") as f:
        json.dump(detections, f)


def load_detections(output_dir: str, idx: int) -> List[Dict]:
    """Load saved detections from JSON."""
    det_path = os.path.join(output_dir, "detections", f"{idx:05d}.json")
    with open(det_path) as f:
        return json.load(f)


def save_tracks(tracks: Dict[int, Dict], output_dir: str):
    """Save object tracks summary to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "tracks.json"), "w") as f:
        json.dump(tracks, f, indent=2)


def load_tracks(output_dir: str) -> Dict[int, Dict]:
    """Load object tracks summary from JSON."""
    with open(os.path.join(output_dir, "tracks.json")) as f:
        return {int(k): v for k, v in json.load(f).items()}


def pool_features_in_box(
    dino_feats: np.ndarray,
    box: List[float],
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """
    Average-pool DINO/FiT3D features inside a bounding box.

    Args:
        dino_feats: Feature map of shape (fh, fw, C) where fh = H//14, fw = W//14.
        box: [x1, y1, x2, y2] in image pixel coordinates.
        img_h: Image height the box was detected in.
        img_w: Image width the box was detected in.

    Returns:
        1D feature vector of shape (C,), L2-normalized. Returns zeros if box
        maps to empty region.
    """
    fh, fw, C = dino_feats.shape
    x1, y1, x2, y2 = box

    # Scale box to feature map resolution
    fx1 = max(0, int(x1 * fw / img_w))
    fy1 = max(0, int(y1 * fh / img_h))
    fx2 = min(fw, int(x2 * fw / img_w))
    fy2 = min(fh, int(y2 * fh / img_h))

    if fx2 <= fx1 or fy2 <= fy1:
        return np.zeros(C, dtype=np.float32)

    region = dino_feats[fy1:fy2, fx1:fx2, :]  # (rh, rw, C)
    embedding = region.reshape(-1, C).mean(axis=0)  # (C,)

    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 1e-6:
        embedding = embedding / norm

    return embedding


def merge_fragmented_tracks(
    tracks: Dict[int, Dict],
    all_detections: List[List[Dict]],
    dino_feats_all: np.ndarray,
    images_shape: tuple,
    cosine_thresh: float = 0.7,
    max_gap: int = 30,
) -> tuple:
    """
    Merge fragmented tracks using FiT3D/DINO feature similarity.

    When BoT-SORT loses an object and re-detects it with a new ID, this
    function finds those fragments and unifies them under one ID.

    Args:
        tracks: Output of build_object_tracks().
        all_detections: Per-frame detection lists from track_objects().
        dino_feats_all: DINO features array of shape (N_keyframes, fh, fw, C).
        images_shape: (N, C, H, W) — needed to know image dimensions.
        cosine_thresh: Minimum cosine similarity to merge two tracks.
        max_gap: Maximum keyframe gap between tracks to consider merging.

    Returns:
        (merged_tracks, merged_detections) with reassigned track_ids.
    """
    if dino_feats_all is None or len(tracks) < 2:
        return tracks, all_detections

    _, _, img_h, img_w = images_shape

    # Step 1: Compute per-track embedding by averaging pooled features
    track_embeddings = {}
    for tid, t in tracks.items():
        embeddings = []
        for frame_idx, box in t["boxes"]:
            if frame_idx < len(dino_feats_all):
                feat = dino_feats_all[frame_idx]  # (fh, fw, C)
                emb = pool_features_in_box(feat, box, img_h, img_w)
                if np.linalg.norm(emb) > 1e-6:
                    embeddings.append(emb)
        if embeddings:
            avg = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(avg)
            track_embeddings[tid] = avg / norm if norm > 1e-6 else avg
        else:
            track_embeddings[tid] = None

    # Step 2: Find merge candidates — same label, non-overlapping time, similar features
    merge_map = {}  # old_tid → new_tid
    tids = sorted(tracks.keys())

    for i, tid_a in enumerate(tids):
        if tid_a in merge_map:
            continue
        ta = tracks[tid_a]
        emb_a = track_embeddings.get(tid_a)
        if emb_a is None:
            continue

        for tid_b in tids[i + 1:]:
            if tid_b in merge_map:
                continue
            tb = tracks[tid_b]
            emb_b = track_embeddings.get(tid_b)
            if emb_b is None:
                continue

            # Must have same label
            if ta["label"] != tb["label"]:
                continue

            # Must not overlap in time (allow small overlap of 1 frame)
            if ta["last_seen"] >= tb["first_seen"] - 1 and tb["last_seen"] >= ta["first_seen"] - 1:
                # Check if they truly overlap (both active at the same frame)
                a_frames = {f for f, _ in ta["boxes"]}
                b_frames = {f for f, _ in tb["boxes"]}
                if a_frames & b_frames:
                    continue

            # Gap between tracks must be reasonable
            gap = min(
                abs(tb["first_seen"] - ta["last_seen"]),
                abs(ta["first_seen"] - tb["last_seen"]),
            )
            if gap > max_gap:
                continue

            # Cosine similarity check
            sim = float(np.dot(emb_a, emb_b))
            if sim >= cosine_thresh:
                merge_map[tid_b] = tid_a

    if not merge_map:
        return tracks, all_detections

    # Step 3: Reassign track_ids in all detections
    merged_detections = []
    for frame_dets in all_detections:
        new_frame = []
        for det in frame_dets:
            det = dict(det)  # copy
            tid = det.get("track_id", -1)
            det["track_id"] = merge_map.get(tid, tid)
            new_frame.append(det)
        merged_detections.append(new_frame)

    # Step 4: Rebuild tracks from merged detections
    merged_tracks = build_object_tracks(merged_detections)

    return merged_tracks, merged_detections


def compute_adaptive_threshold(
    all_uncertainties: np.ndarray,
    k: float = 1.5,
) -> float:
    """
    Compute a scene-adaptive threshold for static/dynamic classification.

    Uses the median + k * MAD (median absolute deviation) of the global
    uncertainty distribution. This adapts to scenes with different overall
    uncertainty levels — a quiet indoor scene gets a lower threshold than
    a busy street scene.

    Args:
        all_uncertainties: Uncertainty maps, shape (N, h, w).
        k: Multiplier for MAD. Higher = fewer false dynamic. Default 1.5.

    Returns:
        Adaptive threshold value.
    """
    flat = all_uncertainties.flatten()
    median = np.median(flat)
    mad = np.median(np.abs(flat - median))
    threshold = median + k * mad
    return float(threshold)


def classify_detections_by_uncertainty(
    detections: List[Dict],
    uncertainty_map: torch.Tensor,
    threshold: float = 0.6,
    percentile: float = 75.0,
    erode_mask_px: int = 2,
) -> List[Dict]:
    """
    Tag each detection as static or dynamic using DROID-W's uncertainty map.

    If a detection has a 'mask' field (from FastSAM), uncertainty is sampled
    inside the eroded mask — the erosion (NID-SLAM style) removes unreliable
    boundary pixels. Falls back to box sampling if no mask.

    Args:
        detections: List of detection dicts. Each may carry an optional
                    'mask': np.uint8 (img_h, img_w) from FastSAM.
        uncertainty_map: DROID-W uncertainty tensor at any resolution.
        threshold: Score above this → dynamic.
        percentile: Which percentile to use (default 75th).
        erode_mask_px: Pixels to erode off mask boundary before sampling.

    Returns:
        Same detections with added 'is_dynamic' and 'dynamic_confidence' fields.
    """
    u_h, u_w = uncertainty_map.shape[-2], uncertainty_map.shape[-1]
    u_np = uncertainty_map.detach().cpu().numpy() if isinstance(uncertainty_map, torch.Tensor) else uncertainty_map

    for det in detections:
        img_h, img_w = det["img_h"], det["img_w"]
        score = 0.0
        used_mask = False

        mask = det.get("mask")
        if mask is not None and mask.any():
            m_small = cv2.resize(mask.astype(np.uint8), (u_w, u_h), interpolation=cv2.INTER_NEAREST)
            if erode_mask_px > 0:
                k = 2 * erode_mask_px + 1
                m_eroded = cv2.erode(m_small, np.ones((k, k), np.uint8), iterations=1)
                if m_eroded.sum() >= 20:
                    m_small = m_eroded
            vals = u_np[m_small == 1]
            if vals.size >= 20:
                score = float(np.quantile(vals, percentile / 100.0))
                used_mask = True

        if not used_mask:
            x1, y1, x2, y2 = det["box"]
            bx1 = max(0, int(x1 * u_w / img_w))
            by1 = max(0, int(y1 * u_h / img_h))
            bx2 = min(u_w, int(x2 * u_w / img_w))
            by2 = min(u_h, int(y2 * u_h / img_h))
            if bx2 > bx1 and by2 > by1:
                region = uncertainty_map[by1:by2, bx1:bx2].flatten()
                score = torch.quantile(region.float(), percentile / 100.0).item()

        det["dynamic_confidence"] = score
        det["is_dynamic"] = score > threshold

    return detections


def classify_tracks_by_uncertainty(
    tracks: Dict[int, Dict],
    all_detections: List[List[Dict]],
) -> Dict[int, Dict]:
    """
    Aggregate per-frame is_dynamic tags into a track-level classification.

    An object is considered dynamic if it was tagged dynamic in >50% of its
    frames. This smooths out per-frame noise from the uncertainty map.

    Args:
        tracks: Output of build_object_tracks().
        all_detections: Per-frame detections (must already have is_dynamic from
                        classify_detections_by_uncertainty).

    Returns:
        Same tracks dict with added 'is_dynamic', 'dynamic_ratio', and
        'mean_dynamic_confidence' fields.
    """
    # Collect per-track dynamic stats
    track_stats = {}
    for frame_dets in all_detections:
        for det in frame_dets:
            tid = det.get("track_id", -1)
            if tid < 0 or "is_dynamic" not in det:
                continue
            if tid not in track_stats:
                track_stats[tid] = {"dynamic_count": 0, "total": 0, "scores": []}
            track_stats[tid]["total"] += 1
            track_stats[tid]["scores"].append(det.get("dynamic_confidence", 0.0))
            if det["is_dynamic"]:
                track_stats[tid]["dynamic_count"] += 1

    for tid, t in tracks.items():
        if tid in track_stats:
            stats = track_stats[tid]
            t["dynamic_ratio"] = stats["dynamic_count"] / max(1, stats["total"])
            t["mean_dynamic_confidence"] = float(np.mean(stats["scores"]))
            t["is_dynamic"] = t["dynamic_ratio"] > 0.5
        else:
            t["dynamic_ratio"] = 0.0
            t["mean_dynamic_confidence"] = 0.0
            t["is_dynamic"] = False

    return tracks


def _pose_to_Rt(pose_7: np.ndarray) -> np.ndarray:
    """
    Convert a DROID-SLAM pose (tx,ty,tz,qx,qy,qz,qw) to a 4x4 SE(3) matrix.

    NOTE on convention: DROID-W's `tum_poses` in video.npz are stored as
    **world-to-camera** (T_wc). This function just materializes the raw SE(3);
    callers must invert to get camera-to-world if they need it.
    See depth_video.py:56 and factor_graph.py for confirmation.
    """
    tx, ty, tz, qx, qy, qz, qw = pose_7
    n = qx*qx + qy*qy + qz*qz + qw*qw
    if n < 1e-12:
        R = np.eye(3)
    else:
        s = 2.0 / n
        R = np.array([
            [1 - s*(qy*qy+qz*qz),     s*(qx*qy-qz*qw),     s*(qx*qz+qy*qw)],
            [    s*(qx*qy+qz*qw), 1 - s*(qx*qx+qz*qz),     s*(qy*qz-qx*qw)],
            [    s*(qx*qz-qy*qw),     s*(qy*qz+qx*qw), 1 - s*(qx*qx+qy*qy)],
        ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def compute_reprojection_consistency(
    tracks: Dict[int, Dict],
    all_detections: List[List[Dict]],
    poses: np.ndarray,
    depths: np.ndarray,
    intrinsics: np.ndarray,
    sample_points: int = 100,
    hit_threshold_px: int = 15,
) -> Dict[int, float]:
    """
    Per-track mean reprojection hit-rate ∈ [0, 1]. Near 1.0 = static.

    Samples ~`sample_points` mask-interior pixels in an anchor keyframe,
    back-projects with anchor depth + pose, reprojects into every other
    keyframe of the track, and counts hits inside that frame's mask (or
    within `hit_threshold_px` of the mask). If an object actually moved in
    3D between keyframes, the reprojections fall off the moved mask → low
    hit-rate → dynamic.

    Inputs are read directly from video.npz:
        poses:      (N, 7) — DROID-SLAM pose (t, q) per keyframe
        depths:     (N, H, W) — keyframe depths (from 1/disps in video.npz)
        intrinsics: (3, 3) — shared pinhole intrinsics

    Returns: {track_id: mean_hit_rate}.  Missing tracks → 1.0 (no penalty).
    """
    if poses is None or depths is None or intrinsics is None:
        return {tid: 1.0 for tid in tracks}

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    _, H, W = depths.shape

    # Per-track frame lookup: track_id -> {frame_idx: det}
    track_frames: Dict[int, Dict[int, Dict]] = {}
    for frame_dets in all_detections:
        for det in frame_dets:
            tid = det.get("track_id", -1)
            if tid < 0 or det.get("mask") is None:
                continue
            track_frames.setdefault(tid, {})[det["frame_idx"]] = det

    hit_rates: Dict[int, float] = {}

    for tid, frames in track_frames.items():
        if len(frames) < 2:
            hit_rates[tid] = 1.0
            continue

        # Anchor = frame with largest mask
        anchor_idx = max(frames.keys(), key=lambda i: int(frames[i]["mask"].sum()))
        anchor = frames[anchor_idx]
        mask_a = anchor["mask"]
        ys, xs = np.where(mask_a > 0)
        if ys.size == 0:
            hit_rates[tid] = 1.0
            continue

        # Sparse sample
        if ys.size > sample_points:
            sel = np.random.choice(ys.size, size=sample_points, replace=False)
            ys, xs = ys[sel], xs[sel]

        # Back-project anchor pixels to camera frame
        d_a = depths[anchor_idx][ys, xs]
        valid = d_a > 1e-3
        if valid.sum() < 10:
            hit_rates[tid] = 1.0
            continue
        xs, ys, d_a = xs[valid], ys[valid], d_a[valid]
        X_cam = np.stack([(xs - cx) * d_a / fx, (ys - cy) * d_a / fy, d_a], axis=1)  # (M, 3)
        X_cam_h = np.concatenate([X_cam, np.ones((X_cam.shape[0], 1))], axis=1)       # (M, 4)

        # tum_poses are **world-to-camera** in DROID-W. Invert the anchor pose
        # to lift camera-frame points into world coordinates.
        T_a_wc = _pose_to_Rt(poses[anchor_idx])                                       # world->cam (as stored)
        T_a_cw = np.linalg.inv(T_a_wc)                                                # cam->world
        X_w = (T_a_cw @ X_cam_h.T).T[:, :3]                                           # world coords

        hits_total, total = 0, 0
        for f_idx, det in frames.items():
            if f_idx == anchor_idx:
                continue
            T_f_wc = _pose_to_Rt(poses[f_idx])                                        # world->cam directly
            X_f = (T_f_wc[:3, :3] @ X_w.T).T + T_f_wc[:3, 3]                          # (M, 3)
            z = X_f[:, 2]
            ok = z > 1e-3
            if ok.sum() == 0:
                continue
            u = fx * X_f[ok, 0] / z[ok] + cx
            v = fy * X_f[ok, 1] / z[ok] + cy
            u_i = np.round(u).astype(int); v_i = np.round(v).astype(int)
            in_bounds = (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)
            if in_bounds.sum() == 0:
                continue
            u_i, v_i = u_i[in_bounds], v_i[in_bounds]
            m_f = det["mask"]
            # Hit = inside mask OR within hit_threshold_px (dilated mask)
            if hit_threshold_px > 0:
                k = 2 * hit_threshold_px + 1
                m_f_d = cv2.dilate(m_f.astype(np.uint8), np.ones((k, k), np.uint8), 1)
            else:
                m_f_d = m_f
            hits = int(m_f_d[v_i, u_i].sum())
            hits_total += hits
            total += int(in_bounds.sum())

        hit_rates[tid] = (hits_total / total) if total > 0 else 1.0

    # Fill missing tracks with 1.0 (no geometric evidence against them)
    for tid in tracks:
        hit_rates.setdefault(tid, 1.0)
    return hit_rates


def _logit(p: float, eps: float = 1e-4) -> float:
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


def classify_tracks_logodds(
    tracks: Dict[int, Dict],
    all_detections: List[List[Dict]],
    reproj_hits: Optional[Dict[int, float]] = None,
    movability: Optional[Dict[str, float]] = None,
    thing_stuff: Optional[Dict[str, str]] = None,
    threshold: float = 0.6,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.5,
    unc_scale: Optional[float] = None,
    static_bias: float = 1.5,
    movability_floor: float = 0.7,
    reproj_gate: float = 0.4,
    reproj_term_cap: float = 2.0,
    min_frames_for_reproj: int = 4,
    clip_margin_cutoff: float = 0.010,
    clip_margin_full: float = 0.020,
    compactness_thing_floor: float = 0.20,
    min_frames_for_dynamic_thing: int = 4,
    dino_trust_floor: float = 0.30,
) -> Dict[int, Dict]:
    """
    Conservative log-odds fusion. Defaults to STATIC; flips to dynamic only
    when at least one signal gives strong evidence of motion.

        log_odds = -static_bias
                 + α · max(0, logit(p_movable) - logit(movability_floor))
                 + β · max(0, mean((score_f - threshold) / unc_scale))
                 + γ · max(0, logit(1 - hit) - logit(1 - reproj_gate))

    is_dynamic = (log_odds > 0).

    Why each piece:
    - **static_bias** — without strong positive evidence, default static.
      Avoids "noisy chair flicker → dynamic" failure mode.
    - **movability_floor** — only classes the VLM clearly labels as self-moving
      (p ≥ 0.7) contribute positively. A chair (p≈0.5) contributes 0, not +0.85.
    - **reproj_gate** — geometric evidence only fires when hit-rate is clearly
      bad (< 0.6). Above that, the noise floor of depth/pose drift dominates.
    - All terms are **one-sided** (max(0, ·)) — signals can argue FOR dynamic
      but cannot argue AGAINST a clearly-moving object via low evidence.

    Stores debug fields: p_movable, reproj_hit_rate, mean_unc_evidence,
    mov_term, unc_term, reproj_term, logodds.
    """
    reproj_hits = reproj_hits or {}
    movability = movability or {}
    thing_stuff = thing_stuff or {}

    frame_scores: Dict[int, List[float]] = {}
    frame_clip_margins: Dict[int, List[float]] = {}
    frame_compactness: Dict[int, List[float]] = {}
    for frame_dets in all_detections:
        for det in frame_dets:
            tid = det.get("track_id", -1)
            if tid < 0:
                continue
            if "dynamic_confidence" in det:
                frame_scores.setdefault(tid, []).append(float(det["dynamic_confidence"]))
            if "clip_margin" in det:
                frame_clip_margins.setdefault(tid, []).append(float(det["clip_margin"]))
            if "compactness" in det:
                frame_compactness.setdefault(tid, []).append(float(det["compactness"]))

    if unc_scale is None or unc_scale <= 0:
        unc_scale = max(float(threshold), 1e-3)

    mov_floor_logit = _logit(movability_floor)
    reproj_floor_logit = _logit(1.0 - reproj_gate)

    for tid, t in tracks.items():
        label = t.get("label", "")
        p_mov = float(movability.get(label, 0.5))
        hit = float(reproj_hits.get(tid, 1.0))
        num_frames = int(t.get("num_frames", 0))

        # PRIMARY VETO 1: thing/stuff. A track whose label was classified by
        # the VLM as "stuff" (vegetation, sky, ground, ...) can never receive
        # the dynamic movability prior, regardless of its CLIP confidence.
        # Direct attack on the bush-as-giraffe failure mode: even if CLIP
        # mistakenly labels a bush as "giraffe", "giraffe" is a "thing" so
        # this veto doesn't help — but if the VLM also discovered "vegetation"
        # and it gets selected for some masks, those stay static. The
        # complementary check is the compactness veto below.
        ts_label = thing_stuff.get(label, "thing")

        # PRIMARY VETO 2: compactness. "thing" labels (animal, person, car)
        # require a compact mask (4π·area/perim² above floor). A sprawling
        # mask labeled "giraffe" is almost always a vegetation mis-match.
        compactness_vals = frame_compactness.get(tid, [])
        mean_compactness = float(np.mean(compactness_vals)) if compactness_vals else 1.0
        shape_veto_thing = (ts_label == "thing"
                            and mean_compactness < compactness_thing_floor)

        scores = frame_scores.get(tid, [])
        if scores:
            unc_ev = float(np.mean([(s - threshold) / unc_scale for s in scores]))
        else:
            unc_ev = (t.get("mean_dynamic_confidence", 0.0) - threshold) / unc_scale

        # CLIP-margin modulator (scale-invariant confidence of the label
        # assignment: top1 - top2 cosine similarity).
        clip_margins = frame_clip_margins.get(tid, [])
        mean_margin = float(np.mean(clip_margins)) if clip_margins else None
        if mean_margin is None:
            label_conf = 1.0   # non-mask-first backend, no margin → trust label
        elif mean_margin >= clip_margin_full:
            label_conf = 1.0
        elif mean_margin <= clip_margin_cutoff:
            label_conf = 0.0
        else:
            label_conf = ((mean_margin - clip_margin_cutoff)
                          / max(1e-6, clip_margin_full - clip_margin_cutoff))

        unc_term = max(0.0, unc_ev)

        # Reprojection: noisy for small masks & short tracks. Cap + damp
        # by label_conf so a weak-label track (mis-classified bush) can't
        # push dynamic via geometry alone.
        if num_frames < min_frames_for_reproj:
            reproj_term = 0.0
        else:
            reproj_term = max(0.0, _logit(1.0 - hit) - reproj_floor_logit)
            reproj_term = min(reproj_term, reproj_term_cap)
            reproj_term = label_conf * reproj_term

        # Movability prior: only fires when we have evidence — EITHER a
        # confident CLIP label (real giraffe's margin is large) OR clear
        # motion (real person walking has high uncertainty). A bush mis-
        # labeled "giraffe" has neither, so the prior contributes 0 and the
        # static bias wins. Uses DAMPED reproj_term so bushes' noisy
        # geometry can't satisfy the motion clause.
        motion_evidence = min(1.0, max(unc_term, reproj_term) / 0.3)
        prior_gate = max(label_conf, motion_evidence)
        conf_factor = prior_gate
        mov_term = prior_gate * max(0.0, _logit(p_mov) - mov_floor_logit)

        # Apply the four vetoes — they zero the dynamic prior even when
        # label_conf or motion_evidence would have fired it.
        # Veto 1: thing/stuff. "stuff" labels can never be dynamic.
        if ts_label == "stuff":
            mov_term = 0.0
        # Veto 2: shape. Sprawling masks labeled as a "thing" are mis-matches.
        if shape_veto_thing:
            mov_term = 0.0
        # Veto 3: persistence. Transient 2-3 frame tracks of high-movability
        # classes are usually false positives.
        if num_frames < min_frames_for_dynamic_thing:
            mov_term = 0.0
        # Veto 4: DINOv2 trust. If the second-opinion appearance check
        # (mask-pooled DINO features, per-frame variance, distance to class
        # prototype) gives low trust, the label assignment is likely wrong
        # and we shouldn't fire the dynamic prior. Bushes mis-labeled as
        # "giraffe" have wildly varying DINO embeddings across frames →
        # dino_trust ≈ 0; real giraffes track tightly → dino_trust ≈ 1.
        # Inactive when DINO features unavailable (`dino_trust is None`).
        dino_trust = t.get("dino_trust", None)
        if dino_trust is not None and dino_trust < dino_trust_floor:
            mov_term = 0.0

        lo = -static_bias + alpha * mov_term + beta * unc_term + gamma * reproj_term

        t["p_movable"] = p_mov
        t["reproj_hit_rate"] = hit
        t["mean_unc_evidence"] = unc_ev
        t["mean_clip_margin"] = mean_margin if mean_margin is not None else -1.0
        t["mean_compactness"] = mean_compactness
        t["thing_stuff"] = ts_label
        t["shape_veto"] = shape_veto_thing
        t["dino_veto"] = dino_trust is not None and dino_trust < dino_trust_floor
        t["conf_factor"] = conf_factor
        t["mov_term"] = mov_term
        t["unc_term"] = unc_term
        t["reproj_term"] = reproj_term
        t["logodds"] = lo
        t["is_dynamic"] = lo > 0.0

    return tracks
