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
    # Appliances & misc
    "tv", "microwave", "oven", "refrigerator", "sink",
    "toilet", "fan", "robot vacuum", "toy", "ball",
]


def get_detector(
    model_name: str = "yolov8s-worldv2.pt",
    classes: List[str] = None,
    device: str = "cuda:0",
    output_dir: str = None,
):
    """
    Load YOLO-World detection model and set vocabulary.

    Class priority:
        1. Explicit `classes` argument (if provided)
        2. VLM-discovered classes from `output_dir/vlm_classes.json` (if exists)
        3. DEFAULT_CLASSES fallback (only when VLM is not used)

    Args:
        model_name: Ultralytics model name (auto-downloads weights).
        classes: Text class list for open-vocab detection.
        device: Device string.
        output_dir: SLAM output directory — checked for VLM-discovered classes.

    Returns:
        Loaded YOLO model with vocabulary set.
    """
    from ultralytics import YOLO

    model = YOLO(model_name)
    model.to(device)

    # Resolve class list: explicit > VLM-discovered > fallback
    if classes is None and output_dir is not None:
        from src.utils.mono_priors.vlm_scene_scout import VLMSceneScout
        vlm_classes = VLMSceneScout.load_classes(output_dir)
        if vlm_classes is not None:
            classes = vlm_classes

    if classes is None:
        classes = DEFAULT_CLASSES

    if isinstance(classes, list) and len(classes) > 0:
        model.set_classes(classes)

    return model


def update_detector_classes(model, classes: List[str]):
    """
    Update YOLO-World vocabulary with a new class list.

    Called when the VLM scene scout discovers new object types. Re-encodes
    text prompts into CLIP embeddings (~50ms), so only call when the list
    actually changes.

    Args:
        model: YOLO-World model from get_detector().
        classes: Updated text class list.
    """
    if isinstance(classes, list) and len(classes) > 0:
        model.set_classes(classes)


@torch.no_grad()
def detect_objects(
    model,
    image_np: np.ndarray,
    conf_thresh: float = 0.15,
) -> List[Dict]:
    """
    Run open-vocabulary detection on a single frame (no tracking).

    Args:
        model: YOLO-World model from get_detector().
        image_np: uint8 numpy array (H, W, 3) in RGB.
        conf_thresh: Minimum detection confidence.

    Returns:
        List of detections, each a dict with:
            box, label, confidence, class_id, img_h, img_w
    """
    H, W = image_np.shape[:2]
    results = model.predict(image_np, conf=conf_thresh, verbose=False)

    detections = []
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(classes)):
                label = result.names.get(classes[i], str(classes[i])) if result.names else str(classes[i])
                detections.append({
                    "box": boxes[i].tolist(),
                    "label": label,
                    "confidence": float(confs[i]),
                    "class_id": int(classes[i]),
                    "img_h": H,
                    "img_w": W,
                })

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
    results = model.track(image_np, conf=conf_thresh, persist=True, tracker=tracker, verbose=False)

    detections = []
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            # track IDs may not exist if tracker loses the object
            if result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = np.full(len(classes), -1, dtype=int)

            for i in range(len(classes)):
                label = result.names.get(classes[i], str(classes[i])) if result.names else str(classes[i])
                detections.append({
                    "box": boxes[i].tolist(),
                    "label": label,
                    "confidence": float(confs[i]),
                    "class_id": int(classes[i]),
                    "track_id": int(track_ids[i]),
                    "frame_idx": frame_idx,
                    "img_h": H,
                    "img_w": W,
                })

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


def classify_detections_by_uncertainty(
    detections: List[Dict],
    uncertainty_map: torch.Tensor,
    threshold: float = 0.6,
    percentile: float = 75.0,
) -> List[Dict]:
    """
    Tag each detection as static or dynamic using DROID-W's uncertainty map.

    Uses the Nth percentile of uncertainty inside each box rather than the
    mean.  This is robust to bounding boxes that include background pixels
    (low uncertainty) which would dilute a simple mean and cause false
    negatives on genuinely dynamic objects.

    Args:
        detections: List of detection dicts from detect_objects() or track_objects().
        uncertainty_map: DROID-W uncertainty tensor at any resolution.
        threshold: Percentile value above this → dynamic.
        percentile: Which percentile to use (default 75th).

    Returns:
        Same detections with added 'is_dynamic' and 'dynamic_confidence' fields.
    """
    u_h, u_w = uncertainty_map.shape[-2], uncertainty_map.shape[-1]

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        img_h, img_w = det["img_h"], det["img_w"]

        # Scale box from image resolution to uncertainty map resolution
        bx1 = max(0, int(x1 * u_w / img_w))
        by1 = max(0, int(y1 * u_h / img_h))
        bx2 = min(u_w, int(x2 * u_w / img_w))
        by2 = min(u_h, int(y2 * u_h / img_h))

        if bx2 > bx1 and by2 > by1:
            region = uncertainty_map[by1:by2, bx1:bx2].flatten()
            score = torch.quantile(region.float(), percentile / 100.0).item()
        else:
            score = 0.0

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
