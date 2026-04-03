"""
Object detection module for scene graph construction.

Runs on DROID-W keyframes and produces per-frame detections (bounding boxes,
labels, confidences). Static/dynamic classification is done exclusively via
DROID-W's uncertainty map — no hardcoded class-based separation.

Data flow:
    DROID-W saves keyframes (RGB, pose, depth, FiT3D features, uncertainty)
    → This module detects all objects via YOLO-World
    → classify_detections_by_uncertainty() tags each detection using the
      uncertainty map (75th percentile inside each box)
    → Scene graph module consumes labeled, tagged detections
"""

from typing import Dict, List
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


def get_detector(model_name: str = "yolov8s-worldv2.pt", classes: List[str] = None, device: str = "cuda:0"):
    """
    Load YOLO-World detection model and set vocabulary.

    Args:
        model_name: Ultralytics model name (auto-downloads weights).
        classes: Text class list for open-vocab detection. Defaults to DEFAULT_CLASSES.
        device: Device string.

    Returns:
        Loaded YOLO model with vocabulary set.
    """
    from ultralytics import YOLO

    model = YOLO(model_name)
    model.to(device)

    classes = classes or DEFAULT_CLASSES
    if isinstance(classes, list) and len(classes) > 0 and isinstance(classes[0], str):
        model.set_classes(classes)

    return model


@torch.no_grad()
def detect_objects(
    model,
    image_np: np.ndarray,
    conf_thresh: float = 0.15,
) -> List[Dict]:
    """
    Run open-vocabulary detection on a single frame.

    Args:
        model: YOLO-World model from get_detector().
        image_np: uint8 numpy array (H, W, 3) in RGB.
        conf_thresh: Minimum detection confidence.

    Returns:
        List of detections, each a dict with:
            box: [x1, y1, x2, y2] in pixel coords
            label: str class name
            confidence: float
            class_id: int (model-internal, not COCO)
            img_h: int
            img_w: int
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
        detections: List of detection dicts from detect_objects().
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
