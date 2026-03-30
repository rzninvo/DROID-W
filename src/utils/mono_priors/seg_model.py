from typing import Dict, List, Optional
import os
import json

import numpy as np
import torch
import torch.nn.functional as F
from src.utils.sys_timer import timer


# Comprehensive prompt list for YOLO-World open-vocabulary detection.
# Covers both indoor and outdoor scenes. NOT split into static/dynamic —
# DROID-W's uncertainty map handles that classification post-hoc.
DEFAULT_CLASSES = [
    # People
    "person", "pedestrian", "cyclist",
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

# Stores the previous mask for temporal smoothing
_prev_mask = None


def get_seg_model(cfg: Dict):
    """
    Load detection model based on configuration.
    Supports any ultralytics model (YOLO-World, YOLOv8-seg, FastSAM, etc.)

    For YOLO-World: sets text classes from config, enabling open-vocab detection.
    For YOLOv8/FastSAM: uses standard COCO class IDs.

    Args:
        cfg: Configuration dictionary.

    Returns:
        Loaded YOLO model with classes configured.
    """
    from ultralytics import YOLO

    device = cfg["device"]
    model_name = cfg["mono_prior"].get("seg_model", "yolov8s-worldv2.pt")
    model = YOLO(model_name)
    model.to(device)

    # For YOLO-World: set the full vocabulary for open-vocab detection
    uncer_cfg = cfg["tracking"]["uncertainty_params"]
    classes = uncer_cfg.get("seg_classes", DEFAULT_CLASSES)
    if isinstance(classes, list) and len(classes) > 0 and isinstance(classes[0], str):
        model.set_classes(classes)

    return model


@torch.no_grad()
@timer.section("Segmentation")
def predict_seg_mask(
    model,
    idx: int,
    input_tensor: torch.Tensor,
    cfg: Dict,
    device: str,
    save_mask: bool = False,
) -> torch.Tensor:
    """
    Run detection and produce a binary dynamic-object mask for SLAM.

    Detects ALL objects from the configured vocabulary. Saves full detection
    metadata (boxes, labels, confidences) per frame for scene graph construction.

    For the SLAM mask: only masks detections that belong to 'likely dynamic'
    categories (people, vehicles, animals). The comprehensive detections are
    still saved so that post-processing can use DROID-W's uncertainty map to
    assign dynamic confidence scores to ALL detections regardless of class.

    Args:
        model: The YOLO model.
        idx: Frame index.
        input_tensor: Input image tensor of shape (3, H, W) in [0, 1] range.
        cfg: Configuration dictionary.
        device: Device string.
        save_mask: Whether to save mask and detections to disk.

    Returns:
        torch.Tensor: Binary mask at (H/8, W/8) with 1.0 = dynamic, 0.0 = static.
    """
    global _prev_mask

    uncer_cfg = cfg["tracking"]["uncertainty_params"]
    conf_thresh = uncer_cfg.get("seg_confidence_thresh", 0.5)
    temporal_alpha = uncer_cfg.get("seg_temporal_alpha", 0.6)
    erode_pixels = uncer_cfg.get("seg_erode_pixels", 2)
    down_scale = 8

    # Classes whose detections contribute to the SLAM dynamic mask.
    # These are the "almost always moving" categories. Everything else
    # is detected and saved but only masked if DROID-W's uncertainty
    # confirms actual motion.
    mask_classes = set(uncer_cfg.get("seg_mask_classes", [
        "person", "pedestrian", "cyclist",
        "car", "bus", "truck", "motorcycle", "bicycle", "scooter",
        "van", "taxi", "ambulance", "train",
        "dog", "cat", "bird", "horse", "cow", "sheep",
        "skateboard", "stroller", "wheelchair",
    ]))

    H, W = input_tensor.shape[-2], input_tensor.shape[-1]
    h_ds, w_ds = H // down_scale, W // down_scale

    # Convert to uint8 numpy HWC for ultralytics inference
    img = input_tensor.cpu()
    if img.dim() == 4:
        img = img.squeeze(0)
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    results = model.predict(img_np, conf=conf_thresh, verbose=False)

    # Build mask and collect ALL detections
    mask_full = torch.zeros(H, W, dtype=torch.float32, device=device)
    detections = []

    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            has_masks = result.masks is not None

            for i in range(len(classes)):
                # Get label name
                label = result.names.get(classes[i], str(classes[i])) if result.names else str(classes[i])

                detections.append({
                    "box": boxes[i].tolist(),
                    "label": label,
                    "confidence": float(confs[i]),
                    "class_id": int(classes[i]),
                    "img_h": H,
                    "img_w": W,
                })

                # Only add to SLAM mask if this is a "likely dynamic" class
                if label in mask_classes:
                    if has_masks:
                        seg_mask = result.masks.data[i]
                        if seg_mask.shape[0] != H or seg_mask.shape[1] != W:
                            seg_mask = F.interpolate(
                                seg_mask.unsqueeze(0).unsqueeze(0).float(),
                                size=(H, W),
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze()
                        mask_full = torch.max(mask_full, seg_mask.to(device).float())
                    else:
                        # Box-based mask for detection-only models (YOLO-World)
                        x1, y1, x2, y2 = boxes[i].astype(int)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(W, x2), min(H, y2)
                        mask_full[y1:y2, x1:x2] = 1.0

    # Erode mask to remove noisy edges
    if erode_pixels > 0:
        kernel_size = 2 * erode_pixels + 1
        mask_full = -F.max_pool2d(
            -mask_full.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=erode_pixels,
        ).squeeze()

    # Downsample to tracking resolution (H/8 x W/8)
    mask_ds = F.interpolate(
        mask_full.unsqueeze(0).unsqueeze(0),
        size=(h_ds, w_ds),
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    # Temporal smoothing
    if _prev_mask is not None and _prev_mask.shape == mask_ds.shape:
        mask_ds = temporal_alpha * mask_ds + (1 - temporal_alpha) * _prev_mask
    _prev_mask = mask_ds.clone()

    # Binarize
    mask_ds = (mask_ds > 0.5).float()

    if save_mask:
        output_dir = f"{cfg['data']['output']}/{cfg['scene']}"
        # Save binary mask
        mask_dir = f"{output_dir}/mono_priors/seg_masks"
        os.makedirs(mask_dir, exist_ok=True)
        np.save(f"{mask_dir}/{idx:05d}.npy", mask_ds.cpu().numpy())
        # Save ALL detections (boxes, labels, confidences) for scene graphs
        det_dir = f"{output_dir}/mono_priors/seg_detections"
        os.makedirs(det_dir, exist_ok=True)
        with open(f"{det_dir}/{idx:05d}.json", "w") as f:
            json.dump(detections, f)

    return mask_ds


def classify_detections_by_uncertainty(
    detections: List[Dict],
    uncertainty_map: torch.Tensor,
    threshold: float = 0.8,
) -> List[Dict]:
    """
    Post-hoc classification of detections as static or dynamic using
    DROID-W's uncertainty map. Call this during mapping or post-processing,
    NOT during tracking (uncertainty isn't mature yet).

    For each detection box, computes mean uncertainty of pixels inside.
    High mean uncertainty → dynamic. Low → static.

    Args:
        detections: List of detection dicts with 'box' key [x1, y1, x2, y2].
        uncertainty_map: DROID-W uncertainty tensor (H, W) or (H/8, W/8).
        threshold: Mean uncertainty above this → dynamic.

    Returns:
        Same detections list with added 'is_dynamic' bool and
        'dynamic_confidence' float fields.
    """
    H, W = uncertainty_map.shape[-2], uncertainty_map.shape[-1]

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        # Scale box to uncertainty map resolution
        scale_x = W / det.get("img_w", W)
        scale_y = H / det.get("img_h", H)
        bx1 = max(0, int(x1 * scale_x))
        by1 = max(0, int(y1 * scale_y))
        bx2 = min(W, int(x2 * scale_x))
        by2 = min(H, int(y2 * scale_y))

        if bx2 > bx1 and by2 > by1:
            region = uncertainty_map[by1:by2, bx1:bx2]
            mean_uncer = region.mean().item()
        else:
            mean_uncer = 0.0

        det["dynamic_confidence"] = mean_uncer
        det["is_dynamic"] = mean_uncer > threshold

    return detections
