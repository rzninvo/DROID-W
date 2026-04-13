"""
FastSAM-based instance segmentation driven by YOLO-World detections.

YOLO-World produces boxes + class labels. FastSAM takes each box as a prompt
and returns a precise pixel mask of the object inside it. Masks are used
downstream for uncertainty sampling (removes background contamination that
caused static objects to be wrongly tagged dynamic).
"""

from typing import List, Dict, Optional

import numpy as np
import cv2
import torch


def get_fastsam_model(model_name: str = "FastSAM-s.pt", device: str = "cuda:0"):
    """Load FastSAM model. Weights auto-download on first run."""
    from ultralytics import FastSAM
    model = FastSAM(model_name)
    return model


@torch.no_grad()
def predict_masks_for_detections(
    model,
    image_np: np.ndarray,
    detections: List[Dict],
    device: str = "cuda:0",
    imgsz: int = 1024,
) -> List[Dict]:
    """
    For each detection, prompt FastSAM with the YOLO-World box and attach the
    resulting binary mask to the detection dict under key 'mask'.

    Args:
        model: FastSAM model from get_fastsam_model().
        image_np: uint8 numpy array (H, W, 3) in RGB.
        detections: List of dicts from track_objects()/detect_objects() — each
                    must have 'box' = [x1, y1, x2, y2].

    Returns:
        Same list with each detection augmented by 'mask': np.uint8 (H, W),
        or None if FastSAM produced no mask for that prompt.
    """
    if not detections:
        return detections

    H, W = image_np.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        results = model(
            image_np, device=device, retina_masks=True, imgsz=imgsz,
            conf=0.4, iou=0.9, bboxes=[[x1, y1, x2, y2]], verbose=False,
        )
        mask = None
        if results and results[0].masks is not None and len(results[0].masks.data) > 0:
            m = (results[0].masks.data[0].cpu().numpy() > 0.5).astype(np.uint8)
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            mask = m
        det["mask"] = mask

    return detections
