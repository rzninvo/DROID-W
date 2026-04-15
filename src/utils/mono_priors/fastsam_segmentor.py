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
def segment_everything(
    model,
    image_np: np.ndarray,
    device: str = "cuda:0",
    imgsz: int = 1024,
    conf: float = 0.4,
    iou: float = 0.9,
) -> List[Dict]:
    """
    Run FastSAM in "segment everything" mode — returns class-agnostic masks
    for every region FastSAM detects (typically 50-100 on indoor scenes).

    This is the mask-first paradigm used by ConceptGraphs / HOV-SG / OVO-SLAM.
    The returned masks are handed to a downstream CLIP-per-mask classifier
    which assigns text labels from a VLM-discovered vocabulary.

    Args:
        model: FastSAM model from get_fastsam_model().
        image_np: uint8 numpy array (H, W, 3) in RGB.
        imgsz: FastSAM inference size. 1024 is the mask quality/speed sweet spot.
        conf: FastSAM region-proposal confidence. Lower = more masks.
        iou: FastSAM internal NMS IoU.

    Returns:
        List of proposals, each a dict with:
            mask: np.uint8 (H, W) — binary mask at image resolution
            box:  [x1, y1, x2, y2] — minimum enclosing box of the mask
            fastsam_confidence: float — FastSAM's own region confidence
            area: int — mask pixel count
            img_h, img_w: ints
    """
    H, W = image_np.shape[:2]
    results = model(
        image_np, device=device, retina_masks=True, imgsz=imgsz,
        conf=conf, iou=iou, verbose=False,
    )
    proposals = []
    if not results or results[0].masks is None or len(results[0].masks.data) == 0:
        return proposals

    r = results[0]
    mask_arr = r.masks.data.detach().cpu().numpy()   # (N, h, w), uint8 or float
    if mask_arr.ndim == 2:
        mask_arr = mask_arr[None]
    mask_arr = (mask_arr > 0.5).astype(np.uint8)

    # Boxes & confs (FastSAM sometimes emits masks without boxes; fall back)
    if r.boxes is not None and len(r.boxes) == len(mask_arr):
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
    else:
        boxes = confs = None

    for i, m in enumerate(mask_arr):
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        area = int(m.sum())
        if area == 0:
            continue
        if boxes is not None:
            x1, y1, x2, y2 = boxes[i].tolist()
        else:
            ys, xs = np.where(m > 0)
            x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
        proposals.append({
            "mask": m,
            "box": [float(x1), float(y1), float(x2), float(y2)],
            "fastsam_confidence": float(confs[i]) if confs is not None else 1.0,
            "area": area,
            "img_h": H,
            "img_w": W,
        })
    return proposals


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
