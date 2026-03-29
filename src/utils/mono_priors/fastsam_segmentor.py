from typing import Dict, List
import os

import numpy as np
import torch
import torch.nn.functional as F
from src.utils.sys_timer import timer


# COCO class IDs for semantically dynamic objects
DEFAULT_DYNAMIC_CLASSES = [
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    6,   # train
    7,   # truck
    14,  # bird
    15,  # cat
    16,  # dog
    17,  # horse
    18,  # sheep
    19,  # cow
]


def get_fastsam_model(cfg: Dict):
    """
    Load FastSAM model based on configuration.

    Args:
        cfg: Configuration dictionary.

    Returns:
        Loaded YOLO/FastSAM model.
    """
    from ultralytics import YOLO

    device = cfg["device"]
    model_name = cfg["mono_prior"].get("fastsam_model", "FastSAM-s.pt")
    model = YOLO(model_name)
    model.to(device)
    return model


@torch.no_grad()
@timer.section("FastSAM Segmentation")
def predict_fastsam_mask(
    model,
    idx: int,
    input_tensor: torch.Tensor,
    cfg: Dict,
    device: str,
    save_mask: bool = False,
) -> torch.Tensor:
    """
    Run FastSAM segmentation and produce a binary dynamic-object mask.

    Args:
        model: The YOLO/FastSAM model.
        idx: Frame index.
        input_tensor: Input image tensor of shape (3, H, W) in [0, 1] range.
        cfg: Configuration dictionary.
        device: Device string.
        save_mask: Whether to save the mask to disk.

    Returns:
        torch.Tensor: Binary mask at (H/8, W/8) with 1.0 = dynamic, 0.0 = static.
    """
    uncer_cfg = cfg["tracking"]["uncertainty_params"]
    dynamic_classes = uncer_cfg.get("fastsam_dynamic_classes", DEFAULT_DYNAMIC_CLASSES)
    conf_thresh = uncer_cfg.get("fastsam_confidence_thresh", 0.5)
    down_scale = 8

    H, W = input_tensor.shape[-2], input_tensor.shape[-1]
    h_ds, w_ds = H // down_scale, W // down_scale

    # Convert to uint8 numpy HWC for ultralytics inference
    img_np = (input_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    results = model.predict(img_np, conf=conf_thresh, verbose=False)

    # Build union mask of all dynamic-class detections
    mask_full = torch.zeros(H, W, dtype=torch.float32, device=device)

    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None and result.masks is not None:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            masks = result.masks.data  # (N, H_mask, W_mask)

            for i, cls_id in enumerate(classes):
                if cls_id in dynamic_classes:
                    seg_mask = masks[i]  # (H_mask, W_mask)
                    # Resize to full resolution if needed
                    if seg_mask.shape[0] != H or seg_mask.shape[1] != W:
                        seg_mask = F.interpolate(
                            seg_mask.unsqueeze(0).unsqueeze(0).float(),
                            size=(H, W),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze()
                    mask_full = torch.max(mask_full, seg_mask.to(device).float())

    # Downsample to tracking resolution (H/8 x W/8)
    mask_ds = F.interpolate(
        mask_full.unsqueeze(0).unsqueeze(0),
        size=(h_ds, w_ds),
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    # Binarize
    mask_ds = (mask_ds > 0.5).float()

    if save_mask:
        output_dir = f"{cfg['data']['output']}/{cfg['scene']}"
        mask_dir = f"{output_dir}/mono_priors/fastsam_masks"
        os.makedirs(mask_dir, exist_ok=True)
        np.save(f"{mask_dir}/{idx:05d}.npy", mask_ds.cpu().numpy())

    return mask_ds
