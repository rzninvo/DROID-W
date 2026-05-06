"""SAM-3 text-prompted semantic refinement.

Drop-in replacement for the upstream SAM-1 sam_refinement(...) when running
under RADIO v4 (sam3 adaptor). SAM-3 (Meta, Ravi et al., 2025;
https://huggingface.co/facebook/sam3) is a text-prompted detection +
segmentation model — it has neither SAM-1's `predict_torch` API nor the
ability to consume coarse box/point/mask prompts the way `sam_refinement`
does. So we don't try to pipe the RADIO logits in: we run SAM-3 directly
with each user query as text, then union its instance masks per query.

Inputs
------
rgb_image_uint8 : (H, W, 3) uint8 RGB ndarray.
queries         : list[str]  — same text queries we passed to RADIO grounder.

Returns
-------
Per-image tuple (drop the batch dim — caller stacks across the batch):
seg_pred  : torch.Tensor (1, H, W) int — argmax over Q queries (0..Q-1).
            Pixels where no query fired have an arbitrary argmax but
            seg_probs[*, h, w] is all zero, so the encoder's
            `ignore_label` step (which checks max < prediction_thresh)
            sends them to the ignore class.
seg_probs : torch.Tensor (Q, H, W) float — per-query score in [0, 1].
            Same shape as upstream's pre-ignore seg_probs, so the
            ignore_label cat step works unchanged.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


@torch.no_grad()
def sam3_refinement(
    rgb_image_uint8: np.ndarray,
    queries: List[str],
    sam3_model,
    sam3_processor,
    device: str = "cuda:0",
    score_threshold: float = 0.7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run SAM-3 per-query, union instance masks → (seg_probs, seg_pred).

    Notes on the score field. SAM-3's `post_process_instance_segmentation`
    returns binary instance masks plus per-instance scores. We:
      • keep instances whose score >= score_threshold (default 0.7 —
        empirically tuned on freiburg3_walking_static; lower values
        admit bleedy low-confidence 'wall' instances)
      • union them for that query
      • use the score-weighted soft-union as the per-pixel "probability"
        so the downstream code that reads seg_probs[c, h, w] still gets a
        meaningful confidence (not just 0/1).
    """
    if rgb_image_uint8.dtype != np.uint8:
        raise ValueError(
            f"sam3_refinement: expected uint8 RGB image, "
            f"got dtype={rgb_image_uint8.dtype}"
        )
    H, W = rgb_image_uint8.shape[:2]
    pil = Image.fromarray(rgb_image_uint8)
    Q = len(queries)

    # (Q, H, W) per-class soft-score buffer. No "ignore" class baked in —
    # the encoder's ignore_label step adds it later by concat-with-zeros.
    seg_probs = torch.zeros((Q, H, W), dtype=torch.float32, device=device)

    for qi, q in enumerate(queries):
        inputs = sam3_processor(
            images=pil, text=[q], return_tensors="pt",
        ).to(device)
        out = sam3_model(**inputs)
        results = sam3_processor.post_process_instance_segmentation(
            out, threshold=score_threshold, target_sizes=[(H, W)],
        )[0]
        masks = results.get("masks")
        if masks is None or masks.numel() == 0:
            continue
        scores = results.get("scores")
        # Device-safe conversion order (reviewer C SHOULD-FIX): cast to float
        # then move to compute device. post_process_instance_segmentation
        # doesn't guarantee its output device matches `device`.
        masks_f = masks.float().to(device)
        if scores is not None and scores.numel() == masks_f.shape[0]:
            scores_f = scores.float().to(device).clamp(0, 1).view(-1, 1, 1)
            # Score-weighted union: out[h,w] = max_i (s_i * mask_i[h,w]).
            weighted = masks_f * scores_f
            sem = weighted.amax(dim=0)
        else:
            sem = (masks_f.sum(dim=0) > 0).float()
        seg_probs[qi] = sem

    # seg_pred: argmax over Q queries; (1, H, W) int. Pixels where every
    # seg_probs[qi]==0 still get an arbitrary argmax, but the upstream
    # encoder's ignore_label step (max < prediction_thresh) catches them.
    seg_pred = seg_probs.argmax(dim=0, keepdim=True).long()  # (1, H, W)
    return seg_probs, seg_pred
