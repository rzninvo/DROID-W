"""
Mask-first open-vocabulary classification.

SOTA scene-graph papers (ConceptGraphs, HOV-SG, OVO-SLAM, ConceptFusion) do
not use open-vocab detectors for recall-critical work. They instead:

    1. Run SAM / FastSAM in 'segment everything' mode → class-agnostic masks
    2. For each mask, encode the crop via CLIP's image encoder
    3. Match against text embeddings of a VLM-discovered class vocabulary
    4. Keep highest-similarity label above a threshold; drop the rest

This file implements that recipe as a drop-in alternative to the YOLO-World +
box-prompted FastSAM pipeline. Output is a list of detections with the same
schema, so the existing uncertainty / reprojection / log-odds classification
downstream is unchanged.
"""

from typing import Dict, List, Optional

import numpy as np
import cv2
import torch


def _as_feature_tensor(out) -> torch.Tensor:
    """Coerce HF / OpenCLIP encoder outputs to a (B, D) feature tensor.

    transformers 5.x's Siglip2Model.get_{text,image}_features can return
    BaseModelOutputWithPooling. Pick the pooled / embeds field; fall back
    to last hidden state mean-pool, else raise.
    """
    if torch.is_tensor(out):
        return out
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        if hasattr(out, attr) and getattr(out, attr) is not None:
            t = getattr(out, attr)
            if torch.is_tensor(t):
                return t
    if hasattr(out, "last_hidden_state") and torch.is_tensor(out.last_hidden_state):
        return out.last_hidden_state.mean(dim=1)
    raise RuntimeError(
        f"[mask_classifier] unexpected encoder output type: {type(out).__name__}"
    )


class MaskClassifier:
    """FastSAM segment-everything + per-mask open-vocab classification.

    Two image-text encoder backends are supported:

      - 'clip'    : OpenCLIP (default, ViT-B-32). Same as the original
                    ConceptGraphs / OVO-SLAM recipe.
      - 'siglip2' : Google SigLIP 2 (Tschannen et al., arXiv 2502.14786).
                    Sigmoid-loss image-text encoder; outperforms CLIP on
                    fine-grained zero-shot classification at similar size.

    The encoder choice is local to per-mask labelling — it does NOT change
    DROID-W's internal DINOv2 feature pipeline.
    """

    # ConceptGraphs/OpenScene-style prompt ensemble. Multiple templates per
    # class are encoded then averaged (and re-normalized) to produce a more
    # robust text embedding. Empirically gains 2-4% AP on open-vocab seg.
    PROMPT_TEMPLATES = (
        "a photo of a {}",
        "a {}",
        "a {} in a scene",
        "an image of a {}",
    )

    # SigLIP 2 text inputs MUST be tokenised with padding='max_length' and
    # max_length=64 (model card requirement, same as SigLIP 1).
    SIGLIP2_TEXT_MAX_LENGTH = 64

    def __init__(
        self,
        classes: List[str],
        fastsam_model,
        clip_model: str = "ViT-B-32",
        device: str = "cuda:0",
        prompt_templates: Optional[tuple] = None,
        encoder: str = "clip",
        siglip2_model: str = "google/siglip2-large-patch16-256",
    ):
        self.device = device
        self.fastsam = fastsam_model
        self.encoder_kind = str(encoder).lower()
        if self.encoder_kind not in ("clip", "siglip2"):
            raise ValueError(f"encoder must be 'clip' or 'siglip2', got {encoder!r}")
        self.clip_model_name = clip_model
        self.siglip2_model_name = siglip2_model
        self.prompt_templates = prompt_templates if prompt_templates is not None else self.PROMPT_TEMPLATES
        self._encoder = None      # backend model (open_clip model OR HF AutoModel)
        self._preprocess = None   # callable(PIL) → tensor (image preprocessing)
        self._tokenizer = None    # CLIP backend only
        self._processor = None    # SigLIP2 backend only (HF AutoProcessor)
        if self.encoder_kind == "siglip2":
            self._load_siglip2()
        else:
            self._load_clip()
        self.classes: List[str] = []
        self._text_emb: Optional[torch.Tensor] = None
        self.update_vocabulary(classes)

    def _load_clip(self):
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.clip_model_name, pretrained="openai"
        )
        model.eval().to(self.device)
        self._encoder = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.clip_model_name)

    def _load_siglip2(self):
        # Requires transformers >= 4.49 (Feb 2025) which ships SigLIP 2.
        from transformers import AutoModel, AutoProcessor
        model = AutoModel.from_pretrained(self.siglip2_model_name)
        model.eval().to(self.device)
        processor = AutoProcessor.from_pretrained(self.siglip2_model_name)
        self._encoder = model
        self._processor = processor
        # Image preprocessing happens via the processor at call time, but
        # callers expect `_preprocess(pil_img) -> tensor`. Adapt:
        def _siglip2_preprocess(pil_img):
            out = processor(images=[pil_img], return_tensors="pt")
            # Strip batch dim → (3, H, W) tensor
            return out["pixel_values"].squeeze(0)
        self._preprocess = _siglip2_preprocess

    @torch.no_grad()
    def _encode_text(self, prompts: List[str]) -> torch.Tensor:
        """Encode a list of prompts → L2-normed (N, D) tensor on self.device."""
        if self.encoder_kind == "siglip2":
            inputs = self._processor(
                text=prompts,
                padding="max_length",
                max_length=self.SIGLIP2_TEXT_MAX_LENGTH,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            emb = self._encoder.get_text_features(**inputs)
        else:
            tokens = self._tokenizer(prompts).to(self.device)
            emb = self._encoder.encode_text(tokens)
        emb = _as_feature_tensor(emb)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    @torch.no_grad()
    def _encode_image_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """Encode a (B, 3, H, W) preprocessed batch → L2-normed (B, D)."""
        batch = batch_tensor.to(self.device)
        if self.encoder_kind == "siglip2":
            emb = self._encoder.get_image_features(pixel_values=batch)
        else:
            emb = self._encoder.encode_image(batch)
        emb = _as_feature_tensor(emb)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    @torch.no_grad()
    def update_vocabulary(self, classes: List[str]):
        """Precompute prompt-ensembled text embeddings for the vocabulary.

        For each class, encode every prompt template, L2-normalize each, then
        average and renormalize. This is the canonical recipe from CLIP /
        ConceptGraphs and consistently outperforms a single template.
        """
        classes = list(classes) if classes else []
        if not classes:
            self.classes, self._text_emb = [], None
            return
        all_prompts = [tmpl.format(c) for c in classes for tmpl in self.prompt_templates]
        emb = self._encode_text(all_prompts)
        # Reshape (K * T, D) -> (K, T, D), mean across templates, renormalize.
        K, T, D = len(classes), len(self.prompt_templates), emb.shape[-1]
        emb = emb.view(K, T, D).mean(dim=1)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        self.classes = classes
        self._text_emb = emb  # (K, D)

    @torch.no_grad()
    def segment_and_classify(
        self,
        image_np: np.ndarray,
        frame_idx: int = 0,
        accept_thresh: float = 0.23,
        mask_iou_dedup: float = 0.65,
        containment_thresh: float = 0.70,
        min_mask_area: int = 20,
        max_masks: int = 120,
        batch_size: int = 32,
    ) -> List[Dict]:
        """
        Segment everything, classify each mask, return detections with the
        same schema as `track_objects()` so downstream code is reusable.

        Returns:
            List of dicts with keys: box, mask, label, confidence, class_id,
            frame_idx, img_h, img_w, fastsam_confidence, area.
            (track_id is NOT set here — use `track_masks()` afterwards.)
        """
        if self._text_emb is None or len(self.classes) == 0:
            return []

        from src.utils.mono_priors.fastsam_segmentor import segment_everything
        proposals = segment_everything(self.fastsam, image_np, device=self.device)

        # Drop tiny masks, cap to max_masks (largest areas first)
        proposals = [p for p in proposals if p["area"] >= min_mask_area]
        if len(proposals) > max_masks:
            proposals.sort(key=lambda p: -p["area"])
            proposals = proposals[:max_masks]
        if not proposals:
            return []

        # Build 224x224 crops for CLIP. We crop from the bounding box and zero
        # pixels OUTSIDE the mask so the CLIP encoder sees only the object
        # (ConceptGraphs does this; it noticeably improves label quality).
        crops = []
        for p in proposals:
            x1, y1, x2, y2 = [int(v) for v in p["box"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(image_np.shape[1], x2)
            y2 = min(image_np.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                crops.append(None)
                continue
            sub_img = image_np[y1:y2, x1:x2].copy()
            sub_mask = p["mask"][y1:y2, x1:x2]
            sub_img[sub_mask == 0] = 0  # zero out background
            # open_clip preprocess expects PIL
            from PIL import Image
            pil = Image.fromarray(sub_img)
            crops.append(self._preprocess(pil))

        # Batch-encode crops
        valid_idx = [i for i, c in enumerate(crops) if c is not None]
        if not valid_idx:
            return []

        img_emb = torch.empty((len(crops), self._text_emb.shape[1]), device=self.device)
        for start in range(0, len(valid_idx), batch_size):
            batch_idx = valid_idx[start:start + batch_size]
            batch = torch.stack([crops[i] for i in batch_idx])
            e = self._encode_image_batch(batch)
            for k, gi in enumerate(batch_idx):
                img_emb[gi] = e[k]

        # Cosine sim (N, K) → top-1 & top-2 used to gauge label confidence:
        # a real match typically scores much higher than the runner-up, but
        # mis-classifications land in a crowded low-confidence zone where
        # several labels are near-tied. The MARGIN (top1 − top2) is
        # scale-invariant across scenes/models and works even when absolute
        # CLIP scores are low (small-res crops of people ≈ 0.26).
        sim = (img_emb @ self._text_emb.T).cpu().numpy()
        if sim.shape[1] >= 2:
            top2_idx = np.argpartition(-sim, 2, axis=1)[:, :2]
            row_idx = np.arange(sim.shape[0])[:, None]
            top2_vals = sim[row_idx, top2_idx]
            # Ensure top1 is column 0
            top1_is_col0 = top2_vals[:, 0] >= top2_vals[:, 1]
            best_conf = np.where(top1_is_col0, top2_vals[:, 0], top2_vals[:, 1])
            runner_up = np.where(top1_is_col0, top2_vals[:, 1], top2_vals[:, 0])
            best_cls = np.where(top1_is_col0, top2_idx[:, 0], top2_idx[:, 1])
        else:
            best_conf = sim.max(axis=1)
            best_cls = sim.argmax(axis=1)
            runner_up = np.zeros_like(best_conf)
        margin = best_conf - runner_up

        # Store per-mask CLIP image embedding on each detection so downstream
        # tracking can match across frames by appearance (ConceptGraphs-style).
        clip_emb_np = img_emb.detach().cpu().numpy().astype(np.float32)

        dets: List[Dict] = []
        H, W = image_np.shape[:2]
        for i, p in enumerate(proposals):
            if i not in valid_idx:
                continue
            conf = float(best_conf[i])
            if conf < accept_thresh:
                continue
            class_id = int(best_cls[i])
            dets.append({
                "box": p["box"],
                "mask": p["mask"],
                "label": self.classes[class_id],
                "confidence": conf,
                "clip_margin": float(margin[i]),   # top1 - top2 CLIP cos; large = confident match
                "class_id": class_id,
                "frame_idx": frame_idx,
                "img_h": H,
                "img_w": W,
                "fastsam_confidence": p["fastsam_confidence"],
                "area": p["area"],
                "compactness": _mask_compactness(p["mask"]),  # 4π·area/perim²; ~1=circle, <0.4=sprawling
                "clip_emb": clip_emb_np[i],   # unit-norm CLIP image embedding
            })

        # Two-stage deduplication to make "one object → one mask":
        #
        # 1) IoU dedup (confidence-sorted). Suppress masks that overlap ≥
        #    mask_iou_dedup IoU with an already-kept mask, regardless of
        #    label ("monitor" vs "computer monitor" on the same region →
        #    keep one).
        #
        # 2) Containment dedup (area-sorted, largest first). Suppress masks
        #    whose pixels are mostly inside a larger kept mask
        #    (sub_area ∩ parent_area / sub_area ≥ containment_thresh).
        #    This kills the "poster → also text, also photo, also border"
        #    over-segmentation FastSAM produces in everything-mode.
        #    Crucially does NOT kill "laptop on desk" because FastSAM
        #    segments around the laptop, so the laptop's mask pixels are
        #    NOT inside the desk's mask pixels.
        dets.sort(key=lambda d: -d["confidence"])
        kept: List[Dict] = []
        for d in dets:
            dup = False
            for k in kept:
                if _mask_iou(d["mask"], k["mask"]) >= mask_iou_dedup:
                    dup = True; break
            if not dup:
                kept.append(d)

        # Containment pass — sort by area descending so parents are always
        # checked against before their children would be.
        kept.sort(key=lambda d: -d["area"])
        final: List[Dict] = []
        for d in kept:
            absorbed = False
            d_mask = d["mask"]; d_area = d["area"]
            for parent in final:
                if parent["area"] <= d_area:
                    continue  # parent must be strictly larger
                inter = int(np.logical_and(d_mask, parent["mask"]).sum())
                if inter == 0:
                    continue
                containment = inter / max(1, d_area)
                if containment >= containment_thresh:
                    absorbed = True
                    break
            if not absorbed:
                final.append(d)
        return final


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Binary mask IoU. Inputs are np.uint8 {0,1} of the same shape."""
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def _mask_compactness(mask: np.ndarray) -> float:
    """
    Isoperimetric compactness of a binary mask: 4π·area / perimeter².
    Circle = 1.0; compact convex blob ≈ 0.6-0.9; sprawling/irregular
    region (vegetation, hair-like) ≈ 0.1-0.4. Returns 0 for empty masks.

    Used as a "thing vs stuff" shape prior — animals/people/cars are
    compact; vegetation/sky/ground are sprawling.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area <= 0:
        return 0.0
    perim = cv2.arcLength(largest, True)
    if perim <= 0:
        return 0.0
    return float(min(1.0, 4.0 * np.pi * area / (perim * perim)))


def _box_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    u = area_a + area_b - inter
    return inter / u if u > 0 else 0.0


class MaskTracker:
    """
    Appearance-aware mask tracker for the mask-first pipeline (FastSAM + CLIP).

    Matching combines three signals:
      • CLIP image-embedding cosine similarity (primary — ConceptGraphs)
      • Mask IoU (for static objects that barely move)
      • Box IoU (fallback, looser)

    Score formula per (prev_track, current_det) pair (same label only):
        score = max( clip_cos, max(mask_iou, box_iou) )

    Default thresholds:
      • match_thresh = 0.65 — accept a match at this combined score or above
      • max_age = 10 keyframes — keep unmatched tracks alive this long

    Why CLIP features dominate: FastSAM's segment-everything produces
    different masks on each keyframe even for the same static object; strict
    IoU matching fragments every track. CLIP embeddings are appearance-based
    and stable across motion, so the SAME person walking keeps the same ID
    even though their mask position changes frame-to-frame.
    """
    def __init__(self, match_thresh: float = 0.65, max_age: int = 10):
        self.match_thresh = match_thresh
        self.max_age = max_age
        self._next_id = 1
        self._tracks: Dict[int, Dict] = {}  # track_id -> {mask, box, label, age, emb}

    def update(self, detections: List[Dict]) -> List[Dict]:
        """Assign `track_id` to each detection."""
        for tid in list(self._tracks):
            self._tracks[tid]["age"] += 1
            if self._tracks[tid]["age"] > self.max_age:
                del self._tracks[tid]

        if not detections:
            return detections

        candidates: List[tuple] = []
        for di, det in enumerate(detections):
            det_emb = det.get("clip_emb")
            for tid, prev in self._tracks.items():
                if prev.get("label") != det["label"]:
                    continue
                iou_m = _mask_iou(det["mask"], prev["mask"])
                iou_b = _box_iou(det["box"], prev["box"])
                geo = max(iou_m, iou_b)
                clip_cos = 0.0
                prev_emb = prev.get("emb")
                if det_emb is not None and prev_emb is not None:
                    clip_cos = float(np.dot(det_emb, prev_emb))
                score = max(clip_cos, geo)
                if score >= self.match_thresh:
                    candidates.append((score, di, tid))

        candidates.sort(reverse=True)
        matched_det = set()
        used_tid = set()
        for score, di, tid in candidates:
            if di in matched_det or tid in used_tid:
                continue
            detections[di]["track_id"] = tid
            matched_det.add(di); used_tid.add(tid)
            # EMA-update the appearance embedding so it drifts with the object
            prev_emb = self._tracks[tid].get("emb")
            cur_emb = detections[di].get("clip_emb")
            if prev_emb is not None and cur_emb is not None:
                new_emb = 0.7 * prev_emb + 0.3 * cur_emb
                n = np.linalg.norm(new_emb) + 1e-8
                new_emb = new_emb / n
            else:
                new_emb = cur_emb
            self._tracks[tid].update(
                mask=detections[di]["mask"], box=detections[di]["box"],
                label=detections[di]["label"], age=0, emb=new_emb,
            )

        # Unmatched detections → new track IDs
        for di, det in enumerate(detections):
            if di in matched_det:
                continue
            det["track_id"] = self._next_id
            self._tracks[self._next_id] = dict(
                mask=det["mask"], box=det["box"], label=det["label"], age=0,
                emb=det.get("clip_emb"),
            )
            self._next_id += 1
        return detections
