"""
RADIO + SigLIP-2 per-keyframe text grounding.

Thin wrapper around the **vendored** RADIO-ViPE RADSegEncoder
(`src/utils/mono_priors/radseg/radseg_encoder.py`) so the rest of HERMES-SLAM
gets a small ergonomic surface. We do NOT re-implement the encoder — we
delegate to the upstream code so every quality knob stays exact:

  - 80-template OpenAI ImageNet prompt ensembling per query
  - Sliding-window inference (crop=336, stride=224) with feature averaging
  - SelfCorrelatingRecursiveAttn on the last attention block (SCRA)
  - Self-Correlating Global Aggregation post-sliding (SCGA)
  - SAM-refined per-class masks (vit_h)
  - Per-pixel softmax over queries + prompt-denoising + prediction-thresh

Reference: `thirdparty/RADIO-ViPE/vipe/priors/embedding/radseg_encoder.py`
Cite: Yakovlev et al., arXiv 2604.26067, April 2026 (be2rlab/RADIO-ViPE)
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.mono_priors.radseg import RADSegEncoder


_DEFAULT_RADIO_VERSION = "c-radio_v3-b"
_DEFAULT_LANG_ADAPTOR = "siglip2"
# Default location for the SAM-1 ViT-H checkpoint (`sam_vit_h_4b8939.pth`).
# Resolved relative to the DROID-W repo root unless the caller passes an
# absolute path. CVG: /home/cvg/HERMES-SLAM/DROID-W/weights/sam_vit_h_4b8939.pth
_DEFAULT_SAM_CKPT = "weights/sam_vit_h_4b8939.pth"


def _resolve_sam_ckpt(sam_ckpt: str) -> str:
    """Return an absolute path; if a relative path is given, anchor it at
    the DROID-W repo root (this file's grandparent.parent.parent.parent)."""
    p = Path(sam_ckpt)
    if p.is_absolute():
        return str(p)
    repo_root = Path(__file__).resolve().parents[3]
    return str(repo_root / sam_ckpt)


class RadioGrounder:
    """Per-keyframe RADIO+SigLIP-2 text grounding using the upstream RADSegEncoder."""

    def __init__(
        self,
        device: str = "cuda:0",
        radio_version: str = _DEFAULT_RADIO_VERSION,
        lang_adaptor: str = _DEFAULT_LANG_ADAPTOR,
        sam_refinement: bool = True,
        sam_ckpt: str = _DEFAULT_SAM_CKPT,
        text_query_mode: str = "labels",
        prediction_thresh: float = 0.0,
        prompt_denoising_thresh: float = 0.5,
        amp: bool = False,
    ):
        self.device = device
        self.text_query_mode = text_query_mode
        self.prediction_thresh = prediction_thresh
        self.prompt_denoising_thresh = prompt_denoising_thresh
        self._sam_refinement = sam_refinement
        self._sam_ckpt = _resolve_sam_ckpt(sam_ckpt) if sam_refinement else None
        self._encoder: Optional[RADSegEncoder] = None
        self._encoder_kwargs = dict(
            device=device,
            model_version=radio_version,
            lang_model=lang_adaptor,
            return_radio_features=True,
            compile=False,
            amp=amp,
            sam_refinement=sam_refinement,
            sam_ckpt=self._sam_ckpt or "",
        )
        self._queries: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Read-only delegations (kept for compatibility with the test script's
    # informational logging).
    # ------------------------------------------------------------------

    @property
    def patch_size(self) -> int:
        if self._encoder is None:
            return 16  # RADIO defaults; only used for logging before set_queries
        return int(getattr(self._encoder.model, "patch_size", 16))

    @property
    def lang_adaptor_name(self) -> str:
        return self._encoder_kwargs.get("lang_model", _DEFAULT_LANG_ADAPTOR)

    @torch.no_grad()
    def set_queries(self, queries: List[str]) -> None:
        """Build (or rebuild) the encoder in predict-mode for these queries."""
        self._queries = list(queries)
        if not queries:
            self._encoder = None
            return

        # The upstream encoder consumes prompts in __init__ when predict=True.
        # Building once per query-list keeps us faithful to upstream semantics
        # without paying the RADIO load cost twice (torch.hub caches).
        self._encoder = RADSegEncoder(
            predict=True,
            classes=list(queries),
            text_query_mode=self.text_query_mode,
            prediction_thresh=self.prediction_thresh,
            prompt_denoising_thresh=self.prompt_denoising_thresh,
            **self._encoder_kwargs,
        )

    @torch.no_grad()
    def segment(self, image_np: np.ndarray) -> dict:
        """Run upstream encoder.encode_image_to_feat_map(predict=True) on one
        keyframe; return the standard dict the test script consumes.

        Returns
        -------
        dict with keys
            'similarity' : float32 (Q, H, W) — per-class softmax probabilities
                           (with prompt-denoising applied per radseg_encoder).
                           First entry is the 'ignore' class (all zeros).
            'softmax'    : same as 'similarity' for API parity with our older
                           stripped-down version.
            'best_query' : int32 (H, W) — argmax over (Q+1) classes; 0 means
                           ignore (below prediction_thresh).
            'best_score' : float32 (H, W) — max prob.
            'queries'    : list[str] — the user-supplied queries (offset by 1
                           in best_query because of the ignore class).
        """
        if self._encoder is None:
            raise RuntimeError("set_queries() must be called before segment()")

        # Upstream wants (B, 3, H, W) float in [0, 1].
        if image_np.dtype == np.uint8:
            x = torch.from_numpy(image_np).float() / 255.0
        else:
            x = torch.from_numpy(np.asarray(image_np, dtype=np.float32))
            xmax = float(x.max().item()) if x.numel() else 0.0
            if xmax > 1.5:
                x = x / 255.0
            elif xmax > 1.0 + 1e-4:
                print(
                    f"[WARN] radio_grounding.segment: image float range "
                    f"[{float(x.min().item()):.3f}, {xmax:.3f}] is ambiguous; "
                    f"expected [0,1] or [0,255], fallback=treat as [0,1]",
                    flush=True,
                )
        if x.dim() == 3:
            x = x.permute(2, 0, 1).unsqueeze(0)
        x = x.contiguous().to(self.device)
        H_orig, W_orig = x.shape[-2], x.shape[-1]

        seg_probs, seg_pred = self._encoder.encode_image_to_feat_map(
            x, orig_img_size=(H_orig, W_orig),
            return_preds=True, ignore_label=True,
        )
        # Upstream returns:
        #   seg_probs (1, Q+1, H, W) — class 0 = ignore (zeros), 1..Q = queries.
        #   seg_pred  (1, 1,   H, W) int — 0 = ignore (no class above thr), 1..Q.
        # We strip the ignore class so query qi maps to index qi (matching
        # the test-script convention from the stripped-down version).
        sim_full = seg_probs[0].detach().cpu().float().numpy()     # (Q+1, H, W)
        sim = sim_full[1:]                                          # (Q, H, W)
        best_idx = seg_pred[0, 0].detach().cpu().int().numpy()      # (H, W) int
        # Convert: 0 → -1 (no class assigned), k → k-1.
        best_query = best_idx.astype(np.int32) - 1
        best_score = sim_full.max(axis=0).astype(np.float32)

        return {
            "similarity": sim,
            "softmax": sim,
            "best_query": best_query,
            "best_score": best_score,
            "queries": list(self._queries),
        }
