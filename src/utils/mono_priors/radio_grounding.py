"""
RADIO + SigLIP-2 per-keyframe text grounding.

Stripped-down port of the RADIO-ViPE paradigm (Yakovlev et al., arXiv 2604.26067,
April 2026 — be2rlab/RADIO-ViPE) for HERMES-SLAM. Keeps only the per-keyframe
grounding pipeline, dropping the paper's quality-of-life upgrades:

  - No SelfCorrelatingRecursiveAttn (the last-block attention rewrite).
  - No sliding-window inference (single forward at the suggested resolution).
  - No SAM refinement.
  - No openai_imagenet_template prompt ensembling (single 'a photo of a {}.').

If the segmentation quality is insufficient, the upgrades can be re-introduced
piece-by-piece — see thirdparty/RADIO-ViPE/vipe/priors/embedding/radseg_encoder.py
for the reference implementation.

Pipeline per keyframe:
    RGB image (H, W, 3) uint8
        → torch.hub.load("NVlabs/RADIO", "radio_model", version=...)
        → forward pass with adaptors stripped → backbone features
                                                (B, N=H/16·W/16, D=768)
        → adaptor['siglip2'].head_mlp → SigLIP-2-aligned (B, N, 1152)
        → reshape to (B, 1152, H/16, W/16)
        → cosine vs SigLIP-2 text embeddings → per-pixel similarity

Reference:
    `thirdparty/RADIO-ViPE/vipe/priors/embedding/radseg_encoder.py:289-365`
    is the equivalent in the upstream code.
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F


# Default adaptor name on the c-radio_v3-b checkpoint exposed via NVlabs/RADIO
# torch.hub. The upstream radseg_encoder.py passes 'siglip2-g' but that name is
# only present on some older RADIO checkpoints; the current pretrained weights
# expose 'siglip2'.
_DEFAULT_LANG_ADAPTOR = "siglip2"
_DEFAULT_RADIO_VERSION = "c-radio_v3-b"

# CLIP/SigLIP cosine-similarity prompt; we keep only the canonical template
# (cf. radseg_encoder.encode_labels which ensembles 80 ImageNet templates).
_PROMPT_TEMPLATE = "a photo of a {}."


class RadioGrounder:
    """Loads RADIO + SigLIP-2 once, exposes per-keyframe text grounding."""

    def __init__(
        self,
        device: str = "cuda:0",
        radio_version: str = _DEFAULT_RADIO_VERSION,
        lang_adaptor: str = _DEFAULT_LANG_ADAPTOR,
        amp: bool = True,
    ):
        self.device = device
        self.radio_version = radio_version
        self.lang_adaptor_name = lang_adaptor
        self.amp = amp

        # Load RADIO via torch.hub. Caches under ~/.cache/torch/hub.
        # Using source='github' (default) so the repo auto-clones.
        model = torch.hub.load(
            "NVlabs/RADIO",
            "radio_model",
            version=radio_version,
            progress=False,
            skip_validation=True,
            adaptor_names=[lang_adaptor],
            force_reload=False,
            trust_repo=True,
        )
        model = model.to(device).eval()

        # Steal the adaptor before nulling so we can call head_mlp/encode_text
        # ourselves; nulling adaptors makes the model return a single
        # RadioOutput (with .features, .summary) instead of a dict per-adaptor.
        self.lang_adaptor = model.adaptors[lang_adaptor]
        model.adaptors = None
        self.model = model
        self.patch_size = int(getattr(model, "patch_size", 16))

        # Cached text embeddings — populated by set_queries().
        self._queries: List[str] = []
        self._text_emb: Optional[torch.Tensor] = None  # (Q, D_lang) L2-normed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def set_queries(self, queries: List[str]) -> None:
        """Encode + cache text embeddings for the given list of text queries.

        Parameters
        ----------
        queries : list of str
            Free-form text labels (e.g. ["person", "monitor", "office chair"]).
        """
        self._queries = list(queries)
        if not queries:
            self._text_emb = None
            return
        prompts = [_PROMPT_TEMPLATE.format(q) for q in queries]
        tokens = self.lang_adaptor.tokenizer(prompts).to(self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            emb = self.lang_adaptor.encode_text(tokens)
        emb = emb.float()
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self._text_emb = emb  # (Q, D_lang)

    @torch.no_grad()
    def segment(self, image_np: np.ndarray) -> dict:
        """Per-pixel similarity for one RGB keyframe.

        Parameters
        ----------
        image_np : np.ndarray
            (H, W, 3) RGB uint8 OR float32 in [0, 1].

        Returns
        -------
        dict with keys
            'similarity' : np.float32 (Q, H, W) — per-query cosine similarity
                           upsampled bilinearly to the original image
                           resolution.
            'best_query' : np.int32 (H, W) — argmax over queries; -inf-thresh
                           handling is left to the caller.
            'best_score' : np.float32 (H, W).
            'queries'    : list of str — for convenience.
        """
        if self._text_emb is None or len(self._queries) == 0:
            raise RuntimeError("set_queries() must be called before segment()")

        # Image → (1, 3, H, W) float in [0, 1].
        # Dtype rules (no silent fallbacks per CLAUDE.md §6):
        #   uint8         → assumed [0, 255], divided by 255.
        #   float, max>1.5→ assumed [0, 255], divided by 255.
        #   float, max≤1.0→ assumed [0, 1], passed through.
        #   float in (1.0, 1.5]: ambiguous — emit a [WARN] and treat as [0, 1].
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

        # RADIO requires patch-aligned dimensions; resize to nearest supported.
        suggested = self.model.get_nearest_supported_resolution(H_orig, W_orig)
        x_res = F.interpolate(
            x, size=(suggested.height, suggested.width),
            mode="bilinear", align_corners=False,
        )
        H_s, W_s = x_res.shape[-2], x_res.shape[-1]
        H_t = H_s // self.patch_size
        W_t = W_s // self.patch_size

        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            out = self.model(x_res)                  # RadioOutput
            tokens = out.features                    # (1, N, 768)
            lang_tokens = self.lang_adaptor.head_mlp(tokens)  # (1, N, D_lang)

        lang_tokens = lang_tokens.float()
        # (1, N, D) → (1, D, H_t, W_t)
        feat_lang = lang_tokens.permute(0, 2, 1).reshape(1, -1, H_t, W_t)

        # Cosine similarity vs cached text embeddings.
        # Flatten to (N, D) for matmul.
        D_lang = feat_lang.shape[1]
        img = feat_lang.permute(0, 2, 3, 1).reshape(-1, D_lang)
        img = img / img.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        sim = (img @ self._text_emb.t()).reshape(1, H_t, W_t, len(self._queries))
        sim = sim.permute(0, 3, 1, 2)  # (1, Q, H_t, W_t)

        # Upsample to original image resolution.
        sim_full = F.interpolate(
            sim, size=(H_orig, W_orig),
            mode="bilinear", align_corners=False,
        )[0]  # (Q, H, W)

        sim_np = sim_full.cpu().numpy().astype(np.float32)
        best_score = sim_np.max(axis=0)
        best_query = sim_np.argmax(axis=0).astype(np.int32)

        return {
            "similarity": sim_np,
            "best_query": best_query,
            "best_score": best_score,
            "queries": list(self._queries),
        }
