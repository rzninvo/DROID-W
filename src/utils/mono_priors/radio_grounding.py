"""
RADIO + SigLIP-2 per-keyframe text grounding.

Port of the RADIO-ViPE paradigm (Yakovlev et al., arXiv 2604.26067, April 2026 —
be2rlab/RADIO-ViPE) for HERMES-SLAM. Includes the upstream quality knobs that
matter for per-pixel discrimination:

  ✓ openai_imagenet_template prompt ensembling (80 templates per query)
  ✓ sliding-window inference (crop=336, stride=224) for richer per-pixel
    receptive field — average overlapping crops in feature space
  ✓ per-pixel softmax over queries with temperature 100

Deliberately NOT ported (cf. thirdparty/RADIO-ViPE/vipe/priors/embedding/
radseg_encoder.py for the reference):
  ✗ SelfCorrelatingRecursiveAttn last-block rewrite — improves spatial
    selectivity but mutates the model in place; deferred until quality is
    insufficient.
  ✗ SAM-based mask refinement — not needed for our test scope.

Pipeline per keyframe:
    RGB image (H, W, 3) uint8
        → torch.hub.load("NVlabs/RADIO", "radio_model", version=...)
        → sliding-window forward → backbone features (B, N, D=768)
        → adaptor['siglip2'].head_mlp → SigLIP-2-aligned (B, N, 1152)
        → cosine vs prompt-ensembled text embeddings → softmax(×100)
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

# OpenAI ImageNet templates (80 entries) — vendored from
# thirdparty/RADIO-ViPE/vipe/priors/embedding/prompt_templates.py. RADIO-ViPE
# uses the full ensemble in encode_labels(); skipping it costs ~3-5% mIoU on
# Replica per their Table II ablation.
_OPENAI_IMAGENET_TEMPLATES = [
    'a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.',
    'a photo of the hard to see {}.', 'a low resolution photo of the {}.',
    'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.',
    'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.',
    'a photo of a hard to see {}.', 'a bright photo of a {}.',
    'a photo of a clean {}.', 'a photo of a dirty {}.',
    'a dark photo of the {}.', 'a drawing of a {}.', 'a photo of my {}.',
    'the plastic {}.', 'a photo of the cool {}.',
    'a close-up photo of a {}.', 'a black and white photo of the {}.',
    'a painting of the {}.', 'a painting of a {}.',
    'a pixelated photo of the {}.', 'a sculpture of the {}.',
    'a bright photo of the {}.', 'a cropped photo of a {}.',
    'a plastic {}.', 'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.',
    'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.',
    'a {} in a video game.', 'a photo of one {}.', 'a doodle of a {}.',
    'a close-up photo of the {}.', 'a photo of a {}.', 'the origami {}.',
    'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.',
    'a origami {}.', 'a low resolution photo of a {}.', 'the toy {}.',
    'a rendition of the {}.', 'a photo of the clean {}.',
    'a photo of a large {}.', 'a rendition of a {}.',
    'a photo of a nice {}.', 'a photo of a weird {}.',
    'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.',
    'a sketch of the {}.', 'a embroidered {}.', 'a pixelated photo of a {}.',
    'itap of the {}.', 'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.', 'a plushie {}.', 'a photo of the nice {}.',
    'a photo of the small {}.', 'a photo of the weird {}.',
    'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.',
    'a photo of the large {}.', 'a black and white photo of a {}.',
    'the plushie {}.', 'a dark photo of a {}.', 'itap of a {}.',
    'graffiti of the {}.', 'a toy {}.', 'itap of my {}.',
    'a photo of a cool {}.', 'a photo of a small {}.',
    'a tattoo of the {}.',
]

# Sliding-window inference parameters. Match radseg_encoder defaults.
_DEFAULT_SLIDE_CROP = 336
_DEFAULT_SLIDE_STRIDE = 224


class RadioGrounder:
    """Loads RADIO + SigLIP-2 once, exposes per-keyframe text grounding."""

    def __init__(
        self,
        device: str = "cuda:0",
        radio_version: str = _DEFAULT_RADIO_VERSION,
        lang_adaptor: str = _DEFAULT_LANG_ADAPTOR,
        amp: bool = True,
        sliding_window: bool = True,
        slide_crop: int = _DEFAULT_SLIDE_CROP,
        slide_stride: int = _DEFAULT_SLIDE_STRIDE,
        prompt_ensemble: bool = True,
    ):
        self.device = device
        self.radio_version = radio_version
        self.lang_adaptor_name = lang_adaptor
        self.amp = amp
        self.sliding_window = sliding_window
        self.slide_crop = slide_crop
        self.slide_stride = slide_stride
        self.prompt_ensemble = prompt_ensemble

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

        Implementation: per query, run all 80 OpenAI ImageNet templates
        through the SigLIP-2 text encoder, L2-normalise each, mean across
        templates, L2-normalise again. This matches radseg_encoder.encode_labels
        (lines 266-277) and is materially better than a single template at
        per-pixel cosine.
        """
        self._queries = list(queries)
        if not queries:
            self._text_emb = None
            return

        templates = _OPENAI_IMAGENET_TEMPLATES if self.prompt_ensemble else ["a photo of a {}."]
        T = len(templates)
        Q = len(queries)
        # Build (Q*T,) prompts in (q, t) row-major order.
        prompts = [tmpl.format(q) for q in queries for tmpl in templates]
        tokens = self.lang_adaptor.tokenizer(prompts).to(self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            emb = self.lang_adaptor.encode_text(tokens)  # (Q*T, D)
        emb = emb.float()
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        emb = emb.view(Q, T, -1).mean(dim=1)             # (Q, D) — average templates
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self._text_emb = emb

    @torch.no_grad()
    def segment(
        self,
        image_np: np.ndarray,
        softmax_temperature: float = 100.0,
    ) -> dict:
        """Per-pixel similarity + softmax-over-queries for one RGB keyframe.

        Parameters
        ----------
        image_np : np.ndarray
            (H, W, 3) RGB uint8 OR float32 in [0, 1].
        softmax_temperature : float
            Scale applied before per-pixel softmax over queries. Matches
            RADIO-ViPE's compute_cos_sim(softmax=True) which uses 100. Higher
            temperature → sharper distribution; the winner dominates.

        Returns
        -------
        dict with keys
            'similarity' : np.float32 (Q, H, W) — raw cosine similarity per
                           query, upsampled bilinearly to original resolution.
            'softmax'    : np.float32 (Q, H, W) — softmax over queries,
                           same upsample. Use this for thresholding when you
                           want "this query wins by a margin".
            'best_query' : np.int32 (H, W) — argmax over queries.
            'best_score' : np.float32 (H, W) — softmax probability of the
                           winning query (range [0, 1]).
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

        if self.sliding_window:
            feat_map, H_t, W_t = self._sliding_inference(x)
        else:
            feat_map, H_t, W_t = self._single_inference(x)

        # head_mlp expects (B, N, D_in); our feat_map is (B, D_in, H_t, W_t).
        B, D_in, _, _ = feat_map.shape
        tokens = feat_map.permute(0, 2, 3, 1).reshape(B, -1, D_in)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            lang_tokens = self.lang_adaptor.head_mlp(tokens)  # (B, N, D_lang)
        lang_tokens = lang_tokens.float()
        feat_lang = lang_tokens.permute(0, 2, 1).reshape(B, -1, H_t, W_t)

        # Cosine similarity vs cached text embeddings.
        # Flatten to (N, D) for matmul.
        D_lang = feat_lang.shape[1]
        img = feat_lang.permute(0, 2, 3, 1).reshape(-1, D_lang)
        img = img / img.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        sim = (img @ self._text_emb.t()).reshape(1, H_t, W_t, len(self._queries))
        sim = sim.permute(0, 3, 1, 2)  # (1, Q, H_t, W_t)

        # Per-pixel softmax over queries at low resolution (cheaper than
        # softmax at full image res). Temperature scaling matches RADIO-ViPE's
        # compute_cos_sim(softmax=True) which uses 100. Without this, pixels
        # whose winning class barely edges out the runner-up will paint
        # whichever class they prefer; with this, only confidently-classified
        # pixels survive the post-softmax threshold.
        prob = F.softmax(sim * float(softmax_temperature), dim=1)

        # Upsample BOTH similarity and softmax probability to original image
        # resolution. Bilinear on similarity is fine; bilinear on softmax
        # breaks the per-pixel sum-to-1 invariant slightly but is what the
        # upstream radseg_encoder._get_seg_logits also does (line 316–317).
        sim_full = F.interpolate(
            sim, size=(H_orig, W_orig),
            mode="bilinear", align_corners=False,
        )[0]
        prob_full = F.interpolate(
            prob, size=(H_orig, W_orig),
            mode="bilinear", align_corners=False,
        )[0]

        sim_np = sim_full.cpu().numpy().astype(np.float32)
        prob_np = prob_full.cpu().numpy().astype(np.float32)
        best_score = prob_np.max(axis=0)
        best_query = prob_np.argmax(axis=0).astype(np.int32)

        return {
            "similarity": sim_np,
            "softmax": prob_np,
            "best_query": best_query,
            "best_score": best_score,
            "queries": list(self._queries),
        }

    # ------------------------------------------------------------------
    # Internal: feature extraction helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _single_inference(self, x: torch.Tensor):
        """Single forward pass at RADIO's nearest supported resolution.

        Returns
        -------
        feat_map : (1, 768, H_t, W_t) — backbone features (pre-head_mlp).
        H_t, W_t : token grid dimensions (image_size // 16).
        """
        H_orig, W_orig = x.shape[-2], x.shape[-1]
        suggested = self.model.get_nearest_supported_resolution(H_orig, W_orig)
        x_res = F.interpolate(
            x, size=(suggested.height, suggested.width),
            mode="bilinear", align_corners=False,
        )
        H_s, W_s = x_res.shape[-2], x_res.shape[-1]
        H_t = H_s // self.patch_size
        W_t = W_s // self.patch_size
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            out = self.model(x_res)                          # RadioOutput
        feat = out.features.float()                          # (1, N, 768)
        feat_map = feat.permute(0, 2, 1).reshape(1, -1, H_t, W_t)
        return feat_map, H_t, W_t

    @torch.no_grad()
    def _sliding_inference(self, x: torch.Tensor):
        """Sliding-window forward — averages backbone features over overlapping
        crops. Direct port of radseg_encoder._sliding_inference (lines
        538-573); we keep just the parts we need (no SCGA, no SAM).

        Returns
        -------
        feat_map : (1, 768, H_t, W_t) — backbone features at H_img/16, W_img/16.
        H_t, W_t : token grid dimensions of the crop-padded input.
        """
        crop = self.slide_crop
        stride = self.slide_stride
        ps = self.patch_size

        # Pad input up to the nearest patch-aligned size that's at least crop.
        H_orig, W_orig = x.shape[-2], x.shape[-1]
        H_pad = max(H_orig, crop)
        W_pad = max(W_orig, crop)
        if H_pad % ps != 0:
            H_pad = (H_pad // ps + 1) * ps
        if W_pad % ps != 0:
            W_pad = (W_pad // ps + 1) * ps
        if (H_pad, W_pad) != (H_orig, W_orig):
            x = F.interpolate(
                x, size=(H_pad, W_pad),
                mode="bilinear", align_corners=False,
            )

        h_grids = max(H_pad - crop + stride - 1, 0) // stride + 1
        w_grids = max(W_pad - crop + stride - 1, 0) // stride + 1
        H_t = H_pad // ps
        W_t = W_pad // ps

        # Collect crops and the (y1,x1,y2,x2) where each one lives in the
        # token grid.
        crop_imgs = []
        patch_locs = []
        for hi in range(h_grids):
            for wi in range(w_grids):
                y1 = hi * stride
                x1 = wi * stride
                y2 = min(y1 + crop, H_pad)
                x2 = min(x1 + crop, W_pad)
                y1 = max(y2 - crop, 0)
                x1 = max(x2 - crop, 0)
                # Patch-align (the upstream asserts these are %ps==0; with
                # our patch-aligned padding that's true by construction).
                crop_imgs.append(x[:, :, y1:y2, x1:x2])
                patch_locs.append((y1 // ps, x1 // ps, y2 // ps, x2 // ps))

        batched = torch.cat(crop_imgs, dim=0)  # (n_crops, 3, crop, crop)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            out = self.model(batched)
        feat = out.features.float()                          # (n_crops, N_c, 768)
        # Each crop's tokens reshape to a (D, h_c, w_c) feature map.
        n_crops, N_c, D = feat.shape
        h_c = crop // ps
        w_c = crop // ps
        per_crop = feat.permute(0, 2, 1).reshape(n_crops, D, h_c, w_c)

        # Accumulate into the full token-grid map, average overlaps.
        feat_map = torch.zeros(
            (1, D, H_t, W_t), dtype=per_crop.dtype, device=per_crop.device,
        )
        count_map = torch.zeros(
            (1, 1, H_t, W_t), dtype=per_crop.dtype, device=per_crop.device,
        )
        for k, (y1, x1, y2, x2) in enumerate(patch_locs):
            feat_map[:, :, y1:y2, x1:x2] += per_crop[k:k+1]
            count_map[:, :, y1:y2, x1:x2] += 1
        feat_map = feat_map / count_map.clamp_min(1)
        return feat_map, H_t, W_t
