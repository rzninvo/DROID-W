"""Vendored RADIO-ViPE per-keyframe segmentation encoder.

Source: https://github.com/be2rlab/RADIO-ViPE/tree/main/vipe/priors/embedding
Files: radseg_encoder.py, base.py, sam_utils.py, prompt_templates.py
Cite : Yakovlev et al., arXiv 2604.26067, April 2026
License: Apache 2.0 (NVIDIA-attributed in upstream LICENSE)

Patches applied during vendoring (see git history for reasoning):
  - radseg_encoder.py: relative imports; torch.hub local-path → github;
    default lang_model 'siglip2-g' → 'siglip2' (current pretrained weights).
  - base.py: vipe.priors.embedding.prompt_templates → relative import.
  - sam_utils.py, prompt_templates.py: verbatim.

Public API: import RADSegEncoder.
"""

from .radseg_encoder import RADSegEncoder

__all__ = ["RADSegEncoder"]
