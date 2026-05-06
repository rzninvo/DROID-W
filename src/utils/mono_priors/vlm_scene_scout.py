"""
VLM-based scene scout for open-vocabulary object discovery.

Runs a Vision-Language Model (Qwen2.5-VL by default) on keyframes to discover
what objects exist in the scene. The discovered class names are fed to
YOLO-World's set_classes() so detection is truly open-vocabulary — no fixed
class list needed.

Design:
    - Runs asynchronously on a background thread (doesn't block SLAM)
    - Queries every N keyframes (configurable)
    - Maintains a growing set of discovered classes
    - Only triggers YOLO-World set_classes() when new classes are found
    - First keyframe is always queried to bootstrap the class list

Usage (real-time, during SLAM):
    scout = VLMSceneScout(cfg, device="cuda:0")
    scout.start()
    ...
    # In tracking loop, feed keyframes:
    scout.submit_keyframe(image_np, keyframe_idx)
    ...
    # In detection, get latest classes:
    classes = scout.get_classes()
    if scout.classes_changed():
        model.set_classes(classes)

Usage (post-processing, on saved keyframes):
    scout = VLMSceneScout(cfg, device="cuda:0")
    classes = scout.query_frame(image_np)
"""

import os
import json
import threading
import queue
import logging
from typing import List, Optional, Set

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _as_feature_tensor(out) -> torch.Tensor:
    """Coerce HF / OpenCLIP encoder outputs to a (B, D) feature tensor.

    transformers 5.x sometimes returns BaseModelOutputWithPooling from
    Siglip2Model.get_{text,image}_features instead of a bare tensor. Pick
    the pooled / image-embeds field when available, fall back to last
    hidden state mean-pool, else raise.
    """
    if torch.is_tensor(out):
        return out
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        if hasattr(out, attr) and getattr(out, attr) is not None:
            t = getattr(out, attr)
            if torch.is_tensor(t):
                return t
    if hasattr(out, "last_hidden_state") and torch.is_tensor(out.last_hidden_state):
        # Mean-pool over the sequence dim as a last resort.
        return out.last_hidden_state.mean(dim=1)
    raise RuntimeError(
        f"[vlm_scout] unexpected encoder output type: {type(out).__name__}"
    )


# Fallback classes if VLM fails or is disabled — minimal seed list
SEED_CLASSES = ["person", "car", "chair", "table", "door"]

# Classes to filter out — backgrounds, clothing, body parts, colors, materials
IGNORE_CLASSES = {
    # Background/surfaces
    "floor", "ceiling", "wall", "ground", "sky", "background",
    "shadow", "light", "air", "space", "none", "nothing", "road",
    "sidewalk", "pavement", "grass", "dirt", "concrete", "asphalt",
    # Clothing (competes with "person" in YOLO-World). Include singular forms
    # too because the parser's plural-stripping turns "jeans" into "jean".
    "shirt", "tshirt", "t-shirt", "jacket", "coat", "pants", "pant",
    "jeans", "jean", "shorts", "short", "dress", "skirt", "hat", "cap",
    "helmet", "shoe", "shoes", "boot", "boots", "sneakers", "sneaker",
    "hoodie", "sweater", "vest", "scarf", "glove", "gloves", "sock",
    "socks", "mask", "glasses", "sunglasses",
    # Body parts
    "hand", "hands", "arm", "arms", "leg", "legs", "head", "face",
    "foot", "feet", "hair", "finger", "fingers",
    # NOTE: gendered/age variants (man, woman, boy, girl, child, lady, etc.)
    # are intentionally NOT filtered here — they are REMAPPED to "person" via
    # CLASS_ALIASES below. Filtering them used to lose all humans when the
    # VLM preferred specific terms over the generic "person".
}

# Alias map: whatever the VLM returns on the left gets rewritten to the canonical
# name on the right before any filtering / dedup. This captures common synonyms
# and gendered/age variants without losing the detection. (IGNORE_CLASSES is
# applied AFTER this rewrite.)
CLASS_ALIASES = {
    # All humans → "person"
    "man": "person", "woman": "person", "boy": "person", "girl": "person",
    "child": "person", "kid": "person", "baby": "person", "toddler": "person",
    "lady": "person", "gentleman": "person", "male": "person", "female": "person",
    "pedestrian": "person", "human": "person", "people": "person",
    "adult": "person", "teenager": "person", "guy": "person",
    # Common synonyms (some also get merged by CLIP dedup, but this is faster+explicit)
    "television": "tv", "tele": "tv",
    "settee": "couch",
    "trashcan": "trash can", "garbage can": "trash can", "bin": "trash can",
}

# Words that indicate an attribute description, not an object
ATTRIBUTE_WORDS = {
    "red", "blue", "green", "yellow", "white", "black", "brown", "gray",
    "grey", "pink", "orange", "purple", "dark", "light", "bright",
    "large", "small", "big", "tall", "short", "long", "old", "new",
    "left", "right", "front", "back", "top", "bottom", "wooden", "metal",
    "plastic", "striped", "colored", "coloured",
}

# VLM prompt for object discovery — multi-pass + JSON to disambiguate
# CLIP-near-duplicates ("tv" vs "monitor") at the source. Asking for compound
# nouns ("computer monitor", "office chair") pushes embeddings apart so the
# CLIP-similarity dedup pass needs to fire less often.
DISCOVERY_PROMPT = (
    "Analyze this image and reply ONLY with a JSON object.\n"
    "Step 1 — In one short phrase, identify the scene type "
    '(e.g. "office", "kitchen", "outdoor street", "lecture hall").\n'
    "Step 2 — Exhaustively list EVERY distinct object you can see. "
    "Aim for 20–40 objects. Check every corner of the image.\n"
    "RULES:\n"
    '  • Call every human "person". NEVER use "man", "woman", "boy", "girl", '
    '"lady", "gentleman", "pedestrian" — use "person" for all of them.\n'
    "  • Use SPECIFIC compound nouns to disambiguate similar categories:\n"
    '      "computer monitor" (not "monitor" or "tv")\n'
    '      "office chair"    (not "chair")\n'
    '      "coffee mug"      (not "cup")\n'
    '      "desk lamp"       (not "lamp")\n'
    "  • Include ALL of these categories if visible:\n"
    "      - furniture: chair, desk, table, shelf, cabinet, drawer, bookcase, bed, couch\n"
    "      - electronics: monitor, laptop, keyboard, mouse, phone, speaker, camera, charger, router\n"
    "      - containers: cup, mug, bottle, bowl, plate, jar, box, bag, backpack, basket, bin\n"
    "      - stationery: book, notebook, paper, pen, pencil, marker, clipboard, folder, stapler\n"
    "      - structural: door, window, wall socket, light switch, radiator, vent, pipe, beam\n"
    "      - decor: plant, poster, painting, photo frame, clock, flag, sign\n"
    "      - wires/cables, cords, headphones, and any tiny item on any surface\n"
    "  • Lowercase common nouns only — no colors, sizes, materials, or adjectives.\n"
    "  • Do NOT use vague phrases like 'items', 'objects', 'elements', 'things'.\n"
    'Reply EXACTLY as JSON, no extra text:\n'
    '{"scene": "<scene type>", "objects": ["<obj1>", "<obj2>", ...]}'
)


def _load_vlm(model_name: str, device: str):
    """
    Load a VLM for scene understanding.

    Supports Qwen2.5-VL models via transformers.
    """
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    logger.info(f"Loading VLM: {model_name} on {device}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(model_name)

    return model, processor


def _query_vlm(model, processor, image_np: np.ndarray, device: str) -> str:
    """
    Query the VLM with an image and return raw text response.

    Args:
        model: Loaded VLM model.
        processor: VLM processor/tokenizer.
        image_np: uint8 numpy array (H, W, 3) RGB.
        device: Device string.

    Returns:
        Raw text response from the VLM.
    """
    from PIL import Image

    pil_image = Image.fromarray(image_np)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": DISCOVERY_PROMPT},
            ],
        }
    ]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_input],
        images=[pil_image],
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    # Decode only the generated tokens (skip input)
    generated = output_ids[0, inputs.input_ids.shape[1]:]
    response = processor.tokenizer.decode(generated, skip_special_tokens=True)

    return response.strip()


def parse_vlm_response(response: str) -> List[str]:
    """
    Parse the VLM response into clean class names.

    Tries the new JSON output format first ({"scene": "...", "objects": [...]});
    falls back to the legacy comma-list parser if the response is unstructured.
    Strips color/size adjectives ("blue tshirt" → "tshirt") and filters
    IGNORE_CLASSES.

    Args:
        response: Raw text from VLM.

    Returns:
        List of cleaned, lowercase class names. Compound nouns are preserved
        ("office chair" stays as "office chair").
    """
    import re
    import json

    raw_objects: List[str] = []
    scene_name: Optional[str] = None

    # Try JSON path first — robust to text wrapping the JSON block
    json_match = re.search(r"\{.*\}", response, flags=re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            if isinstance(obj, dict):
                if isinstance(obj.get("objects"), list):
                    raw_objects = [str(x) for x in obj["objects"]]
                if isinstance(obj.get("scene"), str):
                    scene_name = obj["scene"].strip().lower()
        except (ValueError, TypeError):
            pass

    # Legacy comma-list fallback
    if not raw_objects:
        response = re.sub(r'\d+[\.\)]\s*', '', response)
        response = re.sub(r'[-*•]\s*', '', response)
        response = response.replace('"', '').replace("'", "")
        raw_objects = re.split(r'[,\n]', response)

    # Split entries that smuggled multiple items in via "/" (e.g. "wires/cables")
    split_raw = []
    for part in raw_objects:
        s = str(part)
        if "/" in s:
            split_raw.extend(x.strip() for x in s.split("/") if x.strip())
        else:
            split_raw.append(s)
    raw_objects = split_raw

    classes = []
    for part in raw_objects:
        name = str(part).strip().lower().rstrip('.').rstrip(',').strip()

        # If the VLM emits "structural elements (door, window)", pull out
        # whatever is inside the parentheses as separate classes AND strip
        # the outer phrase. Also handles "items (cable)" → ["cable"].
        paren_match = re.search(r"\(([^)]+)\)", name)
        if paren_match:
            inner = paren_match.group(1)
            # Drop the outer phrase; feed the inner items back through the loop.
            inner_parts = [p.strip() for p in re.split(r"[,;/]", inner) if p.strip()]
            raw_objects.extend(inner_parts)
            # Also consider the name with the parenthetical removed
            name = re.sub(r"\s*\([^)]+\)", "", name).strip()
            if not name:
                continue

        # Skip empty or overly long entries (VLM hallucination)
        if not name or len(name) >= 40 or len(name) <= 1:
            continue

        # Skip JSON-syntax leakage (happens when VLM emits malformed JSON and
        # the legacy parser then picks up keys as objects)
        if any(tok in name for tok in (':', '{', '}', '[', ']')):
            continue
        if name.startswith(('scene', 'objects', 'name', 'type', 'label')) and ' ' not in name[:6]:
            # "scene: office" or "objects" alone — JSON key, not an object
            # But preserve genuine multi-word labels that start with these words
            if name in ('scene', 'objects', 'name', 'type', 'label'):
                continue

        # Don't accept the scene type as an object
        if scene_name and name == scene_name:
            continue

        # Don't accept vague/category-header echoes of the prompt.
        PROMPT_ECHOES = {
            # Generic
            "structural elements", "small items", "large objects", "small objects",
            "items", "item", "elements", "element", "things", "thing", "objects",
            "object", "stuff", "tiny item", "tiny items",
            # Category headers we list in the prompt that the VLM sometimes
            # regurgitates as entries of their own.
            "furniture", "electronics", "containers", "container",
            "stationery", "structural", "decor", "decoration", "decorations",
        }
        if name in PROMPT_ECHOES:
            continue

        # Strip attribute words (e.g., "blue tshirt" → "tshirt"). Compound
        # nouns like "office chair" are preserved because none of their words
        # are attribute words.
        words = name.split()
        words = [w for w in words if w not in ATTRIBUTE_WORDS]
        name = " ".join(words).strip()

        if not name:
            continue

        # Conservative singularization — strip trailing 's' on the head noun
        # when the remainder is a clearly valid singular form. Avoid false
        # positives like "glass"→"glas" by keeping a stop-list. Applied to
        # the LAST word only (so "photo frames" → "photo frame").
        STOP_SINGULARIZE = {
            "glass", "class", "dress", "press", "bus", "this", "lens",
            "cross", "plus", "canvas",
        }
        parts = name.split()
        if parts:
            tail = parts[-1]
            if (len(tail) >= 4 and tail.endswith("s") and not tail.endswith("ss")
                    and tail not in STOP_SINGULARIZE):
                # 'es' suffix: boxes→box, brushes→brush, watches→watch
                if tail.endswith("es") and tail[-3] in "xs" or tail.endswith("ches") or tail.endswith("shes"):
                    parts[-1] = tail[:-2]
                else:
                    parts[-1] = tail[:-1]
                name = " ".join(parts)

        # Remap synonyms / gendered terms to canonical names BEFORE filtering
        # (e.g. "man" → "person", "television" → "tv"). This is why people
        # are preserved even when the VLM prefers gendered words.
        name = CLASS_ALIASES.get(name, name)

        # Skip if the cleaned name is in IGNORE_CLASSES
        if name in IGNORE_CLASSES:
            continue

        # Skip if any word in the name is clothing/body part
        if any(w in IGNORE_CLASSES for w in name.split()):
            continue

        classes.append(name)

    # Deduplicate while preserving order
    seen, out = set(), []
    for c in classes:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


class VLMSceneScout:
    """
    Asynchronous VLM-based object class discovery for open-vocabulary detection.

    Runs a VLM on selected keyframes in a background thread to discover what
    objects exist in the scene. Updates a shared class list that YOLO-World
    reads from.
    """

    def __init__(self, cfg: dict, device: str = "cuda:0"):
        vlm_cfg = cfg.get("vlm_scout", {})

        self.model_name = vlm_cfg.get("model", "Qwen/Qwen2.5-VL-3B-Instruct")
        self.device = device
        self.query_interval = vlm_cfg.get("query_interval", 10)
        self.warmup_keyframes = vlm_cfg.get("warmup_keyframes", 5)
        self.saturation_patience = vlm_cfg.get("saturation_patience", 5)
        self.enabled = vlm_cfg.get("activate", False)
        self.max_classes = vlm_cfg.get("max_classes", 200)

        # SOTA pattern (ConceptGraphs / HOV-SG / CLIO / Mosaic3D / OVO-SLAM):
        # discover the vocabulary entirely from the VLM. The DEFAULT_CLASSES
        # seed is kept ONLY as a fallback for runs where the VLM is disabled.
        self.use_default_seed = vlm_cfg.get("use_default_seed", False)

        # CLIP-similarity vocabulary deduplication ("tv" ≈ "monitor", ...)
        # Default 0.90: calibrated on ViT-B-32 with real office-scene classes.
        # At 0.85 unrelated items collide ("book"-"shelf"=0.867, "window"-"door"=0.880).
        # At 0.90 only true synonyms merge (tv-monitor=0.902, couch-sofa=0.915,
        # computer-monitor-monitor=0.933). ConceptGraphs uses 0.8 on ViT-H-14
        # which has tighter embeddings — our ViT-B-32 needs a stricter threshold.
        self.dedupe_clip_threshold = vlm_cfg.get("dedupe_clip_threshold", 0.90)
        self.dedupe_clip_model = vlm_cfg.get("dedupe_clip_model", "ViT-B-32")
        # SigLIP 2 backend for the same dedup task. Sigmoid-loss image-text
        # encoder; cosine distribution differs from CLIP, so threshold must
        # be re-calibrated (start ~0.80; higher in dev).
        self.dedupe_text_encoder = str(vlm_cfg.get("dedupe_text_encoder", "clip")).lower()
        if self.dedupe_text_encoder not in ("clip", "siglip2"):
            raise ValueError(
                f"dedupe_text_encoder must be 'clip' or 'siglip2', "
                f"got {self.dedupe_text_encoder!r}"
            )
        self.dedupe_siglip2_model = vlm_cfg.get(
            "dedupe_siglip2_model", "google/siglip2-large-patch16-256"
        )
        self._clip_dedup = None        # lazy-loaded (CLIP model, tokenizer)
        self._siglip2_dedup = None     # lazy-loaded (SigLIP 2 model, processor)

        # Adaptive-gating state — count queries since last vocabulary growth.
        self._queries_since_growth = 0

        # Thread-safe class storage. Seed only if explicitly requested or if
        # VLM discovery is disabled (otherwise we'd fight the VLM's choices).
        # Even without the seed, we guarantee "person" is present — this is
        # a critical safety class for any scene and can silently fail to be
        # discovered if the VLM prefers gendered terms that don't survive
        # parsing in older configs.
        self._lock = threading.Lock()
        if self.use_default_seed or not self.enabled:
            from src.utils.mono_priors.seg_model import DEFAULT_CLASSES
            self._classes: Set[str] = {c.lower() for c in DEFAULT_CLASSES}
        else:
            self._classes: Set[str] = {"person"}
        self._classes_changed = True
        self._class_list_version = 1

        # Background processing
        self._queue: queue.Queue = queue.Queue(maxsize=5)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Model (lazy loaded)
        self._model = None
        self._processor = None
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy-load the VLM model on first use."""
        if not self._loaded:
            self._model, self._processor = _load_vlm(self.model_name, self.device)
            self._loaded = True

    def query_frame(self, image_np: np.ndarray) -> List[str]:
        """
        Synchronously query a single frame and return discovered classes.

        Also updates the internal class set. Use this for post-processing
        or for the first keyframe.

        Args:
            image_np: uint8 numpy array (H, W, 3) RGB.

        Returns:
            List of newly discovered class names from this frame.
        """
        self._ensure_loaded()

        response = _query_vlm(self._model, self._processor, image_np, self.device)
        discovered = parse_vlm_response(response)

        new_classes = []
        with self._lock:
            for cls in discovered:
                if cls in IGNORE_CLASSES:
                    continue
                if cls not in self._classes and len(self._classes) < self.max_classes:
                    self._classes.add(cls)
                    new_classes.append(cls)
            if new_classes:
                had_person = "person" in self._classes
                # CLIP-merge any near-duplicates introduced by this batch
                # (e.g. discovered "monitor" while seed already had "computer
                # monitor"). Keeps the canonical name per cluster.
                deduped = set(self.dedupe_classes(sorted(self._classes)))
                # Safety: "person" must stay in the vocabulary once it's
                # been added. Dedup has been observed to transiently merge
                # it into a multi-word neighbour ("jean", "laptop", ...)
                # during early rounds and then never restore it. Missing
                # "person" breaks dynamic-object SLAM downstream.
                if had_person:
                    deduped.add("person")
                if deduped != self._classes:
                    self._classes = deduped
                self._classes_changed = True
                self._class_list_version += 1
                self._queries_since_growth = 0
            else:
                self._queries_since_growth += 1

        if new_classes:
            logger.info(f"VLM discovered {len(new_classes)} new classes: {new_classes}")

        return new_classes

    def _ensure_clip_dedup_loaded(self):
        """Lazy-load OpenCLIP for class-list deduplication. Loaded once."""
        if self._clip_dedup is not None:
            return
        try:
            import open_clip
        except ImportError:
            logger.warning("open_clip not available — CLIP dedup disabled")
            self._clip_dedup = (None, None)
            return
        model, _, _ = open_clip.create_model_and_transforms(
            self.dedupe_clip_model, pretrained="openai"
        )
        model.eval().to(self.device)
        tokenizer = open_clip.get_tokenizer(self.dedupe_clip_model)
        self._clip_dedup = (model, tokenizer)

    def _ensure_siglip2_dedup_loaded(self):
        """Lazy-load SigLIP 2 for class-list deduplication. Requires
        transformers >= 4.49 (Feb 2025)."""
        if self._siglip2_dedup is not None:
            return
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError:
            logger.warning(
                "[WARN] vlm_scout dedup: expected transformers>=4.49, got missing — "
                "fallback=skip dedup (SigLIP 2 disabled)"
            )
            self._siglip2_dedup = (None, None)
            return
        model = AutoModel.from_pretrained(self.dedupe_siglip2_model)
        model.eval().to(self.device)
        processor = AutoProcessor.from_pretrained(self.dedupe_siglip2_model)
        self._siglip2_dedup = (model, processor)

    @torch.no_grad()
    def _encode_classes_for_dedup(self, classes: List[str]) -> Optional[torch.Tensor]:
        """Encode a list of class names to L2-normed text embeddings using
        whichever backend is configured. Returns None if the backend failed
        to load (caller should skip dedup)."""
        if self.dedupe_text_encoder == "siglip2":
            self._ensure_siglip2_dedup_loaded()
            model, processor = self._siglip2_dedup
            if model is None:
                return None
            inputs = processor(
                text=list(classes),
                padding="max_length",
                max_length=64,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            emb = model.get_text_features(**inputs)
        else:
            self._ensure_clip_dedup_loaded()
            model, tokenizer = self._clip_dedup
            if model is None:
                return None
            tokens = tokenizer(list(classes)).to(self.device)
            emb = model.encode_text(tokens)
        emb = _as_feature_tensor(emb)
        return emb / emb.norm(dim=-1, keepdim=True)

    def dedupe_classes(
        self, classes: List[str], threshold: Optional[float] = None
    ) -> List[str]:
        """
        Merge CLIP-near-duplicate class names. ConceptGraphs / HOV-SG style.

        Pairs of names with cosine similarity ≥ threshold are clustered
        (union-find). The most-specific name per cluster wins (token count,
        then string length).

        Examples (threshold ≈ 0.88, ViT-B-32):
            ["tv", "monitor", "computer monitor"]   → ["computer monitor"]
            ["couch", "sofa"]                       → ["couch"] or ["sofa"]
            ["mug", "coffee mug", "cup"]            → ["coffee mug"]

        If open_clip isn't installed or the input has < 2 entries, returns
        the input unchanged.
        """
        if not classes or len(classes) < 2:
            return list(classes)
        thr = float(threshold if threshold is not None else self.dedupe_clip_threshold)

        emb = self._encode_classes_for_dedup(classes)
        if emb is None:
            return list(classes)
        sim = (emb @ emb.T).cpu().numpy()

        # Union-find clustering on CLIP cosine
        parent = list(range(len(classes)))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                if sim[i, j] >= thr:
                    a, b = find(i), find(j)
                    if a != b:
                        parent[b] = a

        clusters: dict = {}
        for i, c in enumerate(classes):
            clusters.setdefault(find(i), []).append(c)

        canonical = []
        for members in clusters.values():
            # Most specific = most tokens, longest string as tiebreak
            members.sort(key=lambda s: (-len(s.split()), -len(s), s))
            canonical.append(members[0])
            if len(members) > 1:
                logger.info(f"VLM dedup: merged {members} → '{members[0]}'")
        return sorted(canonical)

    def get_classes(self) -> List[str]:
        """Get current discovered class list (thread-safe)."""
        with self._lock:
            return sorted(self._classes)

    def classes_changed(self) -> bool:
        """Check if classes changed since last call. Resets the flag."""
        with self._lock:
            changed = self._classes_changed
            self._classes_changed = False
            return changed

    def get_version(self) -> int:
        """Get class list version number (increments on each update)."""
        with self._lock:
            return self._class_list_version

    def seed_classes(self, classes: List[str]):
        """
        Seed the class list with initial classes (e.g., from config or defaults).
        Called before start() to provide a baseline vocabulary.
        """
        with self._lock:
            for cls in classes:
                self._classes.add(cls.lower().strip())
            self._classes_changed = True
            self._class_list_version += 1

    def should_query(self, keyframe_idx: int) -> bool:
        """
        Adaptive gating: aggressive sampling early, throttle once the
        vocabulary saturates. Uses warmup_keyframes + saturation_patience
        on top of query_interval.
        """
        if keyframe_idx < self.warmup_keyframes:
            return True
        if self._queries_since_growth >= self.saturation_patience:
            return False
        return keyframe_idx % self.query_interval == 0

    def submit_keyframe(self, image_np: np.ndarray, keyframe_idx: int):
        """
        Submit a keyframe for async VLM processing.

        Gated by should_query() — queries every keyframe during warmup,
        every `query_interval` after, and stops once vocabulary saturates.
        Non-blocking — drops frame if queue is full.
        """
        if not self.enabled:
            return

        if not self.should_query(keyframe_idx):
            return

        try:
            self._queue.put_nowait((image_np.copy(), keyframe_idx))
        except queue.Full:
            logger.debug(f"VLM scout queue full, skipping keyframe {keyframe_idx}")

    def start(self):
        """Start the background VLM processing thread."""
        if not self.enabled:
            logger.info("VLM scene scout disabled")
            return

        self._ensure_loaded()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="vlm-scout"
        )
        self._thread.start()
        logger.info(
            f"VLM scene scout started (model={self.model_name}, "
            f"interval={self.query_interval})"
        )

    def stop(self):
        """Stop the background thread and release resources."""
        self._stop_event.set()
        if self._thread is not None:
            # Unblock the queue
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            self._thread.join(timeout=10)
            self._thread = None

    def _worker_loop(self):
        """Background thread: process keyframes from queue."""
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:  # poison pill
                break

            image_np, kf_idx = item
            try:
                new = self.query_frame(image_np)
                if new:
                    logger.info(
                        f"[KF {kf_idx}] VLM found new classes: {new} "
                        f"(total: {len(self.get_classes())})"
                    )
            except Exception as e:
                logger.warning(f"VLM scout error on keyframe {kf_idx}: {e}")

    def classify_movability(self, classes: List[str], batch_size: int = 80) -> dict:
        """
        Ask the VLM to label each class with a movability score in [0, 1]:
            0.0 = permanently fixed (wall, floor, ceiling)
            0.5 = moved by a person (chair, cup, laptop)
            1.0 = moves on its own (person, car, dog)

        This replaces a hardcoded SEMANTIC_DYNAMIC_CLASSES set — the VLM
        labels whatever was discovered. One batched call per run.
        """
        import re

        if not classes:
            return {}

        self._ensure_loaded()
        result: dict = {}

        for start in range(0, len(classes), batch_size):
            chunk = classes[start:start + batch_size]
            listing = "\n".join(f"- {c}" for c in chunk)
            prompt = (
                "For each object class below, output a movability score from 0.0 to 1.0.\n"
                "0.0 = permanently fixed (wall, floor, ceiling, building, pipe, mountain).\n"
                "0.5 = can be moved by a person (chair, cup, laptop, book, bag, box).\n"
                "1.0 = moves on its own — ALL animals and ALL vehicles "
                "(person, car, bicycle, dog, cat, horse, cow, sheep, bird, elephant, "
                "giraffe, zebra, lion, monkey, fish, deer, bear, motorcycle, bus, train, plane, balloon).\n"
                "When in doubt for an animal or vehicle, choose 1.0.\n"
                "Reply with exactly one line per class in the form:\n"
                "  class: score\n"
                "No extra text. Here are the classes:\n"
                f"{listing}"
            )

            # Text-only query path (reuse processor without an image)
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text_input = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[text_input], return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                output_ids = self._model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            generated = output_ids[0, inputs.input_ids.shape[1]:]
            response = self._processor.tokenizer.decode(generated, skip_special_tokens=True)

            for line in response.splitlines():
                m = re.match(r"\s*[-*•\d\.\)]*\s*(.+?)\s*[:\-]\s*([0-9]*\.?[0-9]+)", line)
                if not m:
                    continue
                name = m.group(1).strip().lower().rstrip(".")
                try:
                    score = float(m.group(2))
                except ValueError:
                    continue
                score = max(0.0, min(1.0, score))
                if name in chunk:
                    result[name] = score

        # Default any missing classes to 0.5 (unknown → middling prior)
        for c in classes:
            result.setdefault(c, 0.5)

        logger.info(f"Classified movability for {len(classes)} classes")
        return result

    def save_movability(self, output_dir: str, movability: dict):
        """Save per-class movability scores for post-processing to reuse."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "vlm_movability.json")
        with open(path, "w") as f:
            json.dump(movability, f, indent=2)
        logger.info(f"Saved movability for {len(movability)} classes to {path}")

    @staticmethod
    def load_movability(output_dir: str) -> Optional[dict]:
        """Load per-class movability scores; returns None if not present."""
        path = os.path.join(output_dir, "vlm_movability.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def classify_thing_stuff(self, classes: List[str], batch_size: int = 80) -> dict:
        """
        For each discovered class, label it as "thing" or "stuff" via the VLM:
            thing = countable, bounded object  (person, car, mug, giraffe, elephant)
            stuff = amorphous region with no instance boundary
                    (sky, vegetation, grass, ground, water, road, wall, floor, ceiling)

        Used downstream as a hard veto: a track whose label is "stuff" never
        receives the dynamic movability prior. This kills the bush-as-giraffe
        false positive at the source — vegetation can never be a giraffe.

        One batched VLM call per run, cached on disk as vlm_thing_stuff.json.
        """
        import re
        if not classes:
            return {}
        self._ensure_loaded()
        result: dict = {}

        for start in range(0, len(classes), batch_size):
            chunk = classes[start:start + batch_size]
            listing = "\n".join(f"- {c}" for c in chunk)
            prompt = (
                "For each object class below, output exactly 'thing' or 'stuff'.\n"
                "  thing = a countable, bounded object you could point at and say 'one of those' "
                "(person, car, chair, mug, monitor, giraffe, elephant, bottle, lamp).\n"
                "  stuff = an amorphous region with no clear instance boundary "
                "(sky, vegetation, grass, ground, water, road, wall, floor, ceiling, "
                "rocks, sand, snow, foliage, dirt, clouds).\n"
                "When in doubt, prefer 'thing' for any class that could be a single object.\n"
                "Reply with exactly one line per class in the form:\n"
                "  class: thing\n"
                "  class: stuff\n"
                "No extra text. Here are the classes:\n"
                f"{listing}"
            )
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text_input = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[text_input], return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                output_ids = self._model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            generated = output_ids[0, inputs.input_ids.shape[1]:]
            response = self._processor.tokenizer.decode(generated, skip_special_tokens=True)

            for line in response.splitlines():
                m = re.match(r"\s*[-*•\d\.\)]*\s*(.+?)\s*[:\-]\s*(thing|stuff)\b", line, re.IGNORECASE)
                if not m:
                    continue
                name = m.group(1).strip().lower().rstrip(".")
                kind = m.group(2).strip().lower()
                if name in chunk:
                    result[name] = kind

        # Default any missing classes to 'thing' (safer — they get classified
        # normally by the rest of the pipeline)
        for c in classes:
            result.setdefault(c, "thing")
        logger.info(f"Classified thing/stuff for {len(classes)} classes "
                    f"({sum(1 for v in result.values() if v == 'stuff')} stuff)")
        return result

    def save_thing_stuff(self, output_dir: str, thing_stuff: dict):
        """Save per-class thing/stuff labels for post-processing reuse."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "vlm_thing_stuff.json")
        with open(path, "w") as f:
            json.dump(thing_stuff, f, indent=2)
        logger.info(f"Saved thing/stuff for {len(thing_stuff)} classes to {path}")

    @staticmethod
    def load_thing_stuff(output_dir: str) -> Optional[dict]:
        """Load per-class thing/stuff labels; returns None if not present."""
        path = os.path.join(output_dir, "vlm_thing_stuff.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def save_classes(self, output_dir: str):
        """
        Save discovered classes to JSON so post-processing can use them.

        Runs a final CLIP-similarity dedup pass before saving so the on-disk
        list is the canonical, non-redundant vocabulary. Called after SLAM
        finishes.
        """
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "vlm_classes.json")
        classes = self.dedupe_classes(self.get_classes())
        with open(path, "w") as f:
            json.dump({"classes": classes, "version": self.get_version()}, f, indent=2)
        logger.info(f"Saved {len(classes)} VLM-discovered classes to {path}")

    @staticmethod
    def load_classes(output_dir: str) -> Optional[List[str]]:
        """
        Load VLM-discovered classes from a previous SLAM run.

        Returns None if no saved classes exist (VLM was disabled).
        """
        path = os.path.join(output_dir, "vlm_classes.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        classes = data["classes"]
        logger.info(f"Loaded {len(classes)} VLM-discovered classes from {path}")
        return classes

    def __del__(self):
        self.stop()
