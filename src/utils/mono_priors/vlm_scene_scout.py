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


# Fallback classes if VLM fails or is disabled — minimal seed list
SEED_CLASSES = ["person", "car", "chair", "table", "door"]

# Classes to filter out — backgrounds, clothing, body parts, colors, materials
IGNORE_CLASSES = {
    # Background/surfaces
    "floor", "ceiling", "wall", "ground", "sky", "background",
    "shadow", "light", "air", "space", "none", "nothing", "road",
    "sidewalk", "pavement", "grass", "dirt", "concrete", "asphalt",
    # Clothing (competes with "person" in YOLO-World)
    "shirt", "tshirt", "t-shirt", "jacket", "coat", "pants", "jeans",
    "shorts", "dress", "skirt", "hat", "cap", "helmet", "shoe", "shoes",
    "boot", "boots", "sneakers", "hoodie", "sweater", "vest", "scarf",
    "glove", "gloves", "sock", "socks", "mask", "glasses", "sunglasses",
    # Body parts
    "hand", "hands", "arm", "arms", "leg", "legs", "head", "face",
    "foot", "feet", "hair", "finger", "fingers",
    # Gendered/age variants — normalize to "person"
    "man", "woman", "boy", "girl", "child", "kid", "baby",
    "lady", "gentleman", "male", "female", "pedestrian",
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
    "Analyze this image in two steps and reply ONLY with a JSON object.\n"
    "Step 1 — In one short phrase, identify the scene type "
    '(e.g. "office", "kitchen", "outdoor street", "lecture hall").\n'
    "Step 2 — List every distinct object you can see, using SPECIFIC compound "
    "nouns when needed:\n"
    '  - "computer monitor" not "monitor"\n'
    '  - "office chair" not "chair"\n'
    '  - "coffee mug" not "cup"\n'
    '  - "desk lamp" not "lamp"\n'
    "  - structural elements (door, window, radiator, vent, shelf)\n"
    "  - small items (cable, photo frame, door handle, light switch, keyboard)\n"
    "Be thorough — include partially visible items. Use lowercase nouns only "
    "(no colors, sizes, or materials).\n"
    'Reply EXACTLY in this JSON form, no extra text:\n'
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

    classes = []
    for part in raw_objects:
        name = str(part).strip().lower().rstrip('.')
        # Skip empty or overly long entries (VLM hallucination)
        if not name or len(name) >= 40 or len(name) <= 1:
            continue

        # Don't accept the scene type as an object
        if scene_name and name == scene_name:
            continue

        # Strip attribute words (e.g., "blue tshirt" → "tshirt"). Compound
        # nouns like "office chair" are preserved because none of their words
        # are attribute words.
        words = name.split()
        words = [w for w in words if w not in ATTRIBUTE_WORDS]
        name = " ".join(words).strip()

        if not name:
            continue

        # Skip if the cleaned name is in IGNORE_CLASSES
        if name in IGNORE_CLASSES:
            continue

        # Skip if any word in the name is clothing/body part
        if any(w in IGNORE_CLASSES for w in name.split()):
            continue

        classes.append(name)

    return classes


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
        # Default 0.85: ConceptGraphs uses 0.8, HOV-SG uses 0.85–0.9.
        # 0.85 catches "tv"≈"monitor" / "couch"≈"sofa" without over-collapsing.
        self.dedupe_clip_threshold = vlm_cfg.get("dedupe_clip_threshold", 0.85)
        self.dedupe_clip_model = vlm_cfg.get("dedupe_clip_model", "ViT-B-32")
        self._clip_dedup = None  # lazy-loaded (model, tokenizer) tuple

        # Adaptive-gating state — count queries since last vocabulary growth.
        self._queries_since_growth = 0

        # Thread-safe class storage. Seed only if explicitly requested or if
        # VLM discovery is disabled (otherwise we'd fight the VLM's choices).
        self._lock = threading.Lock()
        if self.use_default_seed or not self.enabled:
            from src.utils.mono_priors.seg_model import DEFAULT_CLASSES
            self._classes: Set[str] = {c.lower() for c in DEFAULT_CLASSES}
        else:
            self._classes: Set[str] = set()
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
                # CLIP-merge any near-duplicates introduced by this batch
                # (e.g. discovered "monitor" while seed already had "computer
                # monitor"). Keeps the canonical name per cluster.
                deduped = set(self.dedupe_classes(sorted(self._classes)))
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

        self._ensure_clip_dedup_loaded()
        model, tokenizer = self._clip_dedup
        if model is None:
            return list(classes)

        with torch.no_grad():
            tokens = tokenizer(classes).to(self.device)
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
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
                "0.0 = permanently fixed (wall, floor, ceiling).\n"
                "0.5 = can be moved by a person (chair, cup, laptop).\n"
                "1.0 = moves on its own (person, car, dog, bicycle).\n"
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
