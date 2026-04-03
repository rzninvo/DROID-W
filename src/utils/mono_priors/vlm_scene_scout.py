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

# VLM prompt for object discovery
DISCOVERY_PROMPT = (
    "List every distinct object you can see in this image. "
    "Include large objects (furniture, vehicles, walls) AND small objects "
    "(door handles, mugs, switches, books, bottles, pens, cables). "
    "Include structural elements (door, window, shelf, radiator, vent). "
    "Include things on surfaces (monitor, keyboard, plant, photo frame). "
    "Use common, short nouns. Be extremely thorough — list even partially visible objects. "
    "Return ONLY a comma-separated list, nothing else."
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
    Parse VLM comma-separated response into clean class names.

    Handles common VLM quirks: numbered lists, bullet points, extra whitespace,
    quotes, periods, and mixed formatting. Strips color/size adjectives so
    "blue tshirt" becomes "tshirt" (which then gets filtered by IGNORE_CLASSES).

    Args:
        response: Raw text from VLM.

    Returns:
        List of cleaned, lowercase class names.
    """
    import re

    # Handle numbered lists: "1. person, 2. car" or "1) person"
    response = re.sub(r'\d+[\.\)]\s*', '', response)
    # Remove bullet points
    response = re.sub(r'[-*•]\s*', '', response)
    # Remove quotes
    response = response.replace('"', '').replace("'", "")

    # Split by comma or newline
    parts = re.split(r'[,\n]', response)

    classes = []
    for part in parts:
        name = part.strip().lower().rstrip('.')
        # Skip empty or overly long entries (VLM hallucination)
        if not name or len(name) >= 40 or len(name) <= 1:
            continue

        # Strip attribute words (e.g., "blue tshirt" → "tshirt")
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
        self.enabled = vlm_cfg.get("activate", False)
        self.max_classes = vlm_cfg.get("max_classes", 200)

        # Thread-safe class storage — seeded with DEFAULT_CLASSES so common
        # objects (person, car, chair, etc.) are always detected even if
        # the VLM misses them. VLM adds scene-specific classes on top.
        from src.utils.mono_priors.seg_model import DEFAULT_CLASSES
        self._lock = threading.Lock()
        self._classes: Set[str] = {c.lower() for c in DEFAULT_CLASSES}
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
                self._classes_changed = True
                self._class_list_version += 1

        if new_classes:
            logger.info(f"VLM discovered {len(new_classes)} new classes: {new_classes}")

        return new_classes

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

    def submit_keyframe(self, image_np: np.ndarray, keyframe_idx: int):
        """
        Submit a keyframe for async VLM processing.

        Only processes every Nth keyframe (based on query_interval).
        First keyframe (idx 0) is always processed.
        Non-blocking — drops frame if queue is full.
        """
        if not self.enabled:
            return

        if keyframe_idx > 0 and keyframe_idx % self.query_interval != 0:
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

    def save_classes(self, output_dir: str):
        """
        Save discovered classes to JSON so post-processing can use them.

        Called after SLAM finishes. The detection pipeline loads these
        instead of using DEFAULT_CLASSES.
        """
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "vlm_classes.json")
        classes = self.get_classes()
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
