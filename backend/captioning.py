"""Image captioning with BLIP."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from .config import Config

LOG = logging.getLogger(__name__)


class ImageCaptioner:
    """Generate natural-language captions for images using BLIP."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self._processor: Optional[BlipProcessor] = None
        self._model: Optional[BlipForConditionalGeneration] = None

    def _load(self) -> None:
        if self._processor is not None:
            return
        LOG.info("Loading BLIP model: %s", self.config.blip_model)
        self._processor = BlipProcessor.from_pretrained(self.config.blip_model)
        self._model = BlipForConditionalGeneration.from_pretrained(self.config.blip_model)

    def caption(self, image_path: str) -> str:
        """Return a caption string for the given image path."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        self._load()
        with Image.open(path).convert("RGB") as img:
            inputs = self._processor(images=img, return_tensors="pt")
            ids = self._model.generate(
    **inputs,
    max_new_tokens=80,
    num_beams=5,
    temperature=0.7,
    repetition_penalty=1.2,
    length_penalty=1.0,
    early_stopping=True
)
        return self._processor.decode(ids[0], skip_special_tokens=True).strip()
