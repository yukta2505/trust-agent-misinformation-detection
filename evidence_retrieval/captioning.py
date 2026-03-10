"""Image captioning module based on BLIP."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from .config import AppConfig

LOGGER = logging.getLogger(__name__)


class ImageCaptioner:
    """Generate captions for uploaded images using BLIP."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()
        self.processor: Optional[BlipProcessor] = None
        self.model: Optional[BlipForConditionalGeneration] = None

    def load_model(self) -> None:
        """Load BLIP processor and model into memory."""
        if self.processor is not None and self.model is not None:
            return
        LOGGER.info("Loading BLIP model: %s", self.config.blip_model_name)
        self.processor = BlipProcessor.from_pretrained(self.config.blip_model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.config.blip_model_name)

    def generate_caption(self, image_path: str) -> str:
        """Generate a natural-language caption for an input image."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        self.load_model()
        assert self.processor is not None
        assert self.model is not None

        try:
            with Image.open(path).convert("RGB") as image:
                inputs = self.processor(images=image, return_tensors="pt")
                output_ids = self.model.generate(**inputs, max_new_tokens=30)
            caption = self.processor.decode(output_ids[0], skip_special_tokens=True).strip()
            LOGGER.info("Generated caption for %s", image_path)
            return caption
        except Exception as exc:
            LOGGER.exception("Failed to generate caption for %s", image_path)
            raise RuntimeError(f"Caption generation failed: {exc}") from exc
