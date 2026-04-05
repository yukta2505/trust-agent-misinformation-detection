"""
Image captioning — GPT-4o Vision (primary) with LLaVA fallback.

LLaVA is significantly better than BLIP:
- Better scene understanding
- Handles context (events, protests, environments)
- Produces richer multi-sentence captions
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Optional

from .config import Config

LOG = logging.getLogger(__name__)

GPT4V_PROMPT = """Analyse this image carefully for fact-checking.

Describe:
- What is happening
- Location clues (flags, landmarks, language, architecture)
- People (count, clothing, actions)
- ALL visible text (signs, banners, logos)
- Any symbols, organizations, or political indicators

Be precise and factual. No assumptions.
Write 4-6 detailed sentences.
"""


class ImageCaptioner:
    """
    Generate detailed image captions using:
    - GPT-4o Vision (if API key available)
    - LLaVA (fallback - free & powerful)
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self._llava_model = None
        self._llava_processor = None

    # ── GPT-4o Vision (PRIMARY) ─────────────────────────────────────────────

    def _caption_gpt4v(self, image_path: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.config.openai_api_key)

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        ext = Path(image_path).suffix.lower()
        media_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        media_type = media_type_map.get(ext, "image/jpeg")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": GPT4V_PROMPT},
                    ],
                }
            ],
        )

        caption = response.choices[0].message.content.strip()
        LOG.info("GPT-4o Vision caption generated (%d chars)", len(caption))
        return caption

    # ── LLaVA (FALLBACK) ────────────────────────────────────────────────────

    def _load_llava(self) -> None:
        if self._llava_model is not None:
            return

        from transformers import LlavaForConditionalGeneration, AutoProcessor
        import torch

        model_id = "llava-hf/llava-1.5-7b-hf"

        LOG.info("Loading LLaVA model: %s", model_id)

        self._llava_processor = AutoProcessor.from_pretrained(model_id)

        self._llava_model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    def _caption_llava(self, image_path: str) -> str:
        from PIL import Image
        import torch

        self._load_llava()

        image = Image.open(image_path).convert("RGB")

        prompt = f"<image>\n{GPT4V_PROMPT}"

        inputs = self._llava_processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self._llava_model.device)

        output = self._llava_model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False
        )

        caption = self._llava_processor.decode(
            output[0],
            skip_special_tokens=True
        )

        LOG.info("LLaVA caption generated (%d chars)", len(caption))
        return caption.strip()

    # ── PUBLIC INTERFACE ────────────────────────────────────────────────────

    def caption(self, image_path: str) -> str:
        """
        Generate caption:
        - Try GPT-4o Vision
        - Fallback to LLaVA if unavailable or fails
        """
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Try GPT-4o Vision
        if self.config.openai_api_key:
            try:
                return self._caption_gpt4v(image_path)
            except Exception as exc:
                if "quota" in str(exc).lower():
                    LOG.warning("OpenAI quota exceeded → switching to LLaVA")
                else:
                    LOG.warning("GPT-4o Vision failed: %s", exc)

        # Fallback
        LOG.info("Using LLaVA for captioning")
        return self._caption_llava(image_path)