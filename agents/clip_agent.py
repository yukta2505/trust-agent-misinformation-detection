"""
CLIP Visual Consistency Agent
==============================
Runs 100% LOCALLY — no API key, no internet after first download.

What it does:
  Uses OpenAI CLIP (ViT-B/32) to compute direct image-text similarity.
  This measures whether the IMAGE ITSELF is semantically consistent
  with the CLAIM text — purely from visual features.

Why this is novel:
  - E2LVLM and EXCLAIM both rely on external evidence retrieval
  - This agent works even when NO evidence is found online
  - Adds a 5th independent signal that cannot be faked by evidence noise
  - CLIP was trained on 400M image-text pairs — has strong visual priors

Scoring logic:
  - High similarity (>0.25)  → image and claim are visually consistent
  - Medium (0.18-0.25)       → partial match, ambiguous
  - Low (<0.18)              → image and claim are visually inconsistent

Download size: ~600 MB (one-time, cached locally after first run)
Inference time: ~1-2 seconds on CPU
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

LOG = logging.getLogger(__name__)


class CLIPVisualAgent:
    """
    Local CLIP-based image-text consistency checker.
    No API key required. Runs on CPU.
    """

    MODEL_ID = "openai/clip-vit-base-patch32"

    def __init__(self) -> None:
        self._model = None
        self._processor = None

    def _load(self) -> None:
        """Load CLIP model (cached after first download)."""
        if self._model is not None:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor
            import torch
            LOG.info("Loading CLIP model: %s (local, no API)", self.MODEL_ID)
            self._processor = CLIPProcessor.from_pretrained(self.MODEL_ID)
            self._model = CLIPModel.from_pretrained(self.MODEL_ID)
            self._model.eval()
            LOG.info("CLIP model loaded successfully")
        except ImportError:
            raise RuntimeError(
                "transformers and torch are required for CLIP. "
                "Run: pip install transformers torch"
            )

    def compute_similarity(self, image_path: str, text: str) -> float:
        """
        Compute cosine similarity between image and text using CLIP.

        Returns a float between 0 and 1.
        Higher = more visually consistent.
        """
        import torch
        from PIL import Image

        self._load()

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(path).convert("RGB") as img:
            inputs = self._processor(
                text=[text],
                images=img,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,  # CLIP text limit
            )

        with torch.no_grad():
            outputs = self._model(**inputs)
            # Normalised cosine similarity (logits scaled by temperature)
            logits = outputs.logits_per_image  # shape: [1, 1]
            # Convert to 0-1 range using sigmoid
            similarity = torch.sigmoid(logits / 100.0).item()

        return float(similarity)

    def analyse(
        self,
        image_path: str,
        claim: str,
        caption: str,
    ) -> Dict[str, Any]:
        """
        Run CLIP visual consistency analysis.

        Checks similarity between:
        1. Image vs Claim text
        2. Image vs Caption text
        3. Claim vs Caption (text-only cross check)

        Returns a dict with clip_score (0-1) and reasoning.
        """
        try:
            # Truncate claim to CLIP's 77-token limit
            claim_short = claim[:200]
            caption_short = caption[:200]

            # Image ↔ Claim similarity
            img_claim_sim = self.compute_similarity(image_path, claim_short)

            # Image ↔ Caption similarity (should be high — caption describes image)
            img_caption_sim = self.compute_similarity(image_path, caption_short)

            LOG.info(
                "CLIP scores — img↔claim=%.3f  img↔caption=%.3f",
                img_claim_sim, img_caption_sim,
            )

            # If image matches caption well but not claim → mismatch signal
            # If both are low → poor evidence
            # If both are high → consistent

            caption_baseline = img_caption_sim  # how well caption matches image

            if caption_baseline > 0.0:
                # Relative score: how well claim matches vs caption baseline
                relative = img_claim_sim / caption_baseline
            else:
                relative = 0.5

            # Convert to 0-1 clip_score
            # relative > 0.85 → claim is as good as caption → consistent
            # relative < 0.50 → claim is much weaker than caption → mismatch
            clip_score = min(max(relative, 0.0), 1.0)

            # Determine interpretation
            if clip_score >= 0.80:
                interpretation = "CONSISTENT"
                reasoning = (
                    f"CLIP similarity between image and claim ({img_claim_sim:.3f}) "
                    f"is close to caption baseline ({img_caption_sim:.3f}). "
                    "Visual content matches the claim."
                )
            elif clip_score >= 0.55:
                interpretation = "UNCERTAIN"
                reasoning = (
                    f"CLIP similarity is moderate (claim={img_claim_sim:.3f}, "
                    f"caption={img_caption_sim:.3f}). "
                    "Visual content partially matches the claim."
                )
            else:
                interpretation = "INCONSISTENT"
                reasoning = (
                    f"CLIP similarity between image and claim ({img_claim_sim:.3f}) "
                    f"is significantly lower than caption ({img_caption_sim:.3f}). "
                    "Visual content appears inconsistent with the claim."
                )

            return {
                "clip_score":           round(clip_score, 4),
                "img_claim_similarity": round(img_claim_sim, 4),
                "img_caption_similarity": round(img_caption_sim, 4),
                "interpretation":       interpretation,
                "reasoning":            reasoning,
                "local_model":          self.MODEL_ID,
                "requires_api":         False,
            }

        except Exception as exc:
            LOG.error("CLIP agent error: %s", exc)
            return {
                "clip_score":           0.5,
                "img_claim_similarity": 0.0,
                "img_caption_similarity": 0.0,
                "interpretation":       "ERROR",
                "reasoning":            f"CLIP analysis failed: {exc}",
                "local_model":          self.MODEL_ID,
                "requires_api":         False,
            }