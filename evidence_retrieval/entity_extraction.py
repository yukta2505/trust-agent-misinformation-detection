"""Entity extraction module using spaCy NER."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import spacy
from spacy.language import Language

from .config import AppConfig
from .utils import clean_text

LOGGER = logging.getLogger(__name__)
TARGET_LABELS = ("PERSON", "ORG", "GPE", "DATE", "EVENT")


class EntityExtractor:
    """Extract entities from text with a spaCy model."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()
        self.nlp: Optional[Language] = None

    def _load_model(self) -> None:
        if self.nlp is not None:
            return
        try:
            LOGGER.info("Loading spaCy model: %s", self.config.spacy_model_name)
            self.nlp = spacy.load(self.config.spacy_model_name)
        except Exception:
            LOGGER.warning(
                "spaCy model '%s' not found. Falling back to blank English pipeline.",
                self.config.spacy_model_name,
            )
            self.nlp = spacy.blank("en")

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract selected named entities from the provided text."""
        self._load_model()
        assert self.nlp is not None
        clean = clean_text(text)
        doc = self.nlp(clean)
        output: Dict[str, List[str]] = {label: [] for label in TARGET_LABELS}
        for ent in doc.ents:
            if ent.label_ not in output:
                continue
            value = ent.text.strip()
            if value and value not in output[ent.label_]:
                output[ent.label_].append(value)
        return output
