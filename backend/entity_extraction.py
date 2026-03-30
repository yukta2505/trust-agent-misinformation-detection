"""Named-entity extraction using spaCy."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import spacy
from spacy.language import Language

from .config import Config
from .utils import clean_text

LOG = logging.getLogger(__name__)

# Entity labels we care about
LABELS = ("PERSON", "ORG", "GPE", "DATE", "EVENT", "LOC", "NORP")


class EntityExtractor:
    """Extract named entities from text using a spaCy model."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self._nlp: Optional[Language] = None

    def _load(self) -> None:
        if self._nlp is not None:
            return
        try:
            self._nlp = spacy.load(self.config.spacy_model)
        except OSError:
            LOG.warning("spaCy model '%s' not found — using blank English.", self.config.spacy_model)
            self._nlp = spacy.blank("en")

    def extract(self, text: str) -> Dict[str, List[str]]:
        """Return a dict mapping entity labels to unique entity strings."""
        self._load()
        doc = self._nlp(clean_text(text))
        result: Dict[str, List[str]] = {label: [] for label in LABELS}
        for ent in doc.ents:
            if ent.label_ in result:
                val = ent.text.strip()
                if val and val not in result[ent.label_]:
                    result[ent.label_].append(val)
        return result

    def flat_set(self, entities: Dict[str, List[str]]) -> set[str]:
        """Flatten entity dict to a lowercase set for overlap checks."""
        out: set[str] = set()
        for vals in entities.values():
            for v in vals:
                if v.strip():
                    out.add(v.strip().lower())
        return out
