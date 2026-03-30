"""
Metadata Extractor
==================
Extracts temporal and structural metadata from:
  1. Image EXIF data (capture date, GPS, camera model)
  2. Claim text (dates, years, locations mentioned)

This feeds additional context into the temporal and entity agents
so they have richer signals to reason about.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from PIL.ExifTags import TAGS

from .utils import parse_timestamp

LOG = logging.getLogger(__name__)


class MetadataExtractor:
    """Extract metadata from image EXIF and claim text."""

    # Regex patterns for date/year extraction from text
    _YEAR_RE   = re.compile(r"\b(19[0-9]{2}|20[0-2][0-9])\b")
    _DATE_RE   = re.compile(
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{2}[/-]\d{2})\b"
    )
    _MONTH_RE  = re.compile(
        r"\b(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\b",
        re.IGNORECASE,
    )

    # ── Image EXIF ────────────────────────────────────────────────────────────

    def extract_exif(self, image_path: str) -> Dict[str, Any]:
        """
        Extract EXIF metadata from an image file.

        Returns a dict with keys: capture_date, gps_lat, gps_lon,
        camera_make, camera_model, software, raw (all EXIF tags).
        """
        result: Dict[str, Any] = {
            "capture_date": None,
            "gps_lat": None,
            "gps_lon": None,
            "camera_make": None,
            "camera_model": None,
            "software": None,
            "raw": {},
        }

        path = Path(image_path)
        if not path.exists():
            LOG.warning("Image not found for EXIF extraction: %s", image_path)
            return result

        try:
            with Image.open(path) as img:
                exif_data = img._getexif()   # type: ignore[attr-defined]
                if exif_data is None:
                    LOG.info("No EXIF data in %s", image_path)
                    return result

                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    result["raw"][str(tag)] = str(value)

                    if tag == "DateTimeOriginal":
                        # Convert "2024:03:15 10:22:11" → ISO
                        try:
                            iso = value.replace(":", "-", 2).replace(" ", "T")
                            result["capture_date"] = iso
                        except Exception:
                            pass

                    elif tag == "Make":
                        result["camera_make"] = str(value).strip()

                    elif tag == "Model":
                        result["camera_model"] = str(value).strip()

                    elif tag == "Software":
                        result["software"] = str(value).strip()

                    elif tag == "GPSInfo":
                        result["gps_lat"], result["gps_lon"] = self._parse_gps(value)

        except Exception as exc:
            LOG.warning("EXIF extraction failed for %s: %s", image_path, exc)

        return result

    @staticmethod
    def _parse_gps(gps_info: Any):
        """Parse GPSInfo EXIF tag into (lat, lon) floats."""
        try:
            def to_deg(val):
                d, m, s = val
                return float(d) + float(m) / 60 + float(s) / 3600

            lat = to_deg(gps_info.get(2, (0, 0, 0)))
            if gps_info.get(1) == "S":
                lat = -lat
            lon = to_deg(gps_info.get(4, (0, 0, 0)))
            if gps_info.get(3) == "W":
                lon = -lon
            return round(lat, 6), round(lon, 6)
        except Exception:
            return None, None

    # ── Claim text ────────────────────────────────────────────────────────────

    def extract_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract temporal and structural clues from a text string (claim or caption).

        Returns years, specific dates, month references, and a
        normalised primary_date if one can be determined.
        """
        years:  List[str] = self._YEAR_RE.findall(text)
        dates:  List[str] = self._DATE_RE.findall(text)
        months: List[str] = self._MONTH_RE.findall(text)

        primary_date: Optional[str] = None
        if dates:
            primary_date = parse_timestamp(dates[0]) and dates[0]
        elif years:
            primary_date = years[0]

        return {
            "years_mentioned":  years,
            "dates_mentioned":  dates,
            "months_mentioned": months,
            "primary_date":     primary_date,
        }

    # ── Combined ──────────────────────────────────────────────────────────────

    def extract_all(
        self,
        image_path: str,
        claim: str,
        caption: str = "",
    ) -> Dict[str, Any]:
        """
        Run full metadata extraction on image + claim + caption.

        Returns:
          image_exif   : EXIF fields from the image
          claim_meta   : temporal clues from the claim text
          caption_meta : temporal clues from the generated caption
          summary      : human-readable summary string for agents
        """
        image_exif   = self.extract_exif(image_path)
        claim_meta   = self.extract_from_text(claim)
        caption_meta = self.extract_from_text(caption)

        # Build a short summary string that the agents can use directly
        parts: List[str] = []
        if image_exif.get("capture_date"):
            parts.append(f"Image capture date (EXIF): {image_exif['capture_date']}")
        if image_exif.get("camera_make"):
            parts.append(f"Camera: {image_exif['camera_make']} {image_exif.get('camera_model', '')}")
        if claim_meta.get("years_mentioned"):
            parts.append(f"Years in claim: {', '.join(claim_meta['years_mentioned'])}")
        if claim_meta.get("dates_mentioned"):
            parts.append(f"Dates in claim: {', '.join(claim_meta['dates_mentioned'])}")
        if caption_meta.get("years_mentioned"):
            parts.append(f"Years in caption: {', '.join(caption_meta['years_mentioned'])}")

        return {
            "image_exif":   image_exif,
            "claim_meta":   claim_meta,
            "caption_meta": caption_meta,
            "summary":      " | ".join(parts) if parts else "No metadata extracted.",
        }
