"""Configuration utilities for the evidence retrieval pipeline."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(env_path)
from dataclasses import dataclass


@dataclass
class AppConfig:
    """Runtime configuration for retrieval modules."""

    blip_model_name: str = os.getenv("BLIP_MODEL_NAME", "Salesforce/blip-image-captioning-base")
    clip_model_name: str = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
    sentence_model_name: str = os.getenv(
        "SENTENCE_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    spacy_model_name: str = os.getenv("SPACY_MODEL_NAME", "en_core_web_sm")
    historical_index_path: str = os.getenv("HISTORICAL_INDEX_PATH", "artifacts/historical.faiss")
    historical_metadata_path: str = os.getenv(
        "HISTORICAL_METADATA_PATH", "artifacts/historical_metadata.json"
    )
    reverse_image_index_path: str = os.getenv(
        "REVERSE_IMAGE_INDEX_PATH", "artifacts/reverse_image.faiss"
    )
    reverse_image_metadata_path: str = os.getenv(
        "REVERSE_IMAGE_METADATA_PATH", "artifacts/reverse_image_metadata.json"
    )
    bing_visual_api_key: str = os.getenv("BING_VISUAL_API_KEY", "")
    bing_visual_endpoint: str = os.getenv(
        "BING_VISUAL_ENDPOINT", "https://api.bing.microsoft.com/v7.0/images/visualsearch"
    )
    google_vision_api_key: str = os.getenv("GOOGLE_VISION_API_KEY", "")
    google_vision_endpoint: str = os.getenv(
        "GOOGLE_VISION_ENDPOINT", "https://vision.googleapis.com/v1/images:annotate"
    )
    serpapi_key: str = os.getenv("SERPAPI_API_KEY", "")
    serpapi_reverse_engine: str = os.getenv("SERPAPI_REVERSE_ENGINE", "google_reverse_image")
    serpapi_reverse_gl: str = os.getenv("SERPAPI_REVERSE_GL", "")
    serpapi_reverse_hl: str = os.getenv("SERPAPI_REVERSE_HL", "")
    newsapi_key: str = os.getenv("NEWSAPI_API_KEY", "")
    request_timeout_sec: int = int(os.getenv("REQUEST_TIMEOUT_SEC", "20"))
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    imgbb_api_key: str = os.getenv("IMGBB_API_KEY", "")
