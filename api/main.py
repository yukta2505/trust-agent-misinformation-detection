"""
TRUST-AGENT FastAPI Application
================================
Endpoints:
  GET  /health      → server status check
  POST /analyse     → main misinformation detection endpoint

Run with:
  uvicorn api.main:app --reload --port 8000

Swagger UI:
  http://localhost:8000/docs
"""

from __future__ import annotations

import os
import tempfile
import logging
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import get_config, get_orchestrator
from api.schemas import AnalyseResponse, HealthResponse
from orchestrator import TrustAgentOrchestrator
from backend.config import Config

LOG = logging.getLogger(__name__)

app = FastAPI(
    title="TRUST-AGENT API",
    description=(
        "Out-of-Context Misinformation Detection. "
        "Upload an image and a text claim to check whether "
        "the image is being used out of context."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health(config: Config = Depends(get_config)):
    return HealthResponse(status="ok", model=config.openai_model)


@app.post("/analyse", response_model=AnalyseResponse, tags=["Detection"])
async def analyse(
    image: UploadFile = File(..., description="Image file to fact-check"),
    claim: str = Form(..., description="The claim being made about this image.",
                      min_length=5, max_length=2000),
    orc: TrustAgentOrchestrator = Depends(get_orchestrator),
):
    suffix = Path(image.filename or "image.jpg").suffix or ".jpg"
    tmp_path: str = ""

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await image.read()
            if not contents:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            tmp.write(contents)
            tmp_path = tmp.name

        LOG.info("Received /analyse | file=%s | claim_len=%d", image.filename, len(claim))
        result = orc.run(image_path=tmp_path, claim=claim)

    except HTTPException:
        raise
    except Exception as exc:
        LOG.exception("Pipeline error in /analyse")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return AnalyseResponse(
        verdict=result.verdict,
        confidence_percent=result.confidence_percent,
        explanation=result.explanation,
        caption=result.caption,
        entity_score=round(result.entity_score, 4),
        temporal_score=round(result.temporal_score, 4),
        credibility_score=round(result.credibility_score, 4),
        final_score=round(result.final_score, 4),
        key_evidence_for_verdict=result.key_evidence_for_verdict,
        flags=result.flags,
        evidence=result.evidence,
        processing_time_sec=round(result.processing_time_sec, 2),
        errors=result.errors,
    )
