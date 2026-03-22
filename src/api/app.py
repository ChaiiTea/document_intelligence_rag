"""
FastAPI application — exposes the pipeline as a REST API.

Endpoints:
  POST /extract   — upload a PDF, get back structured extraction JSON
  POST /search    — semantic search across indexed documents
  GET  /health    — liveness check
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger

from src.pipeline import DocumentIntelligencePipeline


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Document Intelligence API",
    description="Structured extraction from PDFs using LayoutLMv3 + semantic search via FAISS.",
    version="0.1.0",
)

# Initialise pipeline once at startup (lazy model loading inside)
_pipeline: DocumentIntelligencePipeline | None = None


def get_pipeline() -> DocumentIntelligencePipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("[API] Initialising pipeline...")
        _pipeline = DocumentIntelligencePipeline.from_config("configs/config.yaml")
    return _pipeline


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchHit(BaseModel):
    text: str
    pdf_name: str
    page: int
    label: str | None
    score: float


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract", response_class=JSONResponse)
async def extract(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Upload a PDF and receive structured extracted fields.

    Returns JSON with labelled fields and any flagged (low-confidence) spans.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    contents = await file.read()

    # Write to a temp file (pdf_loader needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        pipeline = get_pipeline()
        doc = pipeline.run(str(tmp_path), log_to_mlflow=True)
        # Override pdf_name with the original upload filename
        doc.pdf_name = file.filename
        return doc.to_dict()
    except Exception as e:
        logger.exception(f"[/extract] Error processing {file.filename}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """
    Semantic search over all previously indexed documents.
    """
    pipeline = get_pipeline()
    results = pipeline.search(request.query, top_k=request.top_k)

    hits = [
        SearchHit(
            text=r.chunk.text,
            pdf_name=r.chunk.pdf_name,
            page=r.chunk.page,
            label=r.chunk.label,
            score=round(r.score, 4),
        )
        for r in results
    ]
    return SearchResponse(query=request.query, hits=hits)
