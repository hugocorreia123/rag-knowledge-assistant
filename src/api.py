"""FastAPI REST endpoint for the RAG Knowledge Assistant.

Exposes the same LangGraph workflow as the Streamlit UI, but as a JSON HTTP API
so any other application (web, mobile, another service) can consume it.

Run with:
    uvicorn src.api:app --reload

Interactive docs at:
    http://localhost:8000/docs

This file stays small on purpose — all RAG logic lives in src/graph.py. The
API is a thin transport layer that handles HTTP, validates input/output with
Pydantic, and returns clean JSON.
"""


from __future__ import annotations

import logging
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.graph import ask

from fastapi.responses import RedirectResponse

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schemas (Pydantic) — typed request/response contracts
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    """Body of a POST /ask request."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural-language question about the EU AI Act.",
        examples=["What penalties apply for non-compliance with the EU AI Act?"],
    )


class Citation(BaseModel):
    """One source citation referenced by the answer."""

    source: str = Field(..., description="Source document filename.")
    page: int = Field(..., description="Page number in the source.")
    similarity: float = Field(..., description="Retrieval similarity (0-1).")
    snippet: str = Field(..., description="Short preview of the chunk.")


class AskResponse(BaseModel):
    """Body of a POST /ask response."""

    question: str
    answer: str
    citations: List[Citation]
    was_answered: bool = Field(
        ...,
        description="True if the system produced a grounded answer; "
                    "False if it refused due to insufficient context.",
    )


class HealthResponse(BaseModel):
    """Body of a GET /health response."""

    status: str = "ok"
    service: str = "rag-knowledge-assistant"
    version: str = "0.1.0"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RAG Knowledge Assistant — EU AI Act",
    description=(
        "REST API for a Retrieval-Augmented Generation assistant over the "
        "**EU AI Act** (Regulation (EU) 2024/1689).\n\n"
        "Powered by a multi-step **LangGraph** workflow "
        "(rewrite → retrieve → grade → answer) with a swappable LLM backend."
    ),
    version="0.1.0",
    contact={
        "name": "Hugo Correia",
        "url": "https://github.com/hugocorreia123/rag-knowledge-assistant",
    },
    license_info={"name": "MIT"},
)

# CORS — allow any origin for the public demo. Tighten for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root_redirect() -> RedirectResponse:
    """Redirect the root URL to the interactive API docs."""
    return RedirectResponse(url="/docs")

@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness probe — used by deploy environments and the docs page."""
    return HealthResponse()


@app.post("/ask", response_model=AskResponse, tags=["rag"])
def ask_endpoint(request: AskRequest) -> AskResponse:
    """Run a question through the full RAG workflow and return a grounded answer.

    The workflow performs:

    1. **Rewrite** — optionally reformulate the question for better retrieval.
    2. **Retrieve** — semantic search over the EU AI Act vector store.
    3. **Grade** — judge whether the retrieved context can answer the question.
    4. **Answer** — generate a grounded response with citations, or refuse cleanly.
    """
    try:
        result = ask(request.question)
    except Exception as exc:  # noqa: BLE001
        log.exception("Workflow failed for question=%r", request.question)
        raise HTTPException(status_code=500, detail=f"Workflow error: {exc}") from exc

    answer = result.get("answer", "")
    citation_tags = result.get("citations", [])
    chunks = result.get("chunks", []) or []

    # Build rich citations only when the answer was grounded.
    citations: List[Citation] = []
    if citation_tags:
        for c in chunks:
            if c.cite() in citation_tags:
                citations.append(Citation(
                    source=c.source,
                    page=c.page,
                    similarity=round(c.similarity, 3),
                    snippet=c.text.strip()[:280],
                ))

    return AskResponse(
        question=request.question,
        answer=answer,
        citations=citations,
        was_answered=bool(citation_tags),
    )
