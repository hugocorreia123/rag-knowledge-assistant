"""Semantic retriever over the ingested EU AI Act vector store.

Wraps the Chroma collection in a clean interface so the LangGraph workflow
never deals with vector-store details directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from src.config import CHROMA_COLLECTION, EMBEDDING_MODEL_NAME, TOP_K
from src.ingestion import get_chroma_client

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RetrievedChunk:
    """A single chunk returned by the retriever."""

    text: str
    source: str
    page: int
    similarity: float

    def cite(self) -> str:
        """Short citation tag used inside generated answers."""
        return f"[{self.source} p.{self.page}]"


# ---------------------------------------------------------------------------
# Lazy singletons (model + collection)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_embedder() -> SentenceTransformer:
    log.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def _get_collection():
    client = get_chroma_client()
    return client.get_collection(CHROMA_COLLECTION)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def retrieve(query: str, top_k: int = TOP_K) -> List[RetrievedChunk]:
    """Return the top-k most semantically similar chunks for a query."""
    embedder = _get_embedder()
    collection = _get_collection()

    embedding = embedder.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    chunks = [
        RetrievedChunk(
            text=doc,
            source=meta["source"],
            page=int(meta["page"]),
            similarity=float(1 - dist),  # cosine distance → similarity
        )
        for doc, meta, dist in zip(docs, metas, dists)
    ]

    log.info(
        "Retrieved %d chunks (top similarity=%.3f)",
        len(chunks),
        chunks[0].similarity if chunks else 0.0,
    )
    return chunks


def format_context(chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks into a single context block for the LLM.

    Each chunk is labelled with a citation tag the LLM can copy into its answer.
    """
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(
            f"--- Chunk {i} {c.cite()} ---\n{c.text.strip()}"
        )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")

    question = "What penalties apply for non-compliance with the EU AI Act?"
    chunks = retrieve(question)

    print(f"\nQ: {question}")
    print("-" * 80)
    for i, c in enumerate(chunks, start=1):
        print(f"[{i}] {c.cite()}  similarity={c.similarity:.3f}")
        print(f"    {c.text[:200]}...\n")