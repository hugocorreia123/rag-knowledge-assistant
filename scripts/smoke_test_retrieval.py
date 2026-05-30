"""Smoke test: prove the ingested vector store can retrieve relevant chunks.

Run with:
    python scripts/smoke_test_retrieval.py
    python scripts/smoke_test_retrieval.py "your custom question here"

This is *not* RAG yet — there's no LLM in the loop. It only tests that
semantic search over the embedded chunks returns relevant passages, which
is the prerequisite for everything in Phase 2+.
"""

from __future__ import annotations

import logging
import sys
import textwrap

from sentence_transformers import SentenceTransformer

from src.config import CHROMA_COLLECTION, EMBEDDING_MODEL_NAME, TOP_K
from src.ingestion import get_chroma_client

# Quiet the noisy chromadb telemetry warnings
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.ERROR)

DEFAULT_QUESTIONS = [
    "What are the obligations for providers of high-risk AI systems?",
    "Which AI practices are prohibited under the EU AI Act?",
    "What is the definition of a general-purpose AI model?",
    "What penalties apply for non-compliance?",
]


def run_query(question: str, top_k: int = TOP_K) -> None:
    """Embed a question and print the top-k most similar chunks."""
    client = get_chroma_client()
    collection = client.get_collection(CHROMA_COLLECTION)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    embedding = model.encode([question])[0].tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    print()
    print("=" * 80)
    print(f"Q: {question}")
    print("=" * 80)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        # Lower distance = more similar (cosine distance, so 0 = identical)
        similarity = 1 - dist
        snippet = textwrap.shorten(doc.replace("\n", " "), width=320, placeholder=" …")
        print(
            f"\n[{rank}] page {meta['page']:>3}  "
            f"similarity={similarity:.3f}  source={meta['source']}"
        )
        print(f"     {snippet}")


def main() -> None:
    # Accept a custom question from the command line, otherwise run defaults.
    if len(sys.argv) > 1:
        questions = [" ".join(sys.argv[1:])]
    else:
        questions = DEFAULT_QUESTIONS

    for q in questions:
        run_query(q)


if __name__ == "__main__":
    main()