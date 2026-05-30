"""Ingestion pipeline: PDF → chunks → embeddings → Chroma vector store.

Run as a script:
    python -m src.ingestion

This reads every PDF in data/raw/, splits the text into overlapping chunks,
embeds them with a local sentence-transformer model, and writes them to a
persisted Chroma collection on disk.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from src.config import (
    CHROMA_COLLECTION,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    PATHS,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingestion")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Chunk:
    """A single text chunk with its source metadata."""

    text: str
    source: str       # filename it came from
    page: int         # 1-indexed page number
    chunk_index: int  # 0-indexed position within the document


# ---------------------------------------------------------------------------
# Step 1 — PDF parsing
# ---------------------------------------------------------------------------
def extract_pages(pdf_path: Path) -> List[str]:
    """Extract text from each page of a PDF.

    Returns a list where index i is the text of page (i + 1).
    Empty pages are returned as empty strings (kept for page numbering).
    """
    log.info("Reading PDF: %s", pdf_path.name)
    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(text)
        if i % 25 == 0:
            log.info("  ... extracted %d pages", i)
    log.info("Done: %d pages, %d total chars",
             len(pages), sum(len(p) for p in pages))
    return pages


# ---------------------------------------------------------------------------
# Step 2 — Chunking
# ---------------------------------------------------------------------------
def chunk_document(pages: List[str], source: str) -> List[Chunk]:
    """Split a document's pages into overlapping chunks.

    Uses LangChain's RecursiveCharacterTextSplitter, which tries to split on
    paragraph/sentence boundaries before falling back to mid-sentence splits.
    Each chunk carries its source filename and originating page number.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Chunk] = []
    running_index = 0

    for page_number, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue
        for piece in splitter.split_text(page_text):
            chunks.append(
                Chunk(
                    text=piece.strip(),
                    source=source,
                    page=page_number,
                    chunk_index=running_index,
                )
            )
            running_index += 1

    log.info("Chunked into %d chunks (size=%d, overlap=%d)",
             len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)
    return chunks


# ---------------------------------------------------------------------------
# Step 3 — Embedding
# ---------------------------------------------------------------------------
def embed_chunks(
    chunks: List[Chunk],
    model: SentenceTransformer,
    batch_size: int = 32,
) -> List[List[float]]:
    """Compute vector embeddings for every chunk's text."""
    log.info("Embedding %d chunks with %s ...", len(chunks), EMBEDDING_MODEL_NAME)
    t0 = time.perf_counter()
    embeddings = model.encode(
        [c.text for c in chunks],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=False,
    )
    elapsed = time.perf_counter() - t0
    log.info("Embedded %d chunks in %.1fs (%.1f chunks/s)",
             len(chunks), elapsed, len(chunks) / elapsed)
    return [list(map(float, vec)) for vec in embeddings]


# ---------------------------------------------------------------------------
# Step 4 — Chroma storage
# ---------------------------------------------------------------------------
def get_chroma_client() -> chromadb.PersistentClient:
    """Return a Chroma client persisting to data/chroma_db/."""
    return chromadb.PersistentClient(
        path=str(PATHS.chroma_db),
        settings=Settings(anonymized_telemetry=False),
    )


def store_in_chroma(
    chunks: List[Chunk],
    embeddings: List[List[float]],
    collection_name: str = CHROMA_COLLECTION,
) -> None:
    """Upsert chunks and their embeddings into a Chroma collection.

    The collection is recreated each run so re-ingestion produces a clean
    state (no duplicate IDs, no stale chunks).
    """
    client = get_chroma_client()

    # Recreate the collection cleanly.
    try:
        client.delete_collection(collection_name)
        log.info("Deleted existing collection: %s", collection_name)
    except (ValueError, Exception):
        # ValueError on older chromadb versions when collection doesn't exist
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"{c.source}:{c.chunk_index}" for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [{"source": c.source, "page": c.page} for c in chunks]

    # Chroma performs best with batched adds.
    batch = 256
    for i in range(0, len(chunks), batch):
        collection.add(
            ids=ids[i : i + batch],
            embeddings=embeddings[i : i + batch],
            documents=documents[i : i + batch],
            metadatas=metadatas[i : i + batch],
        )

    log.info("Stored %d chunks in Chroma collection '%s'",
             len(chunks), collection_name)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def find_pdfs(directory: Path) -> List[Path]:
    """Return all .pdf files in `directory`, sorted by name."""
    return sorted(directory.glob("*.pdf"))


def run_ingestion() -> None:
    """End-to-end: parse → chunk → embed → store, for every PDF in data/raw/."""
    PATHS.ensure()
    pdfs = find_pdfs(PATHS.data_raw)
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {PATHS.data_raw}. "
            "Run `bash scripts/download_data.sh` first."
        )

    log.info("Found %d PDF(s) to ingest: %s",
             len(pdfs), [p.name for p in pdfs])

    # Load embedding model once and reuse.
    log.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    all_chunks: List[Chunk] = []
    for pdf in pdfs:
        pages = extract_pages(pdf)
        all_chunks.extend(chunk_document(pages, source=pdf.name))

    embeddings = embed_chunks(all_chunks, model)
    store_in_chroma(all_chunks, embeddings)

    log.info("✓ Ingestion complete: %d chunks across %d document(s).",
             len(all_chunks), len(pdfs))


if __name__ == "__main__":
    run_ingestion()