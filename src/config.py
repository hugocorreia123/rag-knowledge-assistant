"""Centralized configuration for the RAG Knowledge Assistant.

All environment variables, paths, and tunable parameters live here so the
rest of the codebase never reads from os.environ directly. This makes the
project easy to reconfigure (e.g. swap LLM providers) without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env file (from project root) into environment variables.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Paths:
    """All filesystem paths used by the project."""

    root: Path = PROJECT_ROOT
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    chroma_db: Path = PROJECT_ROOT / "data" / "chroma_db"
    docs: Path = PROJECT_ROOT / "docs"

    def ensure(self) -> None:
        """Create directories if they don't exist."""
        self.data_raw.mkdir(parents=True, exist_ok=True)
        self.chroma_db.mkdir(parents=True, exist_ok=True)


PATHS = Paths()


# ---------------------------------------------------------------------------
# LLM provider configuration
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Azure OpenAI (optional)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# OpenAI (optional)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ---------------------------------------------------------------------------
# Embedding model (local, free)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)


# ---------------------------------------------------------------------------
# RAG tuning parameters
# ---------------------------------------------------------------------------
CHUNK_SIZE = 800           # characters per chunk
CHUNK_OVERLAP = 120        # overlap between chunks for context continuity
TOP_K = 5                  # number of chunks retrieved per query
CHROMA_COLLECTION = "eu_ai_act"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def validate_llm_config() -> None:
    """Raise a clear error if the chosen LLM provider isn't configured."""
    if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        raise RuntimeError(
            "LLM_PROVIDER=groq but GROQ_API_KEY is not set. "
            "Add it to your .env file (see .env.example)."
        )
    if LLM_PROVIDER == "azure_openai" and not all(
        [AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT]
    ):
        raise RuntimeError(
            "LLM_PROVIDER=azure_openai but Azure credentials are incomplete."
        )
    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        raise RuntimeError(
            "LLM_PROVIDER=openai but OPENAI_API_KEY is not set."
        )


if __name__ == "__main__":
    # Quick sanity check when running `python -m src.config`
    PATHS.ensure()
    print(f"Project root:      {PATHS.root}")
    print(f"Data raw:          {PATHS.data_raw}")
    print(f"Chroma DB:         {PATHS.chroma_db}")
    print(f"LLM provider:      {LLM_PROVIDER}")
    print(f"Embedding model:   {EMBEDDING_MODEL_NAME}")
    print(f"Chunk size:        {CHUNK_SIZE} (overlap {CHUNK_OVERLAP})")
    print(f"Top-K:             {TOP_K}")
    validate_llm_config()
    print("✓ Configuration valid.")