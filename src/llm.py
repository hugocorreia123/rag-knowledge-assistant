"""LLM provider abstraction.

The rest of the codebase calls ``get_llm()`` and receives a LangChain-compatible
chat model. The actual provider (Groq, OpenAI, Azure OpenAI) is selected by the
``LLM_PROVIDER`` environment variable, so switching providers is a one-line
change in ``.env`` — no code edits required.

This is the pattern that makes the same code run on a free Groq tier for the
public demo and on enterprise Azure OpenAI in production.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from langchain_core.language_models import BaseChatModel

from src import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider factories
# ---------------------------------------------------------------------------
def _build_groq(temperature: float) -> BaseChatModel:
    """Build a Groq chat model."""
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=config.GROQ_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=temperature,
        max_retries=2,
    )


def _build_openai(temperature: float) -> BaseChatModel:
    """Build an OpenAI chat model."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=config.OPENAI_MODEL,
        api_key=config.OPENAI_API_KEY,
        temperature=temperature,
        max_retries=2,
    )


def _build_azure_openai(temperature: float) -> BaseChatModel:
    """Build an Azure OpenAI chat model."""
    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(
        deployment_name=config.AZURE_OPENAI_DEPLOYMENT,
        api_key=config.AZURE_OPENAI_API_KEY,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_version="2024-08-01-preview",
        temperature=temperature,
        max_retries=2,
    )


_BUILDERS = {
    "groq": _build_groq,
    "openai": _build_openai,
    "azure_openai": _build_azure_openai,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def get_llm(temperature: float = 0.0, provider: Optional[str] = None) -> BaseChatModel:
    """Return a LangChain chat model for the configured provider.

    Args:
        temperature: Sampling temperature. 0.0 = deterministic (recommended for
            RAG over factual documents), higher = more creative.
        provider: Override the configured provider. Defaults to the value of
            ``LLM_PROVIDER`` in the environment.

    Cached so identical calls reuse the same client (avoids opening new HTTP
    connections per request).
    """
    provider = (provider or config.LLM_PROVIDER).lower()

    if provider not in _BUILDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{provider}'. "
            f"Supported: {sorted(_BUILDERS)}."
        )

    config.validate_llm_config()
    log.info("Building LLM (provider=%s, temperature=%.2f)", provider, temperature)
    return _BUILDERS[provider](temperature)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
    llm = get_llm()
    response = llm.invoke("Say 'hello from the EU AI Act assistant' in one short line.")
    print("\nLLM response:", response.content)