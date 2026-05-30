"""Streamlit chat UI for the RAG Knowledge Assistant.

Run with:
    streamlit run app/streamlit_app.py

The app stays small on purpose — all RAG logic lives in src/. This file
only handles UI state and rendering.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable when Streamlit launches this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.graph import ask
from src.retriever import RetrievedChunk


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Knowledge Assistant — EU AI Act",
    page_icon="🔎",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Sidebar — project info + sample questions
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔎 RAG Assistant")
    st.markdown(
        """
        **About this app**

        Ask questions about the **EU AI Act**
        *(Regulation (EU) 2024/1689)*.

        The assistant uses a multi-step
        **LangGraph** workflow:
        rewrite → retrieve → grade →
        answer (or refuse).

        ---
        """
    )

    st.markdown("**Try a sample question:**")
    sample_questions = [
        "Which AI practices are prohibited under the EU AI Act?",
        "What are the obligations for providers of high-risk AI systems?",
        "What is the definition of a general-purpose AI model?",
        "What penalties apply for non-compliance?",
        "What are the transparency obligations for AI systems?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True, key=f"sample_{q[:20]}"):
            st.session_state["pending_question"] = q

    st.markdown(
        """
        ---
        **Tech stack**
        - Groq · Llama 3.3 70B
        - LangChain + LangGraph
        - Chroma vector store
        - sentence-transformers
        - Streamlit

        Built by **[Hugo Correia](https://www.linkedin.com/in/hugogncorreia)**
        · [GitHub](https://github.com/hugocorreia123/rag-knowledge-assistant)
        """
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("RAG Knowledge Assistant")
st.caption("Ask questions about the EU AI Act — answers grounded in the official text.")


# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # list of {"role": "user|assistant", "content": str, "chunks": [...]}


# ---------------------------------------------------------------------------
# Render conversation history
# ---------------------------------------------------------------------------
def render_chunks(chunks: list[RetrievedChunk]) -> None:
    """Render the retrieved chunks inside an expander."""
    with st.expander(f"📄 Show retrieved context ({len(chunks)} chunks)"):
        for i, c in enumerate(chunks, start=1):
            st.markdown(
                f"**[{i}] {c.cite()}**  "
                f"<span style='color:gray'>similarity {c.similarity:.3f}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"> {c.text.strip()}")
            st.markdown("---")


for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("citations"):
            st.caption("Sources: " + ", ".join(f"`{c}`" for c in msg["citations"]))
        if msg["role"] == "assistant" and msg.get("chunks"):
            render_chunks(msg["chunks"])


# ---------------------------------------------------------------------------
# Input — handle both chat input and sample-question clicks
# ---------------------------------------------------------------------------
question = st.chat_input("Ask anything about the EU AI Act…")

# A sample-question button was clicked
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")


# ---------------------------------------------------------------------------
# Run the workflow on a new question
# ---------------------------------------------------------------------------
if question:
    # Echo user message
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Run the graph and stream a loading spinner
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = ask(question)

        answer = result.get("answer", "")
        citations = result.get("citations", [])
        chunks = result.get("chunks", [])

        st.markdown(answer)
        if citations:
            st.caption("Sources: " + ", ".join(f"`{c}`" for c in citations))
        if chunks and citations:
            render_chunks(chunks)

    # Persist for re-render
    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
        "chunks": chunks if citations else None,  # only show chunks for real answers
    })