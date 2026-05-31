"""LangGraph workflow: rewrite → retrieve → grade → answer (or refuse).

This is the orchestration layer that turns retrieval into a real AI assistant
with guardrails. Each node is a single-responsibility function, and the state
flows between them as a typed dict.

The graph is the difference between "retrieve-and-answer" RAG (hallucination-
prone) and a small AI system that knows when it doesn't know.
"""

from __future__ import annotations

import logging
from typing import List, Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from src.llm import get_llm
from src.retriever import RetrievedChunk, format_context, retrieve

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Workflow state
# ---------------------------------------------------------------------------
class GraphState(TypedDict, total=False):
    """State carried through the workflow.

    Each node reads what it needs and writes its outputs. LangGraph merges
    partial dicts returned by each node into this state.
    """

    question: str                    # the user's original question
    rewritten: str                   # cleaned/expanded query for retrieval
    chunks: List[RetrievedChunk]     # retrieved chunks
    is_relevant: bool                # grader verdict
    answer: str                      # final answer (or refusal)
    citations: List[str]             # citation tags used in the answer


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
REWRITE_SYSTEM = """You optionally rewrite user questions to improve document retrieval.

Rules:
- If the question is already clear, specific, and well-formed, return it UNCHANGED.
- Only rewrite when the question is ambiguous, uses slang, or omits key context.
- Keep rewrites short — same length or shorter than the original.
- Output ONLY the (possibly unchanged) question. No preamble, no explanation, no quotes."""

GRADE_SYSTEM = """You are deciding whether to attempt an answer.

The context below contains retrieved passages from a regulatory document. Reply YES if ANY of the passages mention or relate to the topic of the question, even tangentially. Reply NO only if NONE of the passages are about the topic at all.

When unsure, say YES. The answering step will handle nuance — your only job is to filter out completely unrelated questions (like asking about cooking when the document is about law).

Reply with exactly one word: YES or NO."""

ANSWER_SYSTEM = """You are an assistant specialized in the EU AI Act (Regulation (EU) 2024/1689).

Rules:
- Answer ONLY using the provided context. Never invent facts.
- Cite the source of every claim using the citation tags shown in the context (e.g. [eu_ai_act.pdf p.42]).
- Be concise and precise. Prefer direct quotes for definitions and obligations.
- If the context is insufficient, say so explicitly rather than guessing.
- Write in clear, professional English."""

REFUSE_MESSAGE = (
    "I couldn't find a confident answer to that question in the EU AI Act. "
    "Try rephrasing, or ask about a different topic covered by the Act "
    "(e.g. high-risk AI systems, prohibited practices, penalties, "
    "general-purpose AI models)."
)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def node_rewrite(state: GraphState) -> GraphState:
    """Improve the question for better retrieval — but only if needed.

    Most clear questions retrieve better as-is. We only rewrite when the
    question contains slang, typos, or unresolved references.
    """
    q = state["question"].strip()

    # Heuristic: if the question is already well-formed (capital start,
    # reasonable length, ends with ?), skip the LLM rewrite entirely.
    looks_clean = (
        len(q) >= 15
        and q[0].isupper()
        and q.endswith("?")
        and "?" not in q[:-1]  # only one question mark
    )

    if looks_clean:
        log.info("Rewrite: skipped (question already well-formed)")
        return {"rewritten": q}

    llm = get_llm(temperature=0.0)
    response = llm.invoke([
        SystemMessage(content=REWRITE_SYSTEM),
        HumanMessage(content=q),
    ])
    rewritten = response.content.strip()
    log.info("Rewrite: %r → %r", q, rewritten)
    return {"rewritten": rewritten}


def node_retrieve(state: GraphState) -> GraphState:
    """Semantic search using the rewritten query."""
    query = state.get("rewritten") or state["question"]
    chunks = retrieve(query)
    return {"chunks": chunks}


def node_grade(state: GraphState) -> GraphState:
    """Decide whether the retrieved context is relevant to the question."""
    chunks = state["chunks"]
    if not chunks:
        return {"is_relevant": False}

    context = format_context(chunks)
    llm = get_llm(temperature=0.0)
    response = llm.invoke([
        SystemMessage(content=GRADE_SYSTEM),
        HumanMessage(content=f"QUESTION:\n{state['question']}\n\nCONTEXT:\n{context}"),
    ])
    verdict = response.content.strip().upper().rstrip(".")
    is_relevant = verdict.startswith("YES")
    log.info("Grade verdict: %s (raw=%r)", "RELEVANT" if is_relevant else "IRRELEVANT", response.content)
    return {"is_relevant": is_relevant}


def node_answer(state: GraphState) -> GraphState:
    """Generate the final answer grounded in retrieved context."""
    chunks = state["chunks"]
    context = format_context(chunks)
    llm = get_llm(temperature=0.0)
    response = llm.invoke([
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"QUESTION:\n{state['question']}\n\nCONTEXT:\n{context}"),
    ])
    answer = response.content.strip()
    citations = sorted({c.cite() for c in chunks if c.cite() in answer})
    log.info("Answer generated (%d chars, %d citations used)", len(answer), len(citations))
    return {"answer": answer, "citations": citations}


def node_refuse(state: GraphState) -> GraphState:
    """Polite refusal when context is insufficient."""
    log.info("Refusing — insufficient context.")
    return {"answer": REFUSE_MESSAGE, "citations": []}


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------
def route_after_grade(state: GraphState) -> Literal["answer", "refuse"]:
    return "answer" if state.get("is_relevant") else "refuse"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graph():
    """Assemble the LangGraph state machine."""
    workflow = StateGraph(GraphState)

    workflow.add_node("rewrite", node_rewrite)
    workflow.add_node("retrieve", node_retrieve)
    workflow.add_node("grade", node_grade)
    workflow.add_node("generate_answer", node_answer)
    workflow.add_node("generate_refusal", node_refuse)

    workflow.add_edge(START, "rewrite")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges(
        "grade",
        route_after_grade,
        {"answer": "generate_answer", "refuse": "generate_refusal"},
    )
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("generate_refusal", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def ask(question: str) -> GraphState:
    """Run the full workflow on a single question."""
    graph = build_graph()
    return graph.invoke({"question": question})


def ask_direct(question: str) -> GraphState:
    """Token-efficient variant for evaluation: skip rewrite and grade.

    Runs retrieve → answer directly. Saves ~2/3 of the LLM tokens compared
    to the full graph, at the cost of losing rewrite-quality and grade-based
    refusal. Out-of-scope refusal still works via the answer model's own
    "context insufficient" output.
    """
    chunks = retrieve(question)
    state: GraphState = {
        "question": question,
        "rewritten": question,
        "chunks": chunks,
        "is_relevant": True,  # bypass the grader
    }
    state.update(node_answer(state))

    # Refusal heuristic without grader call: if the answer literally says
    # the context is insufficient and cites nothing, mark it as a refusal.
    if not state.get("citations") and any(
        phrase in state.get("answer", "").lower()
        for phrase in ("insufficient", "i couldn't find", "does not contain", "no information")
    ):
        state["answer"] = REFUSE_MESSAGE
        state["citations"] = []
    return state


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    questions = [
        "What penalties apply for non-compliance with the EU AI Act?",
        "What's the best pizza in Lisbon?",  # out-of-scope — should refuse
    ]

    for q in questions:
        print("\n" + "=" * 80)
        print(f"Q: {q}")
        print("=" * 80)
        result = ask(q)
        print(f"\nA: {result['answer']}")
        if result.get("citations"):
            print(f"\nCitations: {', '.join(result['citations'])}")