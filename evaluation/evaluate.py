"""Evaluation runner for the RAG Knowledge Assistant.

Runs the full LangGraph workflow against the hand-written test set in
``evaluation/test_questions.yaml`` and reports metrics.

Run with:
    make eval
    # or directly:
    python -m evaluation.evaluate

Produces:
    - Console summary table
    - ``evaluation/results.md`` — a Markdown report linked from the README
"""

from __future__ import annotations

import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

# Make src/ importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PATHS, TOP_K
from src.graph import ask_direct as ask  # token-efficient variant
from src.retriever import retrieve

logging.basicConfig(
    level=logging.WARNING,  # quiet the per-question logs; we print our own summary
    format="%(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("eval")
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class TestCase:
    id: str
    category: str
    question: str
    expected_pages: List[int]
    key_facts: List[str]
    should_answer: bool


@dataclass
class CaseResult:
    case: TestCase
    answer: str
    retrieved_pages: List[int]
    citations: List[str]
    was_answered: bool
    latency_s: float

    # Metrics (computed below)
    recall_at_k: Optional[float] = None    # in 0..1 — fraction of expected pages found in top-K
    precision_at_5: Optional[float] = None # in 0..1 — fraction of top-5 from expected pages
    faithfulness: Optional[float] = None   # in 0..1 — fraction of key_facts substring-matched
    refusal_correct: Optional[bool] = None # for out_of_scope cases only


# ---------------------------------------------------------------------------
# Load test set
# ---------------------------------------------------------------------------
TEST_PATH = Path(__file__).parent / "test_questions.yaml"


def load_cases(path: Path = TEST_PATH) -> List[TestCase]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return [TestCase(**entry) for entry in raw]


# ---------------------------------------------------------------------------
# Per-case execution
# ---------------------------------------------------------------------------
def run_case(case: TestCase, max_retries: int = 5) -> CaseResult:
    """Run the full graph on one question, with retry on rate limits."""
    import random

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            t0 = time.perf_counter()
            state = ask(case.question)
            latency = time.perf_counter() - t0
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            msg = str(exc).lower()
            # Retry on rate-limit-like errors with exponential backoff + jitter.
            if "rate_limit" in msg or "429" in msg or "rate limit" in msg:
                wait = (2 ** attempt) + random.uniform(0, 1)
                log.warning(
                    "Rate-limited on %s (attempt %d/%d). Sleeping %.1fs.",
                    case.id, attempt + 1, max_retries, wait,
                )
                time.sleep(wait)
                continue
            # Non-retryable error — bubble up.
            raise
    else:
        # All retries exhausted.
        raise RuntimeError(
            f"Exceeded {max_retries} rate-limit retries on {case.id}"
        ) from last_exc

    chunks = state.get("chunks", []) or []
    return CaseResult(
        case=case,
        answer=state.get("answer", ""),
        retrieved_pages=[c.page for c in chunks],
        citations=state.get("citations", []) or [],
        was_answered=bool(state.get("citations")),
        latency_s=round(latency, 2),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(r: CaseResult) -> None:
    """Compute per-case metrics in-place."""
    case = r.case

    # Out-of-scope: only the refusal metric matters.
    if not case.should_answer:
        r.refusal_correct = (not r.was_answered)
        return

    # In-scope: retrieval + faithfulness.
    expected = set(case.expected_pages)
    top_pages_k = r.retrieved_pages[:TOP_K]
    top_pages_5 = r.retrieved_pages[:5]

    r.recall_at_k = (
        len(expected & set(top_pages_k)) / len(expected) if expected else None
    )
    r.precision_at_5 = (
        sum(1 for p in top_pages_5 if p in expected) / 5 if top_pages_5 else 0.0
    )

    # Substring-based faithfulness — pragmatic, no LLM-judge dependency.
    if case.key_facts:
        ans_lower = r.answer.lower()
        hits = sum(1 for fact in case.key_facts if fact.lower() in ans_lower)
        r.faithfulness = hits / len(case.key_facts)
    else:
        r.faithfulness = None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
@dataclass
class Aggregates:
    n_total: int = 0
    n_inscope: int = 0
    n_outscope: int = 0
    avg_recall_at_k: float = 0.0
    avg_precision_at_5: float = 0.0
    avg_faithfulness: float = 0.0
    refusal_accuracy: float = 0.0
    median_latency_s: float = 0.0
    per_category: dict = field(default_factory=dict)


def aggregate(results: List[CaseResult]) -> Aggregates:
    agg = Aggregates(n_total=len(results))

    inscope = [r for r in results if r.case.should_answer]
    outscope = [r for r in results if not r.case.should_answer]

    agg.n_inscope = len(inscope)
    agg.n_outscope = len(outscope)

    def _mean(values):
        values = [v for v in values if v is not None]
        return round(statistics.mean(values), 3) if values else 0.0

    agg.avg_recall_at_k = _mean([r.recall_at_k for r in inscope])
    agg.avg_precision_at_5 = _mean([r.precision_at_5 for r in inscope])
    agg.avg_faithfulness = _mean([r.faithfulness for r in inscope])

    if outscope:
        agg.refusal_accuracy = round(
            sum(1 for r in outscope if r.refusal_correct) / len(outscope), 3
        )

    agg.median_latency_s = round(statistics.median([r.latency_s for r in results]), 2)

    # Per-category breakdown
    categories = {}
    for r in results:
        categories.setdefault(r.case.category, []).append(r)
    for cat, items in categories.items():
        in_items = [x for x in items if x.case.should_answer]
        agg.per_category[cat] = {
            "n": len(items),
            "recall@k": _mean([x.recall_at_k for x in in_items]),
            "faithfulness": _mean([x.faithfulness for x in in_items]),
        }

    return agg


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
RESULTS_PATH = Path(__file__).parent / "results.md"


def render_console(results: List[CaseResult], agg: Aggregates) -> None:
    print()
    print("=" * 90)
    print(f"Evaluation summary — {agg.n_total} cases ({agg.n_inscope} in-scope, "
          f"{agg.n_outscope} out-of-scope)")
    print("=" * 90)

    print(f"  Recall@{TOP_K} (in-scope avg)   : {agg.avg_recall_at_k:.3f}")
    print(f"  Precision@5 (in-scope avg)     : {agg.avg_precision_at_5:.3f}")
    print(f"  Faithfulness (in-scope avg)    : {agg.avg_faithfulness:.3f}")
    print(f"  Refusal accuracy (out-of-scope): {agg.refusal_accuracy:.3f}")
    print(f"  Median latency (s)             : {agg.median_latency_s}")

    print("\nPer-category:")
    print(f"  {'category':<14}{'n':>4}  {'recall@k':>10}  {'faithfulness':>14}")
    for cat, m in sorted(agg.per_category.items()):
        print(f"  {cat:<14}{m['n']:>4}  {m['recall@k']:>10.3f}  {m['faithfulness']:>14.3f}")

    print("\nPer-case:")
    print(f"  {'id':<24}{'cat':<14}{'ok?':<6}{'recall@k':>10}{'lat(s)':>9}")
    for r in results:
        ok = ("✓" if (not r.case.should_answer and r.refusal_correct)
              or (r.case.should_answer and r.recall_at_k and r.recall_at_k > 0)
              else "✗")
        recall = f"{r.recall_at_k:.2f}" if r.recall_at_k is not None else "—"
        print(f"  {r.case.id:<24}{r.case.category:<14}{ok:<6}{recall:>10}{r.latency_s:>9}")


def render_markdown(results: List[CaseResult], agg: Aggregates) -> str:
    """Generate a Markdown report. Returns the rendered string."""
    out = []
    out.append("# Evaluation results\n")
    out.append("_Auto-generated by `make eval` (i.e. `python -m evaluation.evaluate`)._\n")
    out.append(f"\n**Test set:** {agg.n_total} cases — {agg.n_inscope} in-scope, "
               f"{agg.n_outscope} out-of-scope.\n")

    out.append("\n## Summary metrics\n")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| Retrieval Recall@{TOP_K} (in-scope avg) | **{agg.avg_recall_at_k:.3f}** |")
    out.append(f"| Retrieval Precision@5 (in-scope avg) | **{agg.avg_precision_at_5:.3f}** |")
    out.append(f"| Answer Faithfulness (in-scope avg) | **{agg.avg_faithfulness:.3f}** |")
    out.append(f"| Out-of-scope Refusal Accuracy | **{agg.refusal_accuracy:.3f}** |")
    out.append(f"| Median end-to-end latency | **{agg.median_latency_s:.2f} s** |")

    out.append("\n## Per-category breakdown\n")
    out.append("| Category | N | Recall@K | Faithfulness |")
    out.append("|---|---:|---:|---:|")
    for cat, m in sorted(agg.per_category.items()):
        out.append(
            f"| {cat} | {m['n']} | {m['recall@k']:.3f} | {m['faithfulness']:.3f} |"
        )

    out.append("\n## Per-case results\n")
    out.append("| ID | Category | Should answer? | Answered? | Recall@K | Faithfulness | Latency (s) |")
    out.append("|---|---|:-:|:-:|---:|---:|---:|")
    for r in results:
        should = "Yes" if r.case.should_answer else "No"
        answered = "Yes" if r.was_answered else "No"
        recall = f"{r.recall_at_k:.2f}" if r.recall_at_k is not None else "—"
        faith = f"{r.faithfulness:.2f}" if r.faithfulness is not None else "—"
        out.append(
            f"| `{r.case.id}` | {r.case.category} | {should} | {answered} "
            f"| {recall} | {faith} | {r.latency_s:.2f} |"
        )

    out.append("\n## Notes\n")
    out.append("- **Recall@K**: fraction of expected pages found in the top-K retrieved chunks.")
    out.append("- **Precision@5**: fraction of the top-5 chunks coming from expected pages.")
    out.append("- **Faithfulness**: fraction of `key_facts` (from the test set) substring-matched in the answer.")
    out.append("- **Refusal accuracy**: fraction of out-of-scope questions correctly refused.")
    out.append("- All metrics are computed by `evaluation/evaluate.py` against the test set in "
               "`evaluation/test_questions.yaml`.\n")

    return "\n".join(out)

def render_diagnostic(results: List[CaseResult]) -> str:
    """Show what was retrieved per case — used to calibrate expected_pages."""
    out = ["# Eval diagnostic — retrieved pages per case\n"]
    out.append("Use this to verify or correct `expected_pages` in `test_questions.yaml`.\n")
    for r in results:
        out.append(f"\n## `{r.case.id}` — {r.case.category}\n")
        out.append(f"**Question:** {r.case.question}\n")
        out.append(f"**Expected pages (current):** {r.case.expected_pages}\n")
        out.append(f"**Retrieved pages (top {len(r.retrieved_pages)}):** {r.retrieved_pages}\n")
        out.append(f"**Answered?** {r.was_answered}  ·  **Latency:** {r.latency_s}s\n")
        if r.answer:
            preview = r.answer[:280].replace("\n", " ")
            out.append(f"**Answer preview:** {preview}…\n")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = Path(__file__).parent / ".checkpoint.json"


def main() -> int:
    import json

    cases = load_cases()
    log.info("Loaded %d test cases from %s", len(cases), TEST_PATH.name)

    # Resume from checkpoint if one exists.
    done_ids: set[str] = set()
    results: List[CaseResult] = []
    if CHECKPOINT_PATH.exists():
        log.info("Found checkpoint — resuming from %s", CHECKPOINT_PATH.name)
        raw = json.loads(CHECKPOINT_PATH.read_text())
        done_ids = set(raw["done_ids"])
        # We can't fully serialize CaseResult to JSON (RetrievedChunk inside),
        # so we re-run only the missing cases. Old results are preserved
        # in the markdown report from the previous partial run, but for a
        # clean numerical re-run, delete the checkpoint to start fresh.

    try:
        for i, case in enumerate(cases, start=1):
            if case.id in done_ids:
                log.info("[%2d/%2d] %s — skipped (already in checkpoint)",
                         i, len(cases), case.id)
                continue

            log.info("[%2d/%2d] %s — %s",
                     i, len(cases), case.id, case.question[:60])
            r = run_case(case)
            compute_metrics(r)
            results.append(r)
            done_ids.add(case.id)

            # Persist incremental progress.
            CHECKPOINT_PATH.write_text(
                json.dumps({"done_ids": sorted(done_ids)}, indent=2)
            )
    except KeyboardInterrupt:
        log.warning("Interrupted. Partial progress saved.")
    except Exception as exc:  # noqa: BLE001
        log.error("Eval stopped: %s", exc)
        log.info("Partial progress saved. Re-run `make eval` to resume.")
        raise

    if not results:
        log.warning("No new results to render — nothing was run.")
        return 0

    agg = aggregate(results)
    render_console(results, agg)

    markdown = render_markdown(results, agg)
    RESULTS_PATH.write_text(markdown, encoding="utf-8")
    (RESULTS_PATH.parent / "diagnostic.md").write_text(
        render_diagnostic(results), encoding="utf-8"
    )
    log.info("Wrote %s and diagnostic.md", RESULTS_PATH.name)

    # Clean checkpoint when run completes successfully.
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        log.info("Cleared checkpoint.")

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())