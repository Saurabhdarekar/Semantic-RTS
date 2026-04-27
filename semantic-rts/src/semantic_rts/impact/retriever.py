"""Phase 2 retriever: embed intent → FAISS top-K → Candidate list."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from semantic_rts.config import Config
    from semantic_rts.impact.intent_agent import IntentResult
    from semantic_rts.kb.vector_store import VectorStore
    from semantic_rts.llm.embeddings import GeminiEmbedder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    test_id: str
    score: float
    tier: int
    rank: int          # 0-indexed position in top-K


@dataclass
class AnalysisTrace:
    diff_hash: str
    files_changed: list[str]
    methods_changed: list[str]
    intent_summary: str
    intent_concepts: list[str]
    risk_areas: list[str]
    query_text: str
    top_k: int
    raw_results: list[Candidate]
    intent_failed: bool = False


# ---------------------------------------------------------------------------
# Query formatting (must mirror Phase 1 embedding format)
# ---------------------------------------------------------------------------

def format_query(intent: "IntentResult", methods_changed: list[str] | None = None) -> str:
    """Format intent into a natural-language query string.

    Mirrors format_for_embedding() in kb/builder.py so query and document
    vectors sit in the same embedding space. Including changed method simple
    names lets FAISS directly match KB entries that list those methods under
    'Methods under test'.
    """
    parts = [intent.intent_summary]
    if methods_changed:
        simple = ", ".join(m.split(".")[-1] for m in methods_changed if "." in m) or ", ".join(methods_changed)
        parts.append(f"Methods changed: {simple}.")
    if intent.concepts:
        parts.append(f"Concepts: {', '.join(intent.concepts)}.")
    if intent.risk_areas:
        parts.append(f"Risk areas: {', '.join(intent.risk_areas)}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(
    intent: "IntentResult",
    kb: "VectorStore",
    embedder: "GeminiEmbedder",
    config: "Config",
    diff_hash: str = "",
    files_changed: list[str] | None = None,
    methods_changed: list[str] | None = None,
) -> tuple[list[Candidate], AnalysisTrace]:
    """Embed intent summary and retrieve top-K candidates from the KB.

    Returns (candidates, trace).  candidates is empty if the KB is empty.
    """
    files_changed = files_changed or []
    methods_changed = methods_changed or []

    query_text = format_query(intent, methods_changed)

    if kb.size == 0:
        logger.warning("KB is empty — returning no candidates.")
        trace = AnalysisTrace(
            diff_hash=diff_hash,
            files_changed=files_changed,
            methods_changed=methods_changed,
            intent_summary=intent.intent_summary,
            intent_concepts=intent.concepts,
            risk_areas=intent.risk_areas,
            query_text=query_text,
            top_k=config.retrieval.top_k,
            raw_results=[],
            intent_failed=intent.intent_failed,
        )
        return [], trace

    query_emb = embedder.embed(query_text)
    scores, test_ids = kb.search(query_emb, k=config.retrieval.top_k)

    candidates: list[Candidate] = []
    for rank, (score, test_id) in enumerate(zip(scores, test_ids)):
        tier = kb.tier_for_id(test_id)
        candidates.append(Candidate(
            test_id=test_id,
            score=float(score),
            tier=tier,
            rank=rank,
        ))

    trace = AnalysisTrace(
        diff_hash=diff_hash,
        files_changed=files_changed,
        methods_changed=methods_changed,
        intent_summary=intent.intent_summary,
        intent_concepts=intent.concepts,
        risk_areas=intent.risk_areas,
        query_text=query_text,
        top_k=config.retrieval.top_k,
        raw_results=list(candidates),
        intent_failed=intent.intent_failed,
    )

    return candidates, trace
