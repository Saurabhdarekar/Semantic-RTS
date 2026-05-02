"""Phase 2 retriever: embed intent → FAISS top-K → Candidate list."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from semantic_rts.config import Config
    from semantic_rts.impact.intent_agent import IntentResult
    from semantic_rts.kb.vector_store import VectorStore
    from semantic_rts.llm.client import GeminiClient
    from semantic_rts.llm.embeddings import GeminiEmbedder

from semantic_rts.kb.embed_format import format_for_embedding as _canonical_format

logger = logging.getLogger(__name__)

_CAMEL_SPLIT_RE = re.compile(r'([A-Z]+)([A-Z][a-z])|([a-z\d])([A-Z])')


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
    low_confidence_retrieval: bool = False
    effective_threshold: float = 0.0


# ---------------------------------------------------------------------------
# Query formatting — delegates to shared canonical format
# ---------------------------------------------------------------------------

def format_query(intent: "IntentResult", methods_changed: list[str] | None = None) -> str:
    return _canonical_format(
        summary=intent.intent_summary,
        methods=methods_changed or [],
        concepts=intent.concepts,
        risk_areas=intent.risk_areas,
    )


# ---------------------------------------------------------------------------
# Step 3.1 — CamelCase token overlap
# ---------------------------------------------------------------------------

def _camel_tokens(name: str) -> set[str]:
    """Split camelCase/PascalCase into lowercase tokens, dropping single-char ones."""
    spaced = _CAMEL_SPLIT_RE.sub(r'\1\3 \2\4', name)
    return {t.lower() for t in re.split(r'[\s_\-]+', spaced) if len(t) > 1}


def _token_overlap_score(
    test_id: str,
    changed_methods: list[str],
    changed_files: list[str],
) -> float:
    """Jaccard token overlap between test name tokens and changed entity tokens."""
    method_part = test_id.split("::")[-1] if "::" in test_id else test_id
    class_part = test_id.split("::")[0].split(".")[-1] if "::" in test_id else ""
    test_tokens = _camel_tokens(method_part) | _camel_tokens(class_part)
    if not test_tokens:
        return 0.0

    best = 0.0
    for method in changed_methods:
        simple = method.split(".")[-1] if "." in method else method
        change_tokens = _camel_tokens(simple)
        if not change_tokens:
            continue
        union = test_tokens | change_tokens
        best = max(best, len(test_tokens & change_tokens) / len(union))

    # File stem match at 0.6× weight — less precise than method name
    for f in changed_files:
        stem_tokens = _camel_tokens(Path(f).stem)
        if not stem_tokens:
            continue
        union = test_tokens | stem_tokens
        best = max(best, 0.6 * len(test_tokens & stem_tokens) / len(union))

    return best


# ---------------------------------------------------------------------------
# Step 3.2 — Package proximity
# ---------------------------------------------------------------------------

def _package_proximity_score(test_id: str, changed_files: list[str]) -> float:
    """Score by Java package prefix overlap between test class and changed files."""
    if "::" not in test_id:
        return 0.0
    test_fqn = test_id.split("::")[0]
    test_pkg_parts = test_fqn.split(".")[:-1]   # drop class name

    best = 0.0
    for changed_file in changed_files:
        parts = Path(changed_file).with_suffix("").parts
        try:
            java_idx = next(i for i, p in enumerate(parts) if p == "java")
            changed_pkg_parts = list(parts[java_idx + 1: -1])   # drop class name
        except StopIteration:
            continue
        if not changed_pkg_parts:
            continue
        matching = sum(1 for a, b in zip(test_pkg_parts, changed_pkg_parts) if a == b)
        max_len = max(len(test_pkg_parts), len(changed_pkg_parts), 1)
        best = max(best, matching / max_len)

    return best


# ---------------------------------------------------------------------------
# Step 3.6 — Optional cross-encoder reranking
# ---------------------------------------------------------------------------

def _cross_encode_filter(
    candidates: list["Candidate"],
    intent_summary: str,
    kb_summaries: dict[str, str],
    client: "GeminiClient",
    top_n: int,
) -> list["Candidate"]:
    """Binary YES/NO LLM pass over top_n candidates. Non-top-n are always kept."""
    top = candidates[:top_n]
    rest = candidates[top_n:]
    keep: list["Candidate"] = []
    for candidate in top:
        summary = kb_summaries.get(candidate.test_id, candidate.test_id)
        prompt = (
            f"Code change summary: {intent_summary[:400]}\n"
            f"Test description: {summary}\n"
            "Could this code change plausibly cause this test to fail? "
            "Answer YES or NO only."
        )
        try:
            result = client.chat(prompt, version_tag="CROSS_ENCODER_V1")
            if "YES" in result["text"].upper():
                keep.append(candidate)
        except Exception:
            keep.append(candidate)   # on error, keep (safe default)
    return keep + rest


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _build_method_idf(tested_methods_map: "dict[str, list[str]]") -> "dict[str, float]":
    """Compute IDF weight per method simple name across the KB.

    Methods exercised by few tests get high weight (rare = discriminative).
    Methods exercised by most tests get low weight (common = noise).
    """
    import math
    total = len(tested_methods_map)
    if total == 0:
        return {}
    freq: dict[str, int] = {}
    for methods in tested_methods_map.values():
        for m in methods:
            simple = m.split(".")[-1]
            freq[simple] = freq.get(simple, 0) + 1
    return {m: math.log((total + 1) / (c + 1)) for m, c in freq.items()}


def _method_boost(
    test_id: str,
    tested_methods_map: "dict[str, list[str]]",
    changed_simple: set[str],
    idf: "dict[str, float]",
    max_boost: float = 0.15,
) -> float:
    """IDF-weighted boost: rare matching methods contribute more than common ones."""
    tested = tested_methods_map.get(test_id, [])
    score = sum(
        idf.get(m.split(".")[-1], 0.0)
        for m in tested
        if m.split(".")[-1] in changed_simple
    )
    if score == 0:
        return 0.0
    # Normalise: cap at max_boost using tanh so very high IDF scores don't dominate
    import math
    return max_boost * math.tanh(score / 3.0)


def retrieve(
    intent: "IntentResult",
    kb: "VectorStore",
    embedder: "GeminiEmbedder",
    config: "Config",
    diff_hash: str = "",
    files_changed: list[str] | None = None,
    methods_changed: list[str] | None = None,
    tested_methods_map: "dict[str, list[str]] | None" = None,
    client: "GeminiClient | None" = None,
    kb_summaries: "dict[str, str] | None" = None,
) -> tuple[list[Candidate], AnalysisTrace]:
    """Embed intent and retrieve top-K candidates using FAISS + hybrid signals.

    Signal pipeline (in order):
      1. FAISS cosine similarity (base)
      2. IDF-weighted method intersection boost
      3. CamelCase token overlap boost (3.1)
      4. Package proximity boost (3.2)
      5. BM25 blend (3.3)
      6. Negative pass — penalise clearly unrelated low-tier tests (3.4)
      7. Adaptive threshold (3.5)
      8. Optional cross-encoder filter (3.6)
    """
    files_changed = files_changed or []
    methods_changed = methods_changed or []

    query_text = format_query(intent, methods_changed)

    if kb.size == 0:
        logger.warning("KB is empty — returning no candidates.")
        return [], AnalysisTrace(
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

    # Fetch a wider pool so re-ranking doesn't miss boosted tests
    fetch_k = min(kb.size, config.retrieval.top_k * 2)
    query_emb = embedder.embed(query_text)
    scores, test_ids = kb.search(query_emb, k=fetch_k)

    # --- 1+2: Base FAISS score + IDF method-intersection boost ---
    changed_simple = {m.split(".")[-1] for m in methods_changed if "." in m}
    idf = _build_method_idf(tested_methods_map) if tested_methods_map else {}
    boosted: list[tuple[float, str]] = []
    for score, test_id in zip(scores, test_ids):
        boost = 0.0
        if tested_methods_map and changed_simple:
            boost = _method_boost(test_id, tested_methods_map, changed_simple, idf)
        boosted.append((float(score) + boost, test_id))

    # --- 3: CamelCase token overlap ---
    overlap_w = config.retrieval.token_overlap_weight
    for i, (score, test_id) in enumerate(boosted):
        overlap = _token_overlap_score(test_id, methods_changed, files_changed)
        boosted[i] = (score + overlap * overlap_w, test_id)

    # --- 4: Package proximity ---
    prox_w = config.retrieval.package_proximity_weight
    for i, (score, test_id) in enumerate(boosted):
        prox = _package_proximity_score(test_id, files_changed)
        boosted[i] = (boosted[i][0] + prox * prox_w, test_id)

    # --- 5: BM25 blend ---
    bm25_query: list[str] = []
    for m in methods_changed:
        bm25_query.extend(m.lower().split("."))
    for f in files_changed:
        bm25_query.extend(re.split(r'[./\\]+', Path(f).stem.lower()))

    bm25_w = config.retrieval.bm25_weight
    bm25_map = kb.bm25_scores(bm25_query)
    if bm25_map:
        for i, (score, test_id) in enumerate(boosted):
            bm25 = bm25_map.get(test_id, 0.0)
            boosted[i] = ((1.0 - bm25_w) * score + bm25_w * bm25, test_id)

    # --- 6: Negative pass — penalise tier 4/5 with no structural signal ---
    if config.retrieval.negative_pass_enabled:
        penalty = config.retrieval.negative_pass_penalty
        for i, (score, test_id) in enumerate(boosted):
            tier = kb.tier_for_id(test_id)
            if tier not in (4, 5):
                continue
            tok = _token_overlap_score(test_id, methods_changed, files_changed)
            bm25_val = bm25_map.get(test_id, 0.0)
            pkg = _package_proximity_score(test_id, files_changed)
            if tok == 0.0 and bm25_val < 0.05 and pkg < 0.10:
                boosted[i] = (score * penalty, test_id)

    boosted.sort(key=lambda x: (-x[0], x[1]))   # secondary sort by test_id for determinism

    # --- 7: Adaptive threshold ---
    effective_threshold = config.retrieval.similarity_threshold
    low_confidence = False
    if config.retrieval.adaptive_threshold_enabled and boosted:
        top_score = boosted[0][0]
        if top_score < 0.45:
            effective_threshold = max(
                config.retrieval.adaptive_threshold_min,
                top_score - 0.10,
            )
            logger.warning(
                "Low-confidence retrieval (best=%.3f). Widening threshold %.2f → %.2f",
                top_score, config.retrieval.similarity_threshold, effective_threshold,
            )
            low_confidence = True

    boosted = boosted[: config.retrieval.top_k]

    candidates: list[Candidate] = [
        Candidate(test_id=test_id, score=score, tier=kb.tier_for_id(test_id), rank=rank)
        for rank, (score, test_id) in enumerate(boosted)
    ]

    # --- 8: Optional cross-encoder filter ---
    if config.selector.cross_encoder_enabled and client is not None and kb_summaries is not None:
        candidates = _cross_encode_filter(
            candidates,
            intent.intent_summary,
            kb_summaries,
            client,
            config.selector.cross_encoder_top_n,
        )

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
        low_confidence_retrieval=low_confidence,
        effective_threshold=effective_threshold,
    )

    return candidates, trace
