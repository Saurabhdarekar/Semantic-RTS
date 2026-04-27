"""Phase 1 orchestrator: discover → parse → summarize+tier → batch-embed → index.

Two-phase enrichment:
  Phase A (concurrent LLM): summarize + classify tier + SUT linking per test
  Phase B (batch embed):    one API call per 20 tests instead of one per test
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from semantic_rts.config import Config
from semantic_rts.kb.test_parser import TestMethod, discover_test_files, parse_test_methods
from semantic_rts.kb.summarizer import summarize_test
from semantic_rts.kb.tier_classifier import classify_tier_rule
from semantic_rts.kb.vector_store import VectorStore
from semantic_rts.llm.client import GeminiClient
from semantic_rts.llm.embeddings import GeminiEmbedder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding format (must mirror Phase 2 query format)
# ---------------------------------------------------------------------------

def format_for_embedding(tm: TestMethod) -> str:
    parts = [tm.summary]
    if tm.condition:
        parts.append(f"Condition: {tm.condition}.")
    if tm.tested_methods:
        method_names = ", ".join(m.split(".")[-1] for m in tm.tested_methods)
        parts.append(f"Methods under test: {method_names}.")
    if tm.concepts:
        parts.append(f"Concepts: {', '.join(tm.concepts)}.")
    parts.append(f"Test class: {tm.class_simple}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _load_existing(jsonl_path: Path) -> dict[str, dict]:
    if not jsonl_path.exists():
        return {}
    existing: dict[str, dict] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                existing[row["test_id"]] = row
            except (json.JSONDecodeError, KeyError):
                pass
    return existing


# ---------------------------------------------------------------------------
# Phase A: per-test LLM enrichment (no embedding)
# ---------------------------------------------------------------------------

def _enrich_llm(
    tm: TestMethod,
    client: GeminiClient,
    config: Config,
    project_path: Path,
    request_counter: list[int],
    max_requests: int | None,
) -> TestMethod:
    """Summarize + classify tier for one test. Mutates and returns tm."""
    if max_requests is not None and request_counter[0] >= max_requests:
        raise RuntimeError(f"Reached --max-requests limit ({max_requests}). Stopping.")

    summary, concepts, llm_tier, tested_methods, condition = summarize_test(
        tm, client, project_path
    )
    tm.summary = summary
    tm.concepts = concepts
    tm.tested_methods = tested_methods
    tm.condition = condition
    request_counter[0] += 1

    # Rule-based tier takes precedence over LLM tier (safety: rules catch critical tests)
    rule_result = classify_tier_rule(tm, config.kb.tier_keywords)
    if rule_result is not None:
        tm.tier, tm.tier_source = rule_result
    else:
        tm.tier = llm_tier
        tm.tier_source = "llm"

    return tm


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_kb(
    project_path: str | Path,
    project_name: str,
    config: Config,
    *,
    resume: bool = True,
    max_requests: int | None = None,
) -> None:
    """Build the knowledge base for a project and write it to config.paths.kb_dir."""
    project_path = Path(project_path)
    kb_dir = config.paths.kb_path(project_name)
    kb_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = kb_dir / "tests.jsonl"

    # --- Discover and parse ---
    logger.info("Discovering test files in %s ...", project_path)
    files = discover_test_files(project_path)
    logger.info("Found %d test files.", len(files))

    all_methods = parse_test_methods(files, project_path)
    logger.info("Parsed %d test methods.", len(all_methods))

    if not all_methods:
        logger.warning("No test methods found in %s. KB will be empty.", project_path)
        return

    # --- Resume: skip already-processed tests ---
    existing = _load_existing(jsonl_path) if resume else {}
    to_process: list[TestMethod] = []
    already_done: list[TestMethod] = []

    for tm in all_methods:
        ex = existing.get(tm.test_id)
        if (ex and ex.get("source_hash") == tm.source_hash
                and ex.get("summary") and "tested_methods" in ex):
            tm.summary = ex["summary"]
            tm.condition = ex.get("condition", "")
            tm.tested_methods = ex.get("tested_methods", [])
            tm.concepts = ex.get("concepts", [])
            tm.tier = ex.get("tier", 3)
            tm.tier_source = ex.get("tier_source", "rule")
            already_done.append(tm)
        else:
            to_process.append(tm)

    logger.info("Resume: %d already done, %d to process.", len(already_done), len(to_process))

    # --- Phase A: concurrent LLM enrichment ---
    client = GeminiClient(config)
    request_counter = [0]
    max_workers = min(5, len(to_process)) if to_process else 1
    newly_enriched: list[TestMethod] = []

    if to_process:
        # Rewrite file cleanly if starting fresh
        mode = "a" if resume and jsonl_path.exists() else "w"
        with open(jsonl_path, mode, encoding="utf-8") as out_f:
            if mode == "w":
                for tm in already_done:
                    out_f.write(json.dumps(_tm_to_dict(tm)) + "\n")

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(
                        _enrich_llm, tm, client, config,
                        project_path, request_counter, max_requests,
                    ): tm
                    for tm in to_process
                }
                with tqdm(total=len(to_process), desc=f"LLM [{project_name}]") as pbar:
                    for future in as_completed(futures):
                        tm = futures[future]
                        try:
                            enriched_tm = future.result()
                            newly_enriched.append(enriched_tm)
                            out_f.write(json.dumps(_tm_to_dict(enriched_tm)) + "\n")
                            out_f.flush()
                        except Exception as exc:
                            logger.error("Failed to enrich %s: %s", tm.test_id, exc)
                        finally:
                            pbar.update(1)

    # If fully resumed with no new work, rewrite cleanly
    if not to_process and already_done:
        with open(jsonl_path, "w", encoding="utf-8") as out_f:
            for tm in already_done:
                out_f.write(json.dumps(_tm_to_dict(tm)) + "\n")

    all_enriched = already_done + newly_enriched

    # --- Phase B: batch embed ---
    embedder = GeminiEmbedder(config)
    valid = [tm for tm in all_enriched if tm.summary]

    if not valid:
        logger.warning("No summaries available — FAISS index will be empty.")
    else:
        logger.info("Batch embedding %d tests ...", len(valid))
        texts = [format_for_embedding(tm) for tm in valid]
        with tqdm(total=len(texts), desc=f"Embed [{project_name}]") as pbar:
            # Process in batches so tqdm updates per batch
            batch_size = 20
            for i in range(0, len(valid), batch_size):
                batch_tms = valid[i : i + batch_size]
                batch_texts = texts[i : i + batch_size]
                vecs = embedder.embed_batch_efficient(batch_texts, batch_size=batch_size)
                for tm, vec in zip(batch_tms, vecs):
                    tm.embedding = vec
                pbar.update(len(batch_tms))

    # --- Build FAISS index ---
    logger.info("Building FAISS index over %d tests ...", len(valid))
    store = VectorStore(dim=config.vector_store.embedding_dim)
    indexed = [tm for tm in valid if tm.embedding]
    if indexed:
        store.add(
            vectors=[tm.embedding for tm in indexed],
            test_ids=[tm.test_id for tm in indexed],
            tiers=[tm.tier for tm in indexed],
        )

    store.save(kb_dir)
    logger.info(
        "KB for %s saved to %s (%d tests indexed).", project_name, kb_dir, store.size
    )


def _tm_to_dict(tm: TestMethod) -> dict:
    d = asdict(tm)
    d.pop("embedding", None)  # stored in FAISS, not JSONL
    return d
