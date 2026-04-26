"""Phase 1 orchestrator: discover → parse → summarize → tier → embed → index."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from semantic_rts.config import Config
from semantic_rts.kb.test_parser import TestMethod, discover_test_files, parse_test_methods
from semantic_rts.kb.summarizer import summarize_test
from semantic_rts.kb.tier_classifier import classify_tier
from semantic_rts.kb.vector_store import VectorStore
from semantic_rts.llm.client import GeminiClient
from semantic_rts.llm.embeddings import GeminiEmbedder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding format (must match Phase 2 query format)
# ---------------------------------------------------------------------------

def format_for_embedding(tm: TestMethod) -> str:
    concepts = " ".join(tm.concepts)
    return f"summary: {tm.summary} | tier: {tm.tier} | class: {tm.class_simple} | concepts: {concepts}"


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _load_existing(jsonl_path: Path) -> dict[str, dict]:
    """Return {test_id: row} for all rows already in tests.jsonl."""
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
# Per-test enrichment
# ---------------------------------------------------------------------------

def _enrich(
    tm: TestMethod,
    client: GeminiClient,
    embedder: GeminiEmbedder,
    config: Config,
    request_counter: list[int],
    max_requests: int | None,
) -> TestMethod:
    """Summarize, tier, and embed a single test method in place."""
    if max_requests is not None and request_counter[0] >= max_requests:
        raise RuntimeError(
            f"Reached --max-requests limit ({max_requests}). Stopping."
        )

    # Step 1: Summarize
    summary, concepts = summarize_test(tm, client)
    tm.summary = summary
    tm.concepts = concepts
    request_counter[0] += 1

    # Step 2: Classify tier
    tm.tier, tm.tier_source = classify_tier(tm, config, client)

    # Step 3: Embed
    embed_text = format_for_embedding(tm)
    tm.embedding = embedder.embed(embed_text)
    request_counter[0] += 1

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
    """Build the knowledge base for a project and write it to config.paths.kb_dir.

    Args:
        project_path:  Path to the checked-out project (fixed version).
        project_name:  Short name, e.g. "Chart".
        config:        Loaded Config object.
        resume:        If True, skip tests already in tests.jsonl (same source_hash).
        max_requests:  Abort after this many LLM+embedding calls.
    """
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
        if ex and ex.get("source_hash") == tm.source_hash and ex.get("summary"):
            # Restore enriched fields from disk
            tm.summary = ex["summary"]
            tm.concepts = ex.get("concepts", [])
            tm.tier = ex.get("tier", 3)
            tm.tier_source = ex.get("tier_source", "rule")
            tm.embedding = ex.get("embedding", [])
            already_done.append(tm)
        else:
            to_process.append(tm)

    logger.info(
        "Resume: %d already done, %d to process.", len(already_done), len(to_process)
    )

    # --- LLM enrichment (concurrent) ---
    client = GeminiClient(config)
    embedder = GeminiEmbedder(config)
    request_counter = [0]  # mutable int for closure across threads
    max_workers = min(5, len(to_process)) if to_process else 1

    enriched: list[TestMethod] = list(already_done)

    if to_process:
        with open(jsonl_path, "a", encoding="utf-8") as out_f:
            # Write already-done entries if starting fresh (no existing file)
            if not resume or not jsonl_path.exists():
                out_f.seek(0)
                out_f.truncate()
                for tm in already_done:
                    out_f.write(json.dumps(_tm_to_dict(tm)) + "\n")

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(
                        _enrich, tm, client, embedder, config, request_counter, max_requests
                    ): tm
                    for tm in to_process
                }

                with tqdm(total=len(to_process), desc=f"Building KB [{project_name}]") as pbar:
                    for future in as_completed(futures):
                        tm = futures[future]
                        try:
                            enriched_tm = future.result()
                            enriched.append(enriched_tm)
                            out_f.write(json.dumps(_tm_to_dict(enriched_tm)) + "\n")
                            out_f.flush()
                        except Exception as exc:
                            logger.error("Failed to enrich %s: %s", tm.test_id, exc)
                        finally:
                            pbar.update(1)

    # If we resumed and only have already-done, rewrite cleanly
    if not to_process and already_done:
        with open(jsonl_path, "w", encoding="utf-8") as out_f:
            for tm in enriched:
                out_f.write(json.dumps(_tm_to_dict(tm)) + "\n")

    # --- Build FAISS index ---
    logger.info("Building FAISS index over %d tests ...", len(enriched))
    store = VectorStore(dim=config.vector_store.embedding_dim)

    valid = [tm for tm in enriched if tm.embedding]
    if not valid:
        logger.warning("No embeddings available — FAISS index will be empty.")
    else:
        store.add(
            vectors=[tm.embedding for tm in valid],
            test_ids=[tm.test_id for tm in valid],
            tiers=[tm.tier for tm in valid],
        )

    store.save(kb_dir)
    logger.info(
        "KB for %s saved to %s (%d tests indexed).", project_name, kb_dir, store.size
    )


def _tm_to_dict(tm: TestMethod) -> dict:
    d = asdict(tm)
    # Don't write the full embedding to JSONL — it's stored in the FAISS index
    d.pop("embedding", None)
    return d
