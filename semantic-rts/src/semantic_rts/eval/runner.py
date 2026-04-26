"""Bug-by-bug evaluation loop (M6)."""

from __future__ import annotations

import csv
import dataclasses
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from semantic_rts.eval.defects4j import load_bug
from semantic_rts.eval.metrics import BugMetrics, compute_metrics
from semantic_rts.impact.diff_parser import extract_changed_methods, parse_unified_diff
from semantic_rts.impact.intent_agent import analyze_intent
from semantic_rts.impact.retriever import retrieve
from semantic_rts.kb.vector_store import VectorStore
from semantic_rts.llm.client import GeminiClient
from semantic_rts.llm.embeddings import GeminiEmbedder
from semantic_rts.selector.ranker import select

if TYPE_CHECKING:
    from semantic_rts.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def write_csv(results: list["BugMetrics"], path: Path) -> None:
    """Write a list of BugMetrics to a CSV file."""
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [f.name for f in dataclasses.fields(results[0])]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow(dataclasses.asdict(row))
    logger.info("Results written to %s (%d rows)", path, len(results))


def read_csv(path: Path) -> list[dict]:
    """Read a results CSV back into a list of dicts (for reporting)."""
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Single-bug runner
# ---------------------------------------------------------------------------

def run_bug(
    project: str,
    bug_id: int,
    config: "Config",
    store: "VectorStore",
    client: "GeminiClient",
    embedder: "GeminiEmbedder",
    work_dir: Path,
    d4j_home: Path | None = None,
    method: str = "semantic",
) -> "BugMetrics":
    """Run the full pipeline on one bug and return BugMetrics.

    Checkouts are cached on disk; re-runs are fast after the first call.
    """
    t0 = time.perf_counter()

    # --- Step 1: Checkout + diff + test lists ---
    bug = load_bug(project, bug_id, work_dir, d4j_home)

    if not bug.diff:
        logger.warning("%s-%d: empty diff — returning zero-recall metrics", project, bug_id)
        return compute_metrics(
            selected=[],
            failing=bug.failing_tests,
            all_tests=bug.all_tests,
            method=method,
            project=project,
            bug_id=bug_id,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    # --- Step 2: Parse diff ---
    parsed = parse_unified_diff(bug.diff)
    methods = extract_changed_methods(parsed, project_root=str(bug.fixed_dir))
    logger.info(
        "%s-%d: %d file(s) changed, %d method(s) changed",
        project, bug_id, len(parsed.files_changed), len(methods),
    )

    # --- Step 3: Analyze intent ---
    intent = analyze_intent(bug.diff, parsed.files_changed, methods, client, config)
    logger.info("%s-%d: intent=%s", project, bug_id, intent.intent_summary[:80])

    # --- Step 4: Retrieve ---
    candidates, _trace = retrieve(
        intent, store, embedder, config,
        diff_hash=parsed.diff_hash,
        files_changed=parsed.files_changed,
        methods_changed=methods,
    )
    logger.info("%s-%d: %d candidates retrieved", project, bug_id, len(candidates))

    # --- Step 5: Select ---
    selection = select(candidates, store.all_tests(), config)
    selected_ids = [t.test_id for t in selection.selected]
    logger.info("%s-%d: %d tests selected", project, bug_id, len(selected_ids))

    latency_ms = (time.perf_counter() - t0) * 1000

    return compute_metrics(
        selected=selected_ids,
        failing=bug.failing_tests,
        all_tests=bug.all_tests,
        method=method,
        project=project,
        bug_id=bug_id,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Multi-bug eval loop
# ---------------------------------------------------------------------------

def run_eval(
    project: str,
    bug_ids: list[int],
    config: "Config",
    kb_path: Path,
    work_dir: Path,
    d4j_home: Path | None = None,
    output_dir: Path | None = None,
    method: str = "semantic",
) -> list["BugMetrics"]:
    """Run the eval loop over *bug_ids* and optionally write a CSV.

    Loads the KB once, then processes each bug in sequence.
    Failed bugs are logged and skipped so the loop always completes.

    Args:
        project:    Defects4J project name, e.g. "Chart".
        bug_ids:    List of bug IDs to evaluate.
        config:     Loaded Config object.
        kb_path:    Path to the pre-built KB directory.
        work_dir:   Root for bug checkouts.
        d4j_home:   Defects4J installation path (optional).
        output_dir: If given, writes <project>_results.csv here.
        method:     Label recorded in the CSV ("semantic" by default).
    """
    logger.info("Loading KB from %s (%s bugs to evaluate)", kb_path, len(bug_ids))
    store = VectorStore.load(kb_path)
    logger.info("KB loaded: %d tests", store.size)

    client = GeminiClient(config)
    embedder = GeminiEmbedder(config)

    results: list[BugMetrics] = []
    for bug_id in bug_ids:
        logger.info("--- %s-%d ---", project, bug_id)
        try:
            m = run_bug(
                project, bug_id, config, store, client, embedder,
                work_dir, d4j_home, method=method,
            )
            results.append(m)
            logger.info(
                "%s-%d: recall=%.3f  sel_rate=%.3f  latency=%.0fms",
                project, bug_id, m.recall, m.selection_rate, m.latency_ms,
            )
        except Exception as exc:
            logger.error("%s-%d: FAILED — %s", project, bug_id, exc, exc_info=True)

    if output_dir is not None and results:
        csv_path = Path(output_dir) / f"{project}_{method}_results.csv"
        write_csv(results, csv_path)

    return results
