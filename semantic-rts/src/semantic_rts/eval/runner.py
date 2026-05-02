"""Bug-by-bug evaluation loop (M6)."""

from __future__ import annotations

import csv
import dataclasses
import logging
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm

from semantic_rts.eval.defects4j import DeprecatedBugError, load_bug
from semantic_rts.eval.metrics import BugMetrics, compute_metrics
from semantic_rts.impact.diff_parser import extract_changed_methods, parse_unified_diff
from semantic_rts.impact.intent_agent import analyze_intent
from semantic_rts.impact.retriever import retrieve
from semantic_rts.kb.builder import update_kb_from_test_paths
from semantic_rts.kb.vector_store import VectorStore
from semantic_rts.llm.client import GeminiClient
from semantic_rts.llm.embeddings import GeminiEmbedder
from semantic_rts.selector.ranker import select

if TYPE_CHECKING:
    from semantic_rts.config import Config

logger = logging.getLogger(__name__)

_bug_desc_cache: dict[tuple, str | None] = {}


def _get_bug_description(project: str, bug_id: int) -> str | None:
    """Fetch the Defects4J bug summary for use as commit message context."""
    key = (project, bug_id)
    if key in _bug_desc_cache:
        return _bug_desc_cache[key]
    try:
        result = subprocess.run(
            ["defects4j", "info", "-p", project, "-b", str(bug_id)],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            lines = result.stdout.splitlines()
            for i, line in enumerate(lines):
                if "summary" in line.lower() and i + 1 < len(lines):
                    desc = lines[i + 1].strip() or None
                    _bug_desc_cache[key] = desc
                    return desc
    except Exception:
        pass
    _bug_desc_cache[key] = None
    return None


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
    tested_methods_map: "dict[str, list[str]] | None" = None,
    kb_summaries: "dict[str, str] | None" = None,
    fixture_map: "dict[str, list[str]] | None" = None,
    sensitivity_map: "dict[str, float] | None" = None,
    topology_map: "dict[str, str] | None" = None,
    kb_path: "Path | None" = None,
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

    # --- Step 2b: Upsert test files from diff into KB ---
    test_in_diff_ids: list[str] = []
    if parsed.test_files_changed and kb_path is not None:
        try:
            test_in_diff_ids = update_kb_from_test_paths(
                parsed.test_files_changed,
                bug.fixed_dir,
                store,
                kb_path,
                client,
                embedder,
                config,
            )
            logger.info("%s-%d: upserted %d test(s) from diff", project, bug_id, len(test_in_diff_ids))
        except Exception as exc:
            logger.warning("%s-%d: KB upsert failed: %s", project, bug_id, exc)

    # --- Step 3: Analyze intent ---
    bug_desc = _get_bug_description(project, bug_id)
    intent = analyze_intent(
        bug.diff, parsed.files_changed, methods, client, config,
        project_root=str(bug.fixed_dir),
        change_type=parsed.change_type,
        commit_message=bug_desc,
    )
    logger.info("%s-%d: change_type=%s intent=%s", project, bug_id, parsed.change_type, intent.intent_summary[:80])

    # --- Step 4: Retrieve ---
    candidates, _trace = retrieve(
        intent, store, embedder, config,
        diff_hash=parsed.diff_hash,
        files_changed=parsed.files_changed,
        methods_changed=methods,
        tested_methods_map=tested_methods_map,
        client=client,
        kb_summaries=kb_summaries,
    )
    logger.info("%s-%d: %d candidates retrieved", project, bug_id, len(candidates))

    # --- Step 5: Select ---
    selection = select(
        candidates, store.all_tests(), config,
        fixture_map=fixture_map,
        sensitivity_map=sensitivity_map,
        topology_map=topology_map,
        change_type=parsed.change_type,
        files_changed=parsed.files_changed,
        similarity_threshold=_trace.effective_threshold if _trace.low_confidence_retrieval else None,
        test_in_diff_ids=test_in_diff_ids,
        methods_changed=methods,
        tested_methods_map=tested_methods_map,
    )
    selected_ids = [t.test_id for t in selection.selected]
    logger.info("%s-%d: %d tests selected", project, bug_id, len(selected_ids))

    latency_ms = (time.perf_counter() - t0) * 1000

    # Use KB test IDs as the universe (method-level), not defects4j's class-level list
    kb_all_tests = [t for t, _ in store.all_tests()]

    return compute_metrics(
        selected=selected_ids,
        failing=bug.failing_tests,
        all_tests=kb_all_tests,
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

    # Build all lookup maps in a single pass over tests.jsonl
    tested_methods_map: dict[str, list[str]] = {}
    fixture_map: dict[str, list[str]] = {}
    sensitivity_map: dict[str, float] = {}
    topology_map: dict[str, str] = {}
    kb_summaries: dict[str, str] = {}

    jsonl_path = kb_path / "tests.jsonl"
    if jsonl_path.exists():
        import json as _json
        with open(jsonl_path, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _row = _json.loads(_line)
                    _tid = _row["test_id"]
                    tested_methods_map[_tid] = _row.get("tested_methods", [])
                    fixture_map[_tid] = _row.get("fixture_classes", [])
                    sensitivity_map[_tid] = float(_row.get("sensitivity_score", 0.5))
                    topology_map[_tid] = _row.get("topology_scope", "unit")
                    kb_summaries[_tid] = _row.get("summary", "")
                except Exception:
                    pass

    store.build_bm25(tested_methods_map)
    logger.info("BM25 index built over %d tests", store.size)

    client = GeminiClient(config)
    embedder = GeminiEmbedder(config)

    results: list[BugMetrics] = []
    with tqdm(bug_ids, desc=f"eval [{project}]", unit="bug") as pbar:
        for bug_id in pbar:
            pbar.set_postfix_str(f"bug={bug_id}")
            logger.info("--- %s-%d ---", project, bug_id)
            try:
                m = run_bug(
                    project, bug_id, config, store, client, embedder,
                    work_dir, d4j_home, method=method,
                    tested_methods_map=tested_methods_map,
                    kb_summaries=kb_summaries,
                    fixture_map=fixture_map,
                    sensitivity_map=sensitivity_map,
                    topology_map=topology_map,
                    kb_path=kb_path,
                )
                results.append(m)
                pbar.set_postfix_str(
                    f"bug={bug_id} recall={m.recall:.2f} sel={m.selection_rate:.2f}"
                )
                logger.info(
                    "%s-%d: recall=%.3f  sel_rate=%.3f  latency=%.0fms",
                    project, bug_id, m.recall, m.selection_rate, m.latency_ms,
                )
            except DeprecatedBugError:
                pbar.set_postfix_str(f"bug={bug_id} skipped (deprecated)")
                logger.warning("%s-%d: skipped (deprecated bug)", project, bug_id)
            except Exception as exc:
                pbar.set_postfix_str(f"bug={bug_id} FAILED")
                logger.error("%s-%d: FAILED — %s", project, bug_id, exc, exc_info=True)

    if output_dir is not None and results:
        csv_path = Path(output_dir) / f"{project}_{method}_results.csv"
        write_csv(results, csv_path)

    return results
