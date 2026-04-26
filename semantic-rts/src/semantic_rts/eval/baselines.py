"""Baseline RTS eval orchestration (M7): Retest-All and file-level static."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Literal

from semantic_rts.baselines.retest_all import select_all
from semantic_rts.baselines.starts_runner import select_static_fallback
from semantic_rts.eval.defects4j import load_bug
from semantic_rts.eval.metrics import BugMetrics, compute_metrics
from semantic_rts.eval.runner import write_csv
from semantic_rts.impact.diff_parser import parse_unified_diff

logger = logging.getLogger(__name__)

BaselineMethod = Literal["retest_all", "file_level_static"]


# ---------------------------------------------------------------------------
# Thin wrappers — keep the eval layer consistent with run_bug's interface
# ---------------------------------------------------------------------------

def retest_all(all_tests: list[str]) -> list[str]:
    """Select every test — delegates to baselines.retest_all.select_all."""
    return select_all(all_tests)


def file_level_static(diff: str, all_tests: list[str]) -> list[str]:
    """Name-matching static baseline — delegates to starts_runner.select_static_fallback.

    Parses *diff* to extract changed Java file paths, then selects tests
    whose simple class name contains a changed file's stem.
    """
    parsed = parse_unified_diff(diff)
    return select_static_fallback(parsed.files_changed, all_tests)


# ---------------------------------------------------------------------------
# Per-bug runner
# ---------------------------------------------------------------------------

def run_baseline_bug(
    project: str,
    bug_id: int,
    method: BaselineMethod,
    work_dir: Path,
    d4j_home: Path | None = None,
) -> BugMetrics:
    """Run one baseline method on a single bug and return BugMetrics."""
    t0 = time.perf_counter()
    bug = load_bug(project, bug_id, work_dir, d4j_home)

    if method == "retest_all":
        selected = retest_all(bug.all_tests)
    elif method == "file_level_static":
        selected = file_level_static(bug.diff, bug.all_tests)
    else:
        raise ValueError(f"Unknown baseline method: {method!r}")

    return compute_metrics(
        selected=selected,
        failing=bug.failing_tests,
        all_tests=bug.all_tests,
        method=method,
        project=project,
        bug_id=bug_id,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )


# ---------------------------------------------------------------------------
# Multi-bug eval loop
# ---------------------------------------------------------------------------

def run_baseline_eval(
    project: str,
    bug_ids: list[int],
    method: BaselineMethod,
    work_dir: Path,
    d4j_home: Path | None = None,
    output_dir: Path | None = None,
) -> list[BugMetrics]:
    """Run a baseline method over all bug_ids, log results, and write a CSV.

    Failed bugs are logged and skipped so the loop always completes.
    """
    results: list[BugMetrics] = []

    for bug_id in bug_ids:
        logger.info("--- %s-%d [%s] ---", project, bug_id, method)
        try:
            m = run_baseline_bug(project, bug_id, method, work_dir, d4j_home)
            results.append(m)
            logger.info(
                "%s-%d [%s]: recall=%.3f  sel_rate=%.3f  latency=%.0fms",
                project, bug_id, method, m.recall, m.selection_rate, m.latency_ms,
            )
        except Exception as exc:
            logger.error("%s-%d [%s]: FAILED — %s", project, bug_id, method, exc, exc_info=True)

    if output_dir is not None and results:
        csv_path = Path(output_dir) / f"{project}_{method}_results.csv"
        write_csv(results, csv_path)

    return results
