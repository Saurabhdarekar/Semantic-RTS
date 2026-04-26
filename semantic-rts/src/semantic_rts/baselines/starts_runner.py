"""STARTS baseline wrapper (or file-level static fallback).

STARTS is a Maven plugin that requires bytecode analysis; most Defects4J
projects use Ant and cannot run STARTS directly.  We implement a
name-based static fallback instead: if the diff touches Bar.java we select
any test whose simple class name contains "bar" (case-insensitive).  This
approximates file-level static coverage without bytecode or coverage data.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def select_static_fallback(diff_files: list[str], all_test_ids: list[str]) -> list[str]:
    """Name-matching static baseline.

    For each changed Java file (e.g. ``Auth.java``), selects every test
    whose simple class name contains that file's stem (case-insensitive).
    Falls back to selecting all tests when *diff_files* is empty.

    Args:
        diff_files:    List of changed file paths (from the diff).
        all_test_ids:  All test IDs in format ``pkg.Class::method``.
    """
    java_files = [f for f in diff_files if f.endswith(".java")]
    if not java_files:
        logger.debug("select_static_fallback: no Java files changed — returning all tests")
        return list(all_test_ids)

    # Lowercase stems, e.g. {"auth", "database"}
    changed_stems = {Path(f).stem.lower() for f in java_files}
    logger.debug("select_static_fallback: changed stems = %s", changed_stems)

    selected: list[str] = []
    for tid in all_test_ids:
        class_fqn = tid.split("::")[0] if "::" in tid else tid
        class_simple = class_fqn.split(".")[-1].lower()
        if any(stem in class_simple for stem in changed_stems):
            selected.append(tid)

    logger.debug("select_static_fallback: %d/%d selected", len(selected), len(all_test_ids))
    return selected


def select_starts(
    diff_files: list[str],
    all_test_ids: list[str],
    project_path: Path | None = None,
    d4j_home: str = "",
) -> list[str]:
    """Run STARTS, or fall back to name-based static selection.

    The actual STARTS Maven plugin cannot run on Ant-based Defects4J
    projects, so this always uses ``select_static_fallback``.  The
    *project_path* and *d4j_home* parameters are retained for future
    compatibility with Maven-based projects.
    """
    return select_static_fallback(diff_files, all_test_ids)
