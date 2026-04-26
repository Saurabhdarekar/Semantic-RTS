"""Wrap `defects4j checkout` to prepare buggy/fixed project trees.

Implemented in M5.
"""

from __future__ import annotations

from pathlib import Path


def checkout(
    project: str,
    bug_id: int,
    version: str,   # "b" (buggy) or "f" (fixed)
    work_dir: Path,
    d4j_home: str,
) -> Path:
    """Run `defects4j checkout` and return the path to the checked-out tree."""
    raise NotImplementedError("checkout implemented in M5")
