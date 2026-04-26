"""Generate buggy↔fixed diffs for Defects4J bugs.

Implemented in M5.
"""

from __future__ import annotations

from pathlib import Path


def extract_diff(
    project: str,
    bug_id: int,
    fixed_path: Path,
    buggy_path: Path,
) -> str:
    """Return unified diff text between the fixed and buggy source trees."""
    raise NotImplementedError("extract_diff implemented in M5")
