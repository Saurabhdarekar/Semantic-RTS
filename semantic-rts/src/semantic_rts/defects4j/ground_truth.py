"""Load fault-triggering tests for a Defects4J bug (ground truth for eval).

Implemented in M5.
"""

from __future__ import annotations

from pathlib import Path


def load_failing_tests(
    project: str,
    bug_id: int,
    buggy_path: Path,
    d4j_home: str,
) -> list[str]:
    """Return list of fault-triggering test IDs for this bug."""
    raise NotImplementedError("load_failing_tests implemented in M5")
