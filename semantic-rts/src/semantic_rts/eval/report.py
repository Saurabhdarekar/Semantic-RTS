"""Generate CSV + Markdown summary from eval results.

Implemented in M8.
"""

from __future__ import annotations

from pathlib import Path


def generate_report(results_csv: Path, output_dir: Path) -> None:
    """Read results.csv and write summary.md with tables and plots."""
    raise NotImplementedError("generate_report implemented in M8")
