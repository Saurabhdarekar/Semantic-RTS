"""Per-bug metric computation: recall, selection rate, precision, latency."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BugMetrics:
    project: str
    bug_id: int
    method: str                  # "ours" | "retest_all" | "starts"
    recall: float                # |selected ∩ failing| / |failing|
    selection_rate: float        # |selected| / |all_tests|
    precision: float             # |selected ∩ failing| / |selected|
    latency_ms: float
    cost_usd: float
    n_failing: int
    n_selected: int
    n_total: int


def compute_metrics(
    selected: list[str],
    failing: list[str],
    all_tests: list[str],
    method: str,
    project: str,
    bug_id: int,
    latency_ms: float = 0.0,
    cost_usd: float = 0.0,
) -> BugMetrics:
    """Compute recall, selection_rate, and precision for one bug."""
    selected_set = set(selected)
    failing_set = set(failing)
    all_set = set(all_tests)

    tp = len(selected_set & failing_set)
    recall = tp / len(failing_set) if failing_set else 1.0
    selection_rate = len(selected_set) / len(all_set) if all_set else 0.0
    precision = tp / len(selected_set) if selected_set else 0.0

    return BugMetrics(
        project=project,
        bug_id=bug_id,
        method=method,
        recall=recall,
        selection_rate=selection_rate,
        precision=precision,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        n_failing=len(failing_set),
        n_selected=len(selected_set),
        n_total=len(all_set),
    )
