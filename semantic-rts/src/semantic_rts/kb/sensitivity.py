"""Static test sensitivity scorer — no LLM, no I/O."""

from __future__ import annotations

import re

_MOCK_RE = re.compile(r'\b(mock|when|verify|stub|Mockito)\b', re.IGNORECASE)
_BOUNDARY_RE = re.compile(
    r'\b(0|1|-1|null|""|empty|boundary|limit|Integer\.MAX_VALUE|Integer\.MIN_VALUE)\b'
)
_EXCEPTION_RE = re.compile(r'assertThrows|expected\s*=|ExpectedException')


def compute_sensitivity(source: str) -> float:
    """Estimate how sensitive a test is to logic changes using static heuristics.

    Returns a float in [0.1, 1.0]. Higher = more likely to catch real regressions.
    """
    score = 0.0

    # More distinct assertion types = more sensitive
    distinct_asserts = len(set(re.findall(r'\bassert\w+', source)))
    score += min(distinct_asserts * 0.15, 0.45)

    # Mocks reduce sensitivity — test is isolated from real production behavior
    if _MOCK_RE.search(source):
        score -= 0.15

    # Boundary value testing increases sensitivity to off-by-one errors
    if _BOUNDARY_RE.search(source):
        score += 0.15

    # Exception assertion increases sensitivity to behavioral changes
    if _EXCEPTION_RE.search(source):
        score += 0.15

    # Longer tests tend to cover more behavior paths
    lines = sum(1 for line in source.splitlines() if line.strip())
    score += min(lines / 100.0, 0.15)

    return max(0.1, min(1.0, score))
