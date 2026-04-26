"""Safety Bridge helpers (thin wrappers used by ranker.py for clarity)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from semantic_rts.config import AblationFlags, Config
    from semantic_rts.impact.retriever import Candidate


def tier1_tests(all_kb_tests: list[tuple[str, int]], ablation_flags: "AblationFlags") -> list[str]:
    """Return test_ids of all Tier-1 tests (or [] if bridge disabled)."""
    if not ablation_flags.safety_bridge_enabled:
        return []
    return [tid for tid, tier in all_kb_tests if tier == 1]


def tier2_candidates(candidates: list["Candidate"], config: "Config", ablation_flags: "AblationFlags") -> list["Candidate"]:
    """Return Tier-2 candidates above the tier-2 threshold."""
    if not ablation_flags.safety_bridge_enabled:
        return []
    thresh = config.selector.safety_bridge.tier_2_threshold
    return [c for c in candidates if c.tier == 2 and c.score >= thresh]
