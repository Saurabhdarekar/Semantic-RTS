"""Precision Filter helpers (thin wrappers used by ranker.py for clarity)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from semantic_rts.config import AblationFlags, Config


def should_drop(tier: int, score: float, config: "Config", ablation_flags: "AblationFlags") -> bool:
    """Return True if this test should be dropped by the Precision Filter."""
    if not ablation_flags.precision_filter_enabled:
        return False
    t = tier if ablation_flags.tiers_enabled else 3
    pf = config.selector.precision_filter
    if t == 5 and score < pf.tier_5_min:
        return True
    if t == 4 and score < pf.tier_4_min:
        return True
    return False
