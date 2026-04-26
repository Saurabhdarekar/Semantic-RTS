"""Phase 3 — Risk-Aware Selector.

Pure Python, no LLM, deterministic. The full algorithm is implemented here
(not just a stub) because it can be unit-tested without any external dependencies.

Selection order (all in scored dict):
  1. Safety Bridge Tier 1 — always include if ablation_flags.safety_bridge_enabled
  2. Safety Bridge Tier 2 — include if score >= tier_2_threshold
  3. Semantic match       — include if score >= similarity_threshold
  4. Precision Filter     — drop Tier 4/5 below per-tier min score
  5. Cap                  — keep top max_selected by score; never drop safety_bridge_t1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from semantic_rts.config import AblationFlags, Config
    from semantic_rts.impact.retriever import Candidate


Reason = Literal["safety_bridge_t1", "safety_bridge_t2", "semantic_match"]
DroppedReason = Literal["dropped_precision_t5", "dropped_precision_t4", "dropped_cap"]


@dataclass
class SelectedTest:
    test_id: str
    score: float
    tier: int
    reason: Reason


@dataclass
class SelectionTrace:
    selected: list[SelectedTest]
    dropped: list[tuple[str, DroppedReason]]


def _effective_tier(tier: int, ablation_flags: "AblationFlags") -> int:
    return 3 if not ablation_flags.tiers_enabled else tier


def _should_drop(test_id: str, tier: int, score: float, ablation_flags: "AblationFlags", config: "Config") -> bool:
    t = _effective_tier(tier, ablation_flags)
    pf = config.selector.precision_filter
    if not ablation_flags.precision_filter_enabled:
        return False
    if t == 5 and score < pf.tier_5_min:
        return True
    if t == 4 and score < pf.tier_4_min:
        return True
    return False


def select(
    candidates: list["Candidate"],
    all_kb_tests: list[tuple[str, int]],   # [(test_id, tier), ...]
    config: "Config",
    ablation_flags: "AblationFlags | None" = None,
) -> SelectionTrace:
    """Apply Safety Bridge, semantic matching, Precision Filter, and cap.

    Args:
        candidates:    Top-K from Phase 2, each with test_id / score / tier / rank.
        all_kb_tests:  Every (test_id, tier) pair in the KB (for Tier-1 safety bridge).
        config:        Config object.
        ablation_flags: Override flags for ablation runs.
    """
    from semantic_rts.config import AblationFlags
    if ablation_flags is None:
        ablation_flags = config.ablation

    # scored: test_id → (score, reason, tier)
    scored: dict[str, tuple[float, Reason, int]] = {}

    # --- Safety Bridge ---
    if ablation_flags.safety_bridge_enabled:
        # Tier 1: always include regardless of score
        for test_id, tier in all_kb_tests:
            if _effective_tier(tier, ablation_flags) == 1:
                scored[test_id] = (1.0, "safety_bridge_t1", tier)

        # Tier 2: include if score above threshold
        t2_thresh = config.selector.safety_bridge.tier_2_threshold
        for c in candidates:
            if _effective_tier(c.tier, ablation_flags) == 2 and c.score >= t2_thresh:
                if c.test_id not in scored:
                    scored[c.test_id] = (c.score, "safety_bridge_t2", c.tier)

    # --- Semantic relevance ---
    sim_thresh = config.retrieval.similarity_threshold
    for c in candidates:
        if c.score >= sim_thresh and c.test_id not in scored:
            scored[c.test_id] = (c.score, "semantic_match", c.tier)

    # --- Precision Filter ---
    dropped: list[tuple[str, DroppedReason]] = []
    to_remove: list[str] = []
    for test_id, (score, reason, tier) in scored.items():
        if reason == "safety_bridge_t1":
            continue  # never drop Tier-1 safety bridge entries
        if _should_drop(test_id, tier, score, ablation_flags, config):
            drop_reason: DroppedReason = (
                "dropped_precision_t5" if _effective_tier(tier, ablation_flags) == 5
                else "dropped_precision_t4"
            )
            dropped.append((test_id, drop_reason))
            to_remove.append(test_id)
    for tid in to_remove:
        del scored[tid]

    # --- Cap: keep top max_selected, but never drop safety_bridge_t1 ---
    max_sel = config.selector.max_selected
    sorted_items = sorted(
        scored.items(),
        key=lambda kv: (-kv[1][0], kv[0]),   # desc score, then asc test_id tiebreaker
    )

    guaranteed = [(tid, v) for tid, v in sorted_items if v[1] == "safety_bridge_t1"]
    non_guaranteed = [(tid, v) for tid, v in sorted_items if v[1] != "safety_bridge_t1"]

    # Fill non-guaranteed slots up to max_selected - len(guaranteed)
    remaining_slots = max(0, max_sel - len(guaranteed))
    kept_non_guaranteed = non_guaranteed[:remaining_slots]
    cap_dropped = non_guaranteed[remaining_slots:]

    for tid, _ in cap_dropped:
        dropped.append((tid, "dropped_cap"))

    final_items = guaranteed + kept_non_guaranteed
    selected = [
        SelectedTest(test_id=tid, score=v[0], tier=v[2], reason=v[1])
        for tid, v in final_items
    ]

    return SelectionTrace(selected=selected, dropped=dropped)
