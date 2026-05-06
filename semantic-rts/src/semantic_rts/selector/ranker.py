"""Phase 3 — Risk-Aware Selector.

Pure Python, no LLM, deterministic. The full algorithm is implemented here
(not just a stub) because it can be unit-tested without any external dependencies.

Selection order (all in scored dict):
  1. Safety Bridge Tier 1   — always include if ablation_flags.safety_bridge_enabled
  2. Safety Bridge Tier 2   — include if score >= tier_2_threshold
  3. Fixture bypass         — include if a changed class is used as a fixture
  4. Semantic match         — include if score >= similarity_threshold
  5. Topology multiplier    — boost integration/system tests for structural changes
  6. Sensitivity multiplier — adjust scores by static regression-catch likelihood
  7. Precision Filter       — drop Tier 4/5 below per-tier min score
  8. Cap                    — keep top max_selected; never drop safety_bridge_t1
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from semantic_rts.config import AblationFlags, Config
    from semantic_rts.impact.retriever import Candidate


Reason = Literal[
    "safety_bridge_t1", "safety_bridge_t2", "semantic_match",
    "fixture_bypass", "naming_convention", "test_in_diff",
    "method_coverage", "test_method_match",
]
DroppedReason = Literal["dropped_precision_t5", "dropped_precision_t4", "dropped_cap"]

_PROTECTED_REASONS: frozenset[str] = frozenset({
    "safety_bridge_t1", "fixture_bypass", "naming_convention", "test_in_diff",
    # method_coverage and test_method_match are high-scoring but not cap-exempt:
    # they can be noisy for core classes that many tests exercise
})

# Method names too generic to use as a match signal
_SKIP_METHOD_NAMES: frozenset[str] = frozenset({
    "tostring", "equals", "hashcode", "getclass", "notify", "wait",
    "clone", "get", "set", "add", "remove", "run", "init", "close",
})


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


def _should_drop(
    tier: int,
    score: float,
    reason: str,
    ablation_flags: "AblationFlags",
    config: "Config",
) -> bool:
    if reason in _PROTECTED_REASONS:
        return False
    if not ablation_flags.precision_filter_enabled:
        return False
    t = _effective_tier(tier, ablation_flags)
    pf = config.selector.precision_filter
    if t == 5 and score < pf.tier_5_min:
        return True
    if t == 4 and score < pf.tier_4_min:
        return True
    return False


def _naming_convention_matches(
    all_kb_tests: list[tuple[str, int]],
    files_changed: list[str],
) -> list[tuple[str, int]]:
    """Return (test_id, tier) pairs whose class name is FooTest/FooTests/TestFoo
    for any changed source file Foo.java."""
    source_stems = {
        Path(f).stem.lower() for f in files_changed
        if not (Path(f).stem.endswith("Test") or Path(f).stem.endswith("Tests")
                or Path(f).stem.startswith("Test"))
    }
    matched = []
    for test_id, tier in all_kb_tests:
        class_simple = test_id.split("::")[0].split(".")[-1].lower() if "::" in test_id else ""
        for stem in source_stems:
            if (class_simple == stem + "test"
                    or class_simple == stem + "tests"
                    or class_simple == "test" + stem):
                matched.append((test_id, tier))
                break
    return matched


def _changed_method_simples(methods_changed: list[str]) -> set[str]:
    """Extract lowercased simple method names from ClassName.methodName list.

    Skips class-level placeholders, deleted markers, and generic names.
    Requires length >= 5 to avoid noise from very short names.
    """
    result = set()
    for m in methods_changed:
        if "." not in m:
            continue
        simple = m.split(".")[-1].lower()
        if (simple in _SKIP_METHOD_NAMES
                or simple.startswith("<")
                or len(simple) < 5):
            continue
        result.add(simple)
    return result


def _method_coverage_matches(
    all_kb_tests: list[tuple[str, int]],
    methods_changed: list[str],
    tested_methods_map: "dict[str, list[str]]",
) -> list[tuple[str, int]]:
    """Return tests whose LLM-extracted tested_methods overlap with changed methods.

    This is the strongest deterministic signal: if Phase 1 LLM said a test
    covers method X, and X is changed, that test must run.
    """
    changed = _changed_method_simples(methods_changed)
    if not changed:
        return []
    matched = []
    for test_id, tier in all_kb_tests:
        tested = {m.split(".")[-1].lower() for m in tested_methods_map.get(test_id, [])}
        if tested & changed:
            matched.append((test_id, tier))
    return matched


def _test_method_name_matches(
    all_kb_tests: list[tuple[str, int]],
    methods_changed: list[str],
) -> list[tuple[str, int]]:
    """Return tests whose method name contains a changed source method name.

    E.g. testParseMultiple / testParseWithLong both match when parse() changes.
    Only fires when the method name is specific enough (length >= 5, not generic).
    """
    changed = _changed_method_simples(methods_changed)
    if not changed:
        return []
    matched = []
    for test_id, tier in all_kb_tests:
        test_method = test_id.split("::")[-1].lower() if "::" in test_id else ""
        if any(m in test_method for m in changed):
            matched.append((test_id, tier))
    return matched


def select(
    candidates: list["Candidate"],
    all_kb_tests: list[tuple[str, int]],   # [(test_id, tier), ...]
    config: "Config",
    ablation_flags: "AblationFlags | None" = None,
    # Phase 4 optional enrichment maps — all default to None for backward compat
    fixture_map: "dict[str, list[str]] | None" = None,
    sensitivity_map: "dict[str, float] | None" = None,
    topology_map: "dict[str, str] | None" = None,
    change_type: str = "general",
    files_changed: "list[str] | None" = None,
    similarity_threshold: "float | None" = None,
    test_in_diff_ids: "list[str] | None" = None,
    methods_changed: "list[str] | None" = None,
    tested_methods_map: "dict[str, list[str]] | None" = None,
) -> SelectionTrace:
    """Apply Safety Bridge, fixture bypass, semantic matching, multipliers, and cap.

    Args:
        candidates:          Top-K from Phase 2.
        all_kb_tests:        Every (test_id, tier) pair in the KB.
        config:              Config object.
        ablation_flags:      Override flags for ablation runs.
        fixture_map:         {test_id: [fixture class simple names]}.
        sensitivity_map:     {test_id: sensitivity_score float [0.1, 1.0]}.
        topology_map:        {test_id: "unit"|"integration"|"system"}.
        change_type:         Classifier output from diff_parser (e.g. "api_change").
        files_changed:       List of changed file paths.
        similarity_threshold: Effective threshold from retriever (adaptive); falls
                              back to config value when None.
    """
    from semantic_rts.config import AblationFlags
    if ablation_flags is None:
        ablation_flags = config.ablation

    files_changed = files_changed or []

    # scored: test_id → (score, reason, tier)
    scored: dict[str, tuple[float, str, int]] = {}

    semantic_only = ablation_flags.semantic_only

    # --- 0: Tests touched by the diff itself — always run ---
    tier_lookup = {tid: t for tid, t in all_kb_tests}
    if not semantic_only:
        for test_id in (test_in_diff_ids or []):
            tier = tier_lookup.get(test_id, 3)
            scored[test_id] = (1.0, "test_in_diff", tier)

    # --- 1+2: Safety Bridge ---
    if ablation_flags.safety_bridge_enabled and not semantic_only:
        for test_id, tier in all_kb_tests:
            if _effective_tier(tier, ablation_flags) == 1:
                scored[test_id] = (1.0, "safety_bridge_t1", tier)

        t2_thresh = config.selector.safety_bridge.tier_2_threshold
        for c in candidates:
            if _effective_tier(c.tier, ablation_flags) == 2 and c.score >= t2_thresh:
                if c.test_id not in scored:
                    scored[c.test_id] = (c.score, "safety_bridge_t2", c.tier)

    # --- 3: Fixture bypass ---
    if fixture_map and files_changed and not semantic_only:
        changed_class_simples = {Path(f).stem for f in files_changed}
        for test_id, tier in all_kb_tests:
            if test_id in scored:
                continue
            if changed_class_simples & set(fixture_map.get(test_id, [])):
                scored[test_id] = (0.75, "fixture_bypass", tier)

    # --- 3.5: Naming convention bypass — FooTest always runs when Foo.java changes ---
    if files_changed and not semantic_only:
        for test_id, tier in _naming_convention_matches(all_kb_tests, files_changed):
            if test_id not in scored:
                scored[test_id] = (0.95, "naming_convention", tier)

    # --- 3.6: Method coverage bypass — test's LLM-extracted methods overlap changed methods ---
    if methods_changed and tested_methods_map and not semantic_only:
        for test_id, tier in _method_coverage_matches(all_kb_tests, methods_changed, tested_methods_map):
            if test_id not in scored:
                scored[test_id] = (0.92, "method_coverage", tier)

    # --- 3.7: Test method name match — testParseX matches when parse() changes ---
    if methods_changed and not semantic_only:
        for test_id, tier in _test_method_name_matches(all_kb_tests, methods_changed):
            if test_id not in scored:
                scored[test_id] = (0.88, "test_method_match", tier)

    # --- 4: Semantic relevance ---
    sim_thresh = (
        similarity_threshold
        if similarity_threshold is not None
        else config.retrieval.similarity_threshold
    )
    for c in candidates:
        if c.score >= sim_thresh and c.test_id not in scored:
            scored[c.test_id] = (c.score, "semantic_match", c.tier)

    # --- 5: Topology multiplier ---
    trig_types = set(config.selector.topology_trigger_change_types)
    trig_pkgs = config.selector.topology_trigger_packages
    should_boost_topology = (
        change_type in trig_types
        or any(pkg in f.lower() for f in files_changed for pkg in trig_pkgs)
    )
    if should_boost_topology and topology_map:
        multiplier = config.selector.topology_multiplier
        for test_id in list(scored):
            score, reason, tier = scored[test_id]
            if reason == "safety_bridge_t1":
                continue
            if topology_map.get(test_id, "unit") in ("integration", "system"):
                scored[test_id] = (score * multiplier, reason, tier)

    # --- 6: Sensitivity multiplier ---
    if config.selector.sensitivity_multiplier_enabled and sensitivity_map:
        w = 0.08
        for test_id in list(scored):
            score, reason, tier = scored[test_id]
            if reason == "safety_bridge_t1":
                continue
            sensitivity = sensitivity_map.get(test_id, 0.5)
            scored[test_id] = (score * (1.0 + w * (sensitivity - 0.5)), reason, tier)

    # --- 7: Precision Filter ---
    dropped: list[tuple[str, DroppedReason]] = []
    to_remove: list[str] = []
    for test_id, (score, reason, tier) in scored.items():
        if _should_drop(tier, score, reason, ablation_flags, config):
            drop_reason: DroppedReason = (
                "dropped_precision_t5" if _effective_tier(tier, ablation_flags) == 5
                else "dropped_precision_t4"
            )
            dropped.append((test_id, drop_reason))
            to_remove.append(test_id)
    for tid in to_remove:
        del scored[tid]

    # --- 8: Cap — never drop protected reasons ---
    max_sel = config.selector.max_selected
    sorted_items = sorted(
        scored.items(),
        key=lambda kv: (-kv[1][0], kv[0]),
    )

    guaranteed = [(tid, v) for tid, v in sorted_items if v[1] in _PROTECTED_REASONS]
    non_guaranteed = [(tid, v) for tid, v in sorted_items if v[1] not in _PROTECTED_REASONS]

    remaining_slots = max(0, max_sel - len(guaranteed))
    kept_non_guaranteed = non_guaranteed[:remaining_slots]
    for tid, _ in non_guaranteed[remaining_slots:]:
        dropped.append((tid, "dropped_cap"))

    final_items = guaranteed + kept_non_guaranteed
    selected = [
        SelectedTest(test_id=tid, score=v[0], tier=v[2], reason=v[1])   # type: ignore[arg-type]
        for tid, v in final_items
    ]

    return SelectionTrace(selected=selected, dropped=dropped)
