"""Unit tests for Phase 3 selector (pure Python, no LLM needed)."""

from __future__ import annotations

import pytest
from semantic_rts.config import AblationFlags, Config, load_config
from semantic_rts.impact.retriever import Candidate
from semantic_rts.selector.ranker import select


def _candidates(*args: tuple) -> list[Candidate]:
    """args: (test_id, score, tier, rank)"""
    return [Candidate(test_id=a[0], score=a[1], tier=a[2], rank=a[3]) for a in args]


def _kb_tests(*args: tuple) -> list[tuple[str, int]]:
    return list(args)


@pytest.fixture
def cfg() -> Config:
    return load_config()


def test_tier1_always_included(cfg):
    kb = _kb_tests(("sec.AuthTest::testLogin", 1), ("util.FooTest::testX", 4))
    cands = _candidates(("util.FooTest::testX", 0.9, 4, 0))

    trace = select(cands, kb, cfg)
    ids = {t.test_id for t in trace.selected}
    assert "sec.AuthTest::testLogin" in ids


def test_tier1_never_capped(cfg):
    """Safety Bridge Tier-1 entries must survive even if max_selected is tiny."""
    from semantic_rts.config import SelectorConfig, SafetyBridgeConfig, PrecisionFilterConfig
    import copy
    cfg2 = cfg.model_copy(
        update={"selector": cfg.selector.model_copy(update={"max_selected": 1})}
    )
    # Two Tier-1 tests + one semantic match
    kb = _kb_tests(("t1a", 1), ("t1b", 1))
    cands = _candidates(("match1", 0.8, 3, 0))

    trace = select(cands, kb, cfg2)
    ids = {t.test_id for t in trace.selected}
    assert "t1a" in ids
    assert "t1b" in ids


def test_semantic_match_included_above_threshold(cfg):
    kb = _kb_tests()
    cands = _candidates(("test.Foo::bar", 0.80, 3, 0))
    trace = select(cands, kb, cfg)
    ids = {t.test_id for t in trace.selected}
    assert "test.Foo::bar" in ids


def test_semantic_match_excluded_below_threshold(cfg):
    kb = _kb_tests()
    # Score 0.30 is below default threshold 0.55
    cands = _candidates(("test.Foo::bar", 0.30, 3, 0))
    trace = select(cands, kb, cfg)
    ids = {t.test_id for t in trace.selected}
    assert "test.Foo::bar" not in ids


def test_precision_filter_drops_tier5_low_score(cfg):
    kb = _kb_tests()
    # Score 0.50 < tier_5_min 0.65 → should be dropped
    cands = _candidates(("t.Getter::testGet", 0.60, 5, 0))
    trace = select(cands, kb, cfg)
    ids = {t.test_id for t in trace.selected}
    assert "t.Getter::testGet" not in ids


def test_precision_filter_keeps_tier5_high_score(cfg):
    kb = _kb_tests()
    # Score 0.70 >= tier_5_min 0.65 → kept
    cands = _candidates(("t.Getter::testGet", 0.70, 5, 0))
    trace = select(cands, kb, cfg)
    ids = {t.test_id for t in trace.selected}
    assert "t.Getter::testGet" in ids


def test_ablation_no_bridge(cfg):
    flags = AblationFlags(safety_bridge_enabled=False)
    kb = _kb_tests(("sec.AuthTest::testLogin", 1))
    cands = _candidates()

    trace = select(cands, kb, cfg, ablation_flags=flags)
    ids = {t.test_id for t in trace.selected}
    # With bridge disabled, Tier-1 test NOT auto-included
    assert "sec.AuthTest::testLogin" not in ids


def test_tier2_included_above_threshold(cfg):
    kb = _kb_tests(("db.TxTest::testRollback", 2))
    cands = _candidates(("db.TxTest::testRollback", 0.50, 2, 0))
    # tier_2_threshold is 0.40 by default
    trace = select(cands, kb, cfg)
    ids = {t.test_id for t in trace.selected}
    assert "db.TxTest::testRollback" in ids


def test_trace_records_dropped_reasons(cfg):
    kb = _kb_tests()
    # Tier 5, score below tier_5_min — should appear in dropped list
    cands = _candidates(("t.Getter::testGet", 0.60, 5, 0))
    trace = select(cands, kb, cfg)
    dropped_ids = {d[0] for d in trace.dropped}
    assert "t.Getter::testGet" in dropped_ids


def test_metrics_compute():
    from semantic_rts.eval.metrics import compute_metrics
    selected = ["a", "b", "c"]
    failing = ["a", "d"]
    all_tests = ["a", "b", "c", "d", "e"]
    m = compute_metrics(selected, failing, all_tests, "ours", "Chart", 1)
    assert m.recall == pytest.approx(0.5)           # caught 1 of 2 failing
    assert m.selection_rate == pytest.approx(0.6)   # 3 of 5 tests
    assert m.precision == pytest.approx(1/3, abs=1e-6)
