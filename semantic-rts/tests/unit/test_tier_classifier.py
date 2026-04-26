"""Unit tests for the rule-based tier classifier (no LLM needed)."""

from __future__ import annotations

import pytest
from semantic_rts.config import TierKeywords
from semantic_rts.kb.test_parser import TestMethod
from semantic_rts.kb.tier_classifier import classify_tier_rule


def _make_tm(**kwargs) -> TestMethod:
    defaults = dict(
        test_id="pkg.Foo::bar",
        class_fqn="pkg.Foo",
        class_simple="Foo",
        method="bar",
        file_path="src/test/java/pkg/Foo.java",
        junit="4",
        source="",
        source_hash="abc",
        summary="",
        concepts=[],
    )
    defaults.update(kwargs)
    return TestMethod(**defaults)


def test_tier1_by_class_name():
    tm = _make_tm(class_simple="AuthServiceTest", summary="tests login")
    result = classify_tier_rule(tm, TierKeywords())
    assert result is not None
    tier, source = result
    assert tier == 1
    assert source == "rule"


def test_tier2_by_summary():
    tm = _make_tm(summary="verifies database transaction rollback")
    tier, source = classify_tier_rule(tm, TierKeywords())
    assert tier == 2


def test_tier5_by_method():
    tm = _make_tm(method="testGetterSetter", summary="tests setter method")
    tier, _ = classify_tier_rule(tm, TierKeywords())
    assert tier == 5


def test_no_match_returns_none():
    tm = _make_tm(class_simple="XyzTest", method="testXyz", summary="verifies xyz logic")
    result = classify_tier_rule(tm, TierKeywords())
    assert result is None


def test_tier1_wins_over_tier4():
    """A test that mentions both 'auth' and 'util' should get Tier 1."""
    tm = _make_tm(summary="auth util helper")
    tier, _ = classify_tier_rule(tm, TierKeywords())
    assert tier == 1


def test_concepts_used_in_matching():
    tm = _make_tm(concepts=["payment", "checkout"])
    tier, _ = classify_tier_rule(tm, TierKeywords())
    assert tier == 1
