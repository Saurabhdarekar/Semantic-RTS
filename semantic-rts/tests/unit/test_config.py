"""Unit tests for config loading and Pydantic models."""

from __future__ import annotations

import pytest
from semantic_rts.config import Config, TierKeywords, load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg.llm.provider == "gemini"
    assert cfg.retrieval.top_k == 30
    assert cfg.selector.max_selected == 100
    assert cfg.selector.safety_bridge.always_include_tier_1 is True


def test_tier_keywords_for_tier():
    kw = TierKeywords()
    assert "security" in kw.for_tier(1)
    assert "database" in kw.for_tier(2)
    assert kw.for_tier(6) == []   # out-of-range → empty


def test_deep_merge_overlay(tmp_path):
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("selector:\n  max_selected: 42\n")
    cfg = load_config(overlay_path=overlay)
    assert cfg.selector.max_selected == 42
    # other fields unchanged
    assert cfg.retrieval.top_k == 30


def test_env_var_substitution(monkeypatch):
    monkeypatch.setenv("DEFECTS4J_HOME", "/opt/d4j")
    cfg = load_config()
    assert cfg.paths.defects4j_home == "/opt/d4j"


def test_set_override():
    cfg = load_config(overrides={"retrieval.top_k": "50"})
    # overrides are strings; pydantic coerces
    assert cfg.retrieval.top_k == 50


def test_kb_path_helper(default_config):
    p = default_config.paths.kb_path("Chart")
    assert str(p).endswith("Chart") or "Chart" in str(p)
