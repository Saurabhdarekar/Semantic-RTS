"""Unit tests for Phase 2: diff parser, intent agent, retriever."""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest

MINI_PROJECT = Path(__file__).parent.parent / "fixtures" / "mini_project"
SAMPLE_DIFFS = Path(__file__).parent.parent / "fixtures" / "sample_diffs"


# ---------------------------------------------------------------------------
# Diff parser
# ---------------------------------------------------------------------------

SIMPLE_DIFF = """\
--- a/src/main/java/miniproject/Auth.java
+++ b/src/main/java/miniproject/Auth.java
@@ -5,3 +5,4 @@ public class Auth {
     public static boolean validatePassword(String password) {
-        return password.length() >= 8;
+        // Strengthen: require 12+
+        return password.length() >= 12;
     }
"""

# Two separate one-line hunks — counts match exactly (1 old, 1 new each)
MULTI_FILE_DIFF = """\
--- a/src/main/java/miniproject/Auth.java
+++ b/src/main/java/miniproject/Auth.java
@@ -6,1 +6,1 @@
-        return password.length() >= 8;
+        return password.length() >= 12;
--- a/src/main/java/miniproject/Database.java
+++ b/src/main/java/miniproject/Database.java
@@ -5,1 +5,1 @@
-        conn.setAutoCommit(true);
+        conn.setAutoCommit(false);
"""


class TestParseUnifiedDiff:
    def test_parses_files_changed(self):
        from semantic_rts.impact.diff_parser import parse_unified_diff
        parsed = parse_unified_diff(SIMPLE_DIFF)
        assert "src/main/java/miniproject/Auth.java" in parsed.files_changed

    def test_diff_hash_is_hex(self):
        from semantic_rts.impact.diff_parser import parse_unified_diff
        parsed = parse_unified_diff(SIMPLE_DIFF)
        assert len(parsed.diff_hash) == 40

    def test_deterministic_hash(self):
        from semantic_rts.impact.diff_parser import parse_unified_diff
        p1 = parse_unified_diff(SIMPLE_DIFF)
        p2 = parse_unified_diff(SIMPLE_DIFF)
        assert p1.diff_hash == p2.diff_hash

    def test_multi_file_diff(self):
        from semantic_rts.impact.diff_parser import parse_unified_diff
        parsed = parse_unified_diff(MULTI_FILE_DIFF)
        assert len(parsed.files_changed) == 2

    def test_non_java_files_excluded(self):
        diff = "--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-old\n+new\n"
        from semantic_rts.impact.diff_parser import parse_unified_diff
        parsed = parse_unified_diff(diff)
        assert parsed.files_changed == []

    def test_hunk_line_numbers_captured(self):
        from semantic_rts.impact.diff_parser import parse_unified_diff
        parsed = parse_unified_diff(SIMPLE_DIFF)
        assert len(parsed.file_changes) == 1
        fc = parsed.file_changes[0]
        assert len(fc.hunks) == 1
        assert fc.hunks[0].target_start == 5

    def test_bad_diff_returns_empty(self):
        from semantic_rts.impact.diff_parser import parse_unified_diff
        parsed = parse_unified_diff("this is not a diff")
        # Should not raise; returns empty structure
        assert parsed.files_changed == []

    def test_fixture_diff_file(self):
        from semantic_rts.impact.diff_parser import parse_unified_diff
        diff_text = (SAMPLE_DIFFS / "miniproject_auth_change.diff").read_text()
        parsed = parse_unified_diff(diff_text)
        assert any("Auth.java" in f for f in parsed.files_changed)


class TestExtractChangedMethods:
    def test_finds_validatePassword(self):
        from semantic_rts.impact.diff_parser import extract_changed_methods, parse_unified_diff
        parsed = parse_unified_diff(SIMPLE_DIFF)
        methods = extract_changed_methods(parsed, project_root=str(MINI_PROJECT))
        assert any("validatePassword" in m for m in methods)

    def test_missing_file_does_not_raise(self):
        from semantic_rts.impact.diff_parser import extract_changed_methods, parse_unified_diff
        diff = "--- a/src/main/java/miniproject/Missing.java\n+++ b/src/main/java/miniproject/Missing.java\n@@ -1,3 +1,3 @@\n-old\n+new\n"
        parsed = parse_unified_diff(diff)
        # Should not raise
        methods = extract_changed_methods(parsed, project_root="/nonexistent")
        assert isinstance(methods, list)

    def test_deduplication(self):
        from semantic_rts.impact.diff_parser import extract_changed_methods, parse_unified_diff
        # Two hunks in same method should produce only one entry
        diff = """\
--- a/src/main/java/miniproject/Auth.java
+++ b/src/main/java/miniproject/Auth.java
@@ -5,1 +5,1 @@
-    public static boolean validatePassword(String password) {
+    public static boolean validatePassword(String pw) {
@@ -6,1 +6,1 @@
-        return password.length() >= 8;
+        return password.length() >= 12;
"""
        parsed = parse_unified_diff(diff)
        methods = extract_changed_methods(parsed, project_root=str(MINI_PROJECT))
        # Auth.validatePassword should appear only once
        count = sum(1 for m in methods if "validatePassword" in m)
        assert count <= 1


# ---------------------------------------------------------------------------
# Intent agent
# ---------------------------------------------------------------------------

class TestIntentAgent:
    def _make_client(self, response_json: dict) -> MagicMock:
        client = MagicMock()
        client.chat.return_value = {
            "text": json.dumps(response_json),
            "model": "mock",
            "version_tag": "INTENT_V1",
        }
        return client

    def test_parses_valid_response(self):
        from semantic_rts.config import load_config
        from semantic_rts.impact.intent_agent import analyze_intent

        cfg = load_config()
        client = self._make_client({
            "intent_summary": "Strengthens password validation to require 12 chars.",
            "concepts": ["auth", "password", "validation"],
            "risk_areas": ["security"],
        })
        result = analyze_intent(SIMPLE_DIFF, ["Auth.java"], ["Auth.validatePassword"], client, cfg)
        assert "password" in result.intent_summary.lower() or "12" in result.intent_summary
        assert "auth" in result.concepts or "password" in result.concepts
        assert result.intent_failed is False

    def test_falls_back_on_bad_json(self):
        from semantic_rts.config import load_config
        from semantic_rts.impact.intent_agent import analyze_intent

        cfg = load_config()
        client = MagicMock()
        client.chat.return_value = {
            "text": "not json at all",
            "model": "mock",
            "version_tag": "INTENT_V1",
        }
        result = analyze_intent(SIMPLE_DIFF, ["Auth.java"], ["Auth.validatePassword"], client, cfg)
        assert result.intent_failed is True
        assert "Auth.java" in result.intent_summary or "Auth.validatePassword" in result.concepts

    def test_falls_back_on_exception(self):
        from semantic_rts.config import load_config
        from semantic_rts.impact.intent_agent import analyze_intent

        cfg = load_config()
        client = MagicMock()
        client.chat.side_effect = RuntimeError("API down")
        result = analyze_intent(SIMPLE_DIFF, ["Auth.java"], [], client, cfg)
        assert result.intent_failed is True


# ---------------------------------------------------------------------------
# format_query
# ---------------------------------------------------------------------------

def test_format_query_contains_intent():
    from semantic_rts.impact.intent_agent import IntentResult
    from semantic_rts.impact.retriever import format_query

    intent = IntentResult(
        intent_summary="Changes to password logic",
        concepts=["auth", "password"],
        risk_areas=["security"],
    )
    q = format_query(intent)
    assert "Changes to password logic" in q
    assert "auth" in q
    assert "security" in q


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

def _make_store(tests: list[tuple[str, int, list[float]]]):
    """Build a tiny VectorStore from (test_id, tier, vector) tuples."""
    from semantic_rts.kb.vector_store import VectorStore
    dim = len(tests[0][2])
    store = VectorStore(dim=dim)
    store.add(
        vectors=[t[2] for t in tests],
        test_ids=[t[0] for t in tests],
        tiers=[t[1] for t in tests],
    )
    return store


def _unit_vec(dim: int, hot: int) -> list[float]:
    v = [0.0] * dim
    v[hot] = 1.0
    return v


class TestRetriever:
    def test_retrieve_returns_candidates(self):
        from semantic_rts.config import load_config
        from semantic_rts.impact.intent_agent import IntentResult
        from semantic_rts.impact.retriever import retrieve

        cfg = load_config()
        store = _make_store([
            ("AuthTest::testLogin", 1, _unit_vec(4, 0)),
            ("UtilTest::testFmt", 4, _unit_vec(4, 1)),
        ])
        embedder = MagicMock()
        embedder.embed.return_value = _unit_vec(4, 0)  # query closest to AuthTest

        intent = IntentResult(intent_summary="auth change", concepts=["auth"], risk_areas=["security"])
        candidates, trace = retrieve(intent, store, embedder, cfg)

        assert len(candidates) > 0
        assert candidates[0].test_id == "AuthTest::testLogin"
        assert candidates[0].score > candidates[1].score

    def test_retrieve_empty_kb_returns_no_candidates(self):
        from semantic_rts.config import load_config
        from semantic_rts.impact.intent_agent import IntentResult
        from semantic_rts.impact.retriever import retrieve
        from semantic_rts.kb.vector_store import VectorStore

        cfg = load_config()
        store = VectorStore(dim=4)
        embedder = MagicMock()
        embedder.embed.return_value = _unit_vec(4, 0)

        intent = IntentResult(intent_summary="some change", concepts=[], risk_areas=["other"])
        candidates, trace = retrieve(intent, store, embedder, cfg)
        assert candidates == []

    def test_trace_populated(self):
        from semantic_rts.config import load_config
        from semantic_rts.impact.intent_agent import IntentResult
        from semantic_rts.impact.retriever import retrieve

        cfg = load_config()
        store = _make_store([("t1", 3, _unit_vec(4, 0))])
        embedder = MagicMock()
        embedder.embed.return_value = _unit_vec(4, 0)

        intent = IntentResult(intent_summary="test", concepts=["c1"], risk_areas=["other"])
        _, trace = retrieve(intent, store, embedder, cfg, diff_hash="abc123", files_changed=["F.java"])

        assert trace.diff_hash == "abc123"
        assert "F.java" in trace.files_changed
        assert trace.intent_summary == "test"


# ---------------------------------------------------------------------------
# Full pipeline (Phase 2 → Phase 3)
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_end_to_end_selects_tier1_test(self):
        """Auth change should select AuthTest (tier 1) via safety bridge."""
        from semantic_rts.config import load_config
        from semantic_rts.impact.diff_parser import extract_changed_methods, parse_unified_diff
        from semantic_rts.impact.intent_agent import IntentResult, analyze_intent
        from semantic_rts.impact.retriever import retrieve
        from semantic_rts.selector.ranker import select

        cfg = load_config()

        # Build a tiny KB with AuthTest at tier 1
        store = _make_store([
            ("miniproject.AuthTest::testPasswordValidation", 1, _unit_vec(4, 0)),
            ("miniproject.ModelTest::testGetterSetter", 5, _unit_vec(4, 1)),
        ])

        embedder = MagicMock()
        embedder.embed.return_value = _unit_vec(4, 0)  # query matches AuthTest

        intent = IntentResult(
            intent_summary="Strengthens password validation",
            concepts=["auth", "password"],
            risk_areas=["security"],
        )

        candidates, _ = retrieve(intent, store, embedder, cfg)
        trace = select(candidates, store.all_tests(), cfg)

        selected_ids = {t.test_id for t in trace.selected}
        # Tier-1 AuthTest must always be in the selection (Safety Bridge)
        assert "miniproject.AuthTest::testPasswordValidation" in selected_ids
        # Confirm reason for AuthTest
        auth = next(t for t in trace.selected if "Auth" in t.test_id)
        assert auth.reason == "safety_bridge_t1"
