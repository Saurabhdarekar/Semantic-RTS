"""Unit tests for Phase 1 KB builder — mocked LLM calls, real FAISS."""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

MINI_PROJECT = Path(__file__).parent.parent / "fixtures" / "mini_project"


def _make_mock_client(summary="Tests a method.", concepts=None):
    """Return a GeminiClient mock whose chat() returns a valid summarizer response."""
    concepts = concepts or ["test", "example"]
    client = MagicMock()
    client.chat.return_value = {
        "text": json.dumps({"summary": summary, "concepts": concepts}),
        "model": "mock-model",
        "version_tag": "SUMMARIZER_V1",
    }
    return client


def _make_mock_embedder(dim=768):
    """Return a GeminiEmbedder mock whose embed() returns a unit vector."""
    embedder = MagicMock()
    vec = [1.0 / math.sqrt(dim)] * dim
    embedder.embed.return_value = vec
    return embedder


# ---------------------------------------------------------------------------
# format_for_embedding
# ---------------------------------------------------------------------------

def test_format_for_embedding_contains_summary():
    from semantic_rts.kb.builder import format_for_embedding
    from semantic_rts.kb.test_parser import TestMethod
    tm = TestMethod(
        test_id="pkg.Foo::bar", class_fqn="pkg.Foo", class_simple="Foo",
        method="bar", file_path="Foo.java", junit="4",
        source="", source_hash="abc",
        summary="verifies login logic", concepts=["auth", "login"], tier=1,
    )
    text = format_for_embedding(tm)
    assert "verifies login logic" in text
    assert "auth" in text
    assert "1" in text   # tier


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class TestVectorStore:
    def test_add_and_search(self):
        from semantic_rts.kb.vector_store import VectorStore
        import numpy as np

        store = VectorStore(dim=4)
        vec_a = [1.0, 0.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0, 0.0]
        store.add([vec_a, vec_b], ["test_a", "test_b"], tiers=[1, 3])

        scores, ids = store.search([1.0, 0.0, 0.0, 0.0], k=2)
        assert ids[0] == "test_a"
        assert scores[0] > scores[1]

    def test_all_tests_returns_pairs(self):
        from semantic_rts.kb.vector_store import VectorStore

        store = VectorStore(dim=2)
        store.add([[1.0, 0.0], [0.0, 1.0]], ["a", "b"], tiers=[1, 2])
        pairs = store.all_tests()
        assert set(pairs) == {("a", 1), ("b", 2)}

    def test_save_and_load(self, tmp_path):
        from semantic_rts.kb.vector_store import VectorStore

        store = VectorStore(dim=2)
        store.add([[1.0, 0.0]], ["t1"], tiers=[3])
        store.save(tmp_path)

        loaded = VectorStore.load(tmp_path)
        assert loaded.size == 1
        assert loaded.test_id_at(0) == "t1"
        assert loaded.tier_at(0) == 3

    def test_search_empty_returns_empty(self):
        from semantic_rts.kb.vector_store import VectorStore

        store = VectorStore(dim=4)
        scores, ids = store.search([1.0, 0.0, 0.0, 0.0], k=5)
        assert scores == []
        assert ids == []


# ---------------------------------------------------------------------------
# End-to-end build_kb on mini fixture (LLM mocked)
# ---------------------------------------------------------------------------

@pytest.fixture
def mini_kb_dir(tmp_path):
    return tmp_path / "kb" / "MiniProject"


def test_build_kb_produces_jsonl_and_faiss(mini_kb_dir, monkeypatch):
    """Full Phase 1 pipeline on mini fixture with mocked LLM."""
    from semantic_rts.config import load_config
    from semantic_rts.kb.builder import build_kb

    cfg = load_config(overrides={
        "paths.kb_dir": str(mini_kb_dir.parent),
        "paths.cache_dir": str(mini_kb_dir.parent / "cache"),
        "paths.logs_dir": str(mini_kb_dir.parent / "logs"),
    })

    mock_client = _make_mock_client()
    mock_embedder = _make_mock_embedder()

    with (
        patch("semantic_rts.kb.builder.GeminiClient", return_value=mock_client),
        patch("semantic_rts.kb.builder.GeminiEmbedder", return_value=mock_embedder),
    ):
        build_kb(MINI_PROJECT, "MiniProject", cfg)

    # Check tests.jsonl
    jsonl = mini_kb_dir / "tests.jsonl"
    assert jsonl.exists()
    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    assert len(rows) == 5, f"Expected 5 rows, got {len(rows)}"

    for row in rows:
        assert row["summary"], f"Empty summary for {row['test_id']}"
        assert 1 <= row["tier"] <= 5, f"Invalid tier {row['tier']}"
        assert row["source_hash"]

    # Check FAISS index
    from semantic_rts.kb.vector_store import VectorStore
    store = VectorStore.load(mini_kb_dir)
    assert store.size == 5


def test_build_kb_resumes_without_re_enriching(mini_kb_dir, monkeypatch):
    """Running build_kb twice should not call the LLM a second time for unchanged tests."""
    from semantic_rts.config import load_config
    from semantic_rts.kb.builder import build_kb

    cfg = load_config(overrides={
        "paths.kb_dir": str(mini_kb_dir.parent),
        "paths.cache_dir": str(mini_kb_dir.parent / "cache"),
        "paths.logs_dir": str(mini_kb_dir.parent / "logs"),
    })

    mock_client = _make_mock_client()
    mock_embedder = _make_mock_embedder()

    patch_client = patch("semantic_rts.kb.builder.GeminiClient", return_value=mock_client)
    patch_embedder = patch("semantic_rts.kb.builder.GeminiEmbedder", return_value=mock_embedder)

    with patch_client, patch_embedder:
        build_kb(MINI_PROJECT, "MiniProject", cfg)

    first_call_count = mock_client.chat.call_count

    # Second run with resume=True should not re-call chat
    with patch_client, patch_embedder:
        build_kb(MINI_PROJECT, "MiniProject", cfg, resume=True)

    # No additional chat calls on second run
    assert mock_client.chat.call_count == first_call_count


def test_build_kb_tiers_assigned(mini_kb_dir):
    """AuthTest should get Tier 1 via the rule pass (keyword 'auth')."""
    from semantic_rts.config import load_config
    from semantic_rts.kb.builder import build_kb

    cfg = load_config(overrides={
        "paths.kb_dir": str(mini_kb_dir.parent),
        "paths.cache_dir": str(mini_kb_dir.parent / "cache"),
        "paths.logs_dir": str(mini_kb_dir.parent / "logs"),
    })

    mock_client = _make_mock_client()
    mock_embedder = _make_mock_embedder()

    with (
        patch("semantic_rts.kb.builder.GeminiClient", return_value=mock_client),
        patch("semantic_rts.kb.builder.GeminiEmbedder", return_value=mock_embedder),
    ):
        build_kb(MINI_PROJECT, "MiniProject", cfg)

    jsonl = mini_kb_dir / "tests.jsonl"
    rows = {json.loads(l)["test_id"]: json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()}

    auth_row = next(r for tid, r in rows.items() if "Auth" in tid)
    assert auth_row["tier"] == 1, f"AuthTest should be Tier 1, got {auth_row['tier']}"

    model_row = next(r for tid, r in rows.items() if "Model" in tid)
    assert model_row["tier"] == 5, f"ModelTest should be Tier 5, got {model_row['tier']}"
