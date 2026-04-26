"""Integration tests for live Gemini API calls.

Run with:  RUN_LIVE_TESTS=1 pytest tests/integration/

These tests make real API calls and consume free-tier quota (~3 requests).
They assert only response shape, not content, so they're model-agnostic.
"""

from __future__ import annotations

import math
import os

import pytest

LIVE = pytest.mark.skipif(
    not os.environ.get("RUN_LIVE_TESTS"),
    reason="Set RUN_LIVE_TESTS=1 to run live API tests",
)


@LIVE
def test_gemini_chat_returns_text():
    from semantic_rts.config import load_config
    from semantic_rts.llm.client import GeminiClient

    cfg = load_config()
    client = GeminiClient(cfg)
    result = client.chat(
        'Reply with exactly this JSON and nothing else: {"ok": true}',
        version_tag="TEST_CHAT_V1",
        force=True,
    )

    assert isinstance(result, dict)
    assert "text" in result
    assert isinstance(result["text"], str)
    assert len(result["text"]) > 0
    assert result["model"] == cfg.llm.chat_model


@LIVE
def test_gemini_chat_cache_hit(tmp_path, monkeypatch):
    """Second call with same prompt must return cached result without hitting API."""
    monkeypatch.setenv("SRTS_CACHE_ONLY", "0")
    from semantic_rts.config import load_config
    from semantic_rts.llm.client import GeminiClient

    cfg = load_config(overrides={
        "paths.cache_dir": str(tmp_path / "cache"),
        "paths.logs_dir": str(tmp_path / "logs"),
    })
    client = GeminiClient(cfg)

    prompt = "Reply with: {\"cached\": true}"
    r1 = client.chat(prompt, "TEST_CACHE_V1", force=True)
    count_after_first = client.request_count

    r2 = client.chat(prompt, "TEST_CACHE_V1")   # should hit cache
    assert client.request_count == count_after_first   # no extra API call
    assert r1["text"] == r2["text"]


@LIVE
def test_gemini_embed_shape_and_normalization():
    from semantic_rts.config import load_config
    from semantic_rts.llm.embeddings import GeminiEmbedder

    cfg = load_config()
    embedder = GeminiEmbedder(cfg)
    vec = embedder.embed("verifies that login fails with wrong password", force=True)

    assert isinstance(vec, list)
    assert len(vec) == cfg.vector_store.embedding_dim   # 768

    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"


@LIVE
def test_gemini_embed_cache_hit(tmp_path, monkeypatch):
    monkeypatch.setenv("SRTS_CACHE_ONLY", "0")
    from semantic_rts.config import load_config
    from semantic_rts.llm.embeddings import GeminiEmbedder

    cfg = load_config(overrides={
        "paths.cache_dir": str(tmp_path / "cache"),
        "paths.logs_dir": str(tmp_path / "logs"),
    })
    embedder = GeminiEmbedder(cfg)
    text = "unique test phrase for cache test"

    v1 = embedder.embed(text, force=True)
    v2 = embedder.embed(text)              # should come from cache
    assert v1 == v2


@LIVE
def test_cache_only_raises_on_miss(tmp_path, monkeypatch):
    monkeypatch.setenv("SRTS_CACHE_ONLY", "1")
    from semantic_rts.config import load_config
    from semantic_rts.llm.client import GeminiClient

    cfg = load_config(overrides={
        "paths.cache_dir": str(tmp_path / "cache"),
        "paths.logs_dir": str(tmp_path / "logs"),
    })
    client = GeminiClient(cfg)
    with pytest.raises(RuntimeError, match="SRTS_CACHE_ONLY"):
        client.chat("this prompt has no cached response xyz123", "V1")
