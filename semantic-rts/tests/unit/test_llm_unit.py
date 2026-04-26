"""Unit tests for LLM client helpers — no network, no API key needed."""

from __future__ import annotations

import json
import math
import time

import pytest

from semantic_rts.llm.client import _TokenBucket, _make_cache_key
from semantic_rts.llm.embeddings import _normalize, _emb_cache_key


# ---------------------------------------------------------------------------
# TokenBucket
# ---------------------------------------------------------------------------

class TestTokenBucket:
    def test_first_acquire_instant(self):
        bucket = _TokenBucket(rate=10, per_seconds=1.0)
        t0 = time.monotonic()
        bucket.acquire()
        assert time.monotonic() - t0 < 0.1

    def test_fills_over_time(self):
        bucket = _TokenBucket(rate=2, per_seconds=1.0)
        bucket.acquire()
        bucket.acquire()           # drains to 0
        # Manually refill by backdating last_refill
        bucket._last_refill -= 1.5
        bucket._refill()
        assert bucket._tokens >= 1.5   # should have 3 * (1.5s * 2 tokens/s) but capped at 2

    def test_tokens_capped_at_rate(self):
        bucket = _TokenBucket(rate=5, per_seconds=1.0)
        bucket._last_refill -= 100   # long time elapsed
        bucket._refill()
        assert bucket._tokens == 5.0  # never exceeds rate


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

class TestCacheKey:
    def test_deterministic(self):
        k1 = _make_cache_key("hello", "gemini-1.5", "V1")
        k2 = _make_cache_key("hello", "gemini-1.5", "V1")
        assert k1 == k2

    def test_different_prompt_different_key(self):
        k1 = _make_cache_key("foo", "gemini-1.5", "V1")
        k2 = _make_cache_key("bar", "gemini-1.5", "V1")
        assert k1 != k2

    def test_different_model_different_key(self):
        k1 = _make_cache_key("foo", "gemini-1.5", "V1")
        k2 = _make_cache_key("foo", "gemini-2.0", "V1")
        assert k1 != k2

    def test_different_version_different_key(self):
        k1 = _make_cache_key("foo", "gemini-1.5", "V1")
        k2 = _make_cache_key("foo", "gemini-1.5", "V2")
        assert k1 != k2

    def test_embedding_cache_key_deterministic(self):
        k1 = _emb_cache_key("text", "text-embedding-005")
        k2 = _emb_cache_key("text", "text-embedding-005")
        assert k1 == k2


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_unit_length(self):
        vec = _normalize([3.0, 4.0])
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-6

    def test_already_unit(self):
        vec = [1.0, 0.0]
        result = _normalize(vec)
        assert result == pytest.approx([1.0, 0.0])

    def test_zero_vector_unchanged(self):
        vec = [0.0, 0.0, 0.0]
        result = _normalize(vec)
        assert result == [0.0, 0.0, 0.0]

    def test_negative_values(self):
        vec = _normalize([-3.0, 4.0])
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Disk cache round-trip (uses tmp_path, no API)
# ---------------------------------------------------------------------------

class TestDiskCache:
    def test_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-unit-test")
        monkeypatch.setenv("SRTS_CACHE_ONLY", "0")

        from semantic_rts.config import load_config
        cfg = load_config(overrides={
            "paths.cache_dir": str(tmp_path / "cache"),
            "paths.logs_dir": str(tmp_path / "logs"),
        })

        from semantic_rts.llm.client import GeminiClient
        # Instantiate to create dirs; don't actually call the API
        # Patch _call_with_retry to return a fake response
        client = GeminiClient.__new__(GeminiClient)
        client._model_name = "fake-model"
        client._max_retries = 1
        client._cache_only = False
        client._cache_dir = tmp_path / "cache" / "llm"
        client._cache_dir.mkdir(parents=True, exist_ok=True)
        client._log_path = tmp_path / "logs" / "llm_calls.jsonl"
        client._log_path.parent.mkdir(parents=True, exist_ok=True)
        client._request_count = 0
        from threading import Lock
        client._count_lock = Lock()

        fake = {"text": "hello", "model": "fake-model", "version_tag": "V1"}
        key = _make_cache_key("prompt", "fake-model", "V1")
        client._save_cache(key, fake)

        loaded = client._load_cache(key)
        assert loaded == fake

    def test_missing_cache_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-unit-test")

        from semantic_rts.llm.client import GeminiClient
        client = GeminiClient.__new__(GeminiClient)
        client._cache_dir = tmp_path / "nonexistent"
        result = client._load_cache("abcd1234")
        assert result is None
