"""text-embedding-005 wrapper: normalize to unit length, disk cache."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

from google import genai

from semantic_rts.config import Config
from semantic_rts.llm.client import _TokenBucket


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-10:
        return vec
    return [x / norm for x in vec]


def _emb_cache_key(text: str, model: str) -> str:
    h = hashlib.sha1(usedforsecurity=False)
    h.update(text.encode("utf-8"))
    h.update(b"\x00")
    h.update(model.encode("utf-8"))
    return h.hexdigest()


class GeminiEmbedder:
    """Calls text-embedding-005 and returns 768-d unit-normalized vectors.

    Responses are cached on disk so re-runs don't re-call the API.
    """

    def __init__(self, config: Config) -> None:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")
        self._client = genai.Client(api_key=api_key)
        self._model_name = config.llm.embedding_model
        self._max_retries = config.llm.max_retries
        self._cache_only = os.environ.get("SRTS_CACHE_ONLY", "0") == "1"

        self._rpm_bucket = _TokenBucket(config.llm.rate_limit_rpm, 60.0)

        self._cache_dir = Path(config.paths.cache_dir) / "embeddings"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str, *, force: bool = False) -> list[float]:
        """Return a 768-d unit-normalized embedding for text."""
        key = _emb_cache_key(text, self._model_name)
        if not force:
            cached = self._load_cache(key)
            if cached is not None:
                return cached

        if self._cache_only:
            raise RuntimeError(
                f"SRTS_CACHE_ONLY=1 but no cached embedding for key {key}"
            )

        return self._call_with_retry(text, key)

    def embed_batch(self, texts: list[str], *, force: bool = False) -> list[list[float]]:
        """Embed a list of texts, returning one vector per text (one call per text)."""
        return [self.embed(t, force=force) for t in texts]

    def embed_batch_efficient(
        self,
        texts: list[str],
        batch_size: int = 20,
        *,
        force: bool = False,
    ) -> list[list[float]]:
        """Embed texts using batched API calls: ceil(n/batch_size) calls instead of n.

        Cache is checked per-text before calling the API, so resumed runs
        only pay for uncached texts.
        """
        results: list[list[float] | None] = [None] * len(texts)
        keys = [_emb_cache_key(t, self._model_name) for t in texts]

        uncached: list[int] = []
        if not force:
            for i, key in enumerate(keys):
                cached = self._load_cache(key)
                if cached is not None:
                    results[i] = cached
                else:
                    uncached.append(i)
        else:
            uncached = list(range(len(texts)))

        for batch_start in range(0, len(uncached), batch_size):
            batch_indices = uncached[batch_start : batch_start + batch_size]
            batch_texts = [texts[i] for i in batch_indices]
            batch_vecs = self._call_batch_with_retry(batch_texts)
            for idx, vec in zip(batch_indices, batch_vecs):
                self._save_cache(keys[idx], vec)
                results[idx] = vec

        return [r if r is not None else [] for r in results]

    def _call_batch_with_retry(self, texts: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                self._rpm_bucket.acquire()
                result = self._client.models.embed_content(
                    model=self._model_name,
                    contents=texts,
                )
                return [_normalize(list(e.values)) for e in result.embeddings]
            except Exception as exc:
                code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                is_rate_limit = code == 429 or "quota" in str(exc).lower()
                is_transient = is_rate_limit or code in (500, 503, 504)
                if is_transient and attempt < self._max_retries - 1:
                    last_error = exc
                    wait = 60 if is_rate_limit else 2 ** attempt
                    wait += random.uniform(0, 5)
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError(
            f"Batch embedding failed after {self._max_retries} retries: {last_error}"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_with_retry(self, text: str, key: str) -> list[float]:
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                self._rpm_bucket.acquire()

                result = self._client.models.embed_content(
                    model=self._model_name,
                    contents=text,
                )
                raw_vec = list(result.embeddings[0].values)
                vec = _normalize(raw_vec)
                self._save_cache(key, vec)
                return vec

            except Exception as exc:
                code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                is_transient = code in (429, 500, 503, 504) or "quota" in str(exc).lower()
                if is_transient and attempt < self._max_retries - 1:
                    last_error = exc
                    time.sleep(2 ** attempt)
                else:
                    raise

        raise RuntimeError(
            f"Embedding call failed after {self._max_retries} retries: {last_error}"
        )

    def _load_cache(self, key: str) -> list[float] | None:
        path = self._cache_dir / f"{key}.json"
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def _save_cache(self, key: str, vec: list[float]) -> None:
        with open(self._cache_dir / f"{key}.json", "w", encoding="utf-8") as f:
            json.dump(vec, f)
