"""Gemini chat wrapper: retry, token-bucket rate limiting, disk cache, JSONL logging."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types as genai_types

from semantic_rts.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token-bucket rate limiter (thread-safe)
# ---------------------------------------------------------------------------

class _TokenBucket:
    def __init__(self, rate: float, per_seconds: float) -> None:
        self._tokens = rate
        self._rate = rate
        self._per_seconds = per_seconds
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, n: float = 1.0) -> None:
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= n:
                    self._tokens -= n
                    return
                wait = (n - self._tokens) * (self._per_seconds / self._rate)
            time.sleep(min(wait, 1.0))

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._rate, self._tokens + elapsed * self._rate / self._per_seconds)
        self._last_refill = now


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def _make_cache_key(prompt: str, model: str, version_tag: str) -> str:
    h = hashlib.sha1(usedforsecurity=False)
    h.update(prompt.encode("utf-8"))
    h.update(b"\x00")
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(version_tag.encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# GeminiClient
# ---------------------------------------------------------------------------

class GeminiClient:
    """Wraps google-genai chat with retry, rate-limit, disk cache, and JSONL logging."""

    def __init__(self, config: Config) -> None:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Copy .env.example to .env and add your key."
            )

        self._client = genai.Client(api_key=api_key)
        self._model_name = config.llm.chat_model
        self._max_retries = config.llm.max_retries
        self._cache_only = os.environ.get("SRTS_CACHE_ONLY", "0") == "1"

        self._rpm_bucket = _TokenBucket(config.llm.rate_limit_rpm, 60.0)

        self._cache_dir = Path(config.paths.cache_dir) / "llm"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        log_dir = Path(config.paths.logs_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = log_dir / "llm_calls.jsonl"

        self._request_count = 0
        self._count_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, prompt: str, version_tag: str, *, force: bool = False) -> dict[str, Any]:
        """Send a prompt and return {"text": str, "model": str, "version_tag": str}."""
        key = _make_cache_key(prompt, self._model_name, version_tag)
        if not force:
            cached = self._load_cache(key)
            if cached is not None:
                self._log(version_tag, cached=True, latency_ms=0)
                return cached

        if self._cache_only:
            raise RuntimeError(
                f"SRTS_CACHE_ONLY=1 but no cached response for key {key} "
                f"(version_tag={version_tag!r})"
            )

        return self._call_with_retry(prompt, version_tag, key)

    @property
    def request_count(self) -> int:
        with self._count_lock:
            return self._request_count

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_with_retry(self, prompt: str, version_tag: str, key: str) -> dict[str, Any]:
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            t0 = time.monotonic()
            try:
                self._rpm_bucket.acquire()
                with self._count_lock:
                    self._request_count += 1

                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(temperature=0),
                )
                text = response.text
                latency_ms = (time.monotonic() - t0) * 1000

                result: dict[str, Any] = {
                    "text": text,
                    "model": self._model_name,
                    "version_tag": version_tag,
                }
                self._save_cache(key, result)
                self._log(version_tag, cached=False, latency_ms=latency_ms)
                return result

            except Exception as exc:
                # Retry on rate-limit / transient server errors
                code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                is_transient = code in (429, 500, 503, 504) or "quota" in str(exc).lower()
                if is_transient and attempt < self._max_retries - 1:
                    last_error = exc
                    wait = 2 ** attempt
                    logger.warning(
                        "Transient error attempt %d/%d: %s. Waiting %ds.",
                        attempt + 1, self._max_retries, exc, wait,
                    )
                    self._log(version_tag, cached=False, latency_ms=0, error=str(exc))
                    time.sleep(wait)
                else:
                    self._log(version_tag, cached=False, latency_ms=0, error=str(exc))
                    raise

        raise RuntimeError(
            f"LLM call failed after {self._max_retries} retries: {last_error}"
        )

    def _load_cache(self, key: str) -> dict[str, Any] | None:
        path = self._cache_dir / f"{key}.json"
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def _save_cache(self, key: str, data: dict[str, Any]) -> None:
        with open(self._cache_dir / f"{key}.json", "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _log(
        self,
        version_tag: str,
        *,
        cached: bool,
        latency_ms: float,
        error: str | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "ts": time.time(),
            "method": version_tag,
            "model": self._model_name,
            "cached": cached,
            "latency_ms": round(latency_ms, 1),
        }
        if error:
            entry["error"] = error
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass
