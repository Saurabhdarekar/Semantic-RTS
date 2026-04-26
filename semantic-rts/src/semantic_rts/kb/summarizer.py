"""LLM-based test summarizer (Phase 1, step 3c)."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from semantic_rts.llm.prompts import SUMMARIZER_V1, SUMMARIZER_V1_TEMPLATE

if TYPE_CHECKING:
    from semantic_rts.kb.test_parser import TestMethod
    from semantic_rts.llm.client import GeminiClient

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def summarize_test(tm: "TestMethod", client: "GeminiClient") -> tuple[str, list[str]]:
    """Return (summary, concepts) for a test method via LLM.

    Retries once with a stricter JSON reminder on parse failure.
    Falls back to a canned summary on second failure (never raises).
    """
    prompt = SUMMARIZER_V1_TEMPLATE.format(
        class_fqn=tm.class_fqn,
        method_name=tm.method,
        source=tm.source,
    )

    for attempt in range(2):
        if attempt == 1:
            prompt += "\n\nIMPORTANT: Respond with valid JSON only — no markdown, no explanation."

        try:
            result = client.chat(prompt, version_tag=SUMMARIZER_V1)
            parsed = _parse_json_response(result["text"])
            if parsed:
                summary = str(parsed.get("summary", "")).strip()
                concepts = [str(c) for c in parsed.get("concepts", [])]
                if summary:
                    return summary, concepts
        except Exception as exc:
            logger.warning("Summarizer error for %s (attempt %d): %s", tm.test_id, attempt + 1, exc)

    # Fallback
    logger.warning("Summarizer gave up for %s; using fallback summary.", tm.test_id)
    return f"Tests {tm.class_simple}.{tm.method}", []


def _parse_json_response(text: str) -> dict | None:
    text = text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None
