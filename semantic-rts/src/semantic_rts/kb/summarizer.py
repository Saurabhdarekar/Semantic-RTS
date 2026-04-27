"""LLM-based test summarizer (Phase 1, step 3c).

Uses SUMMARIZER_V2: single call returns summary + condition +
tested_methods + concepts + tier, eliminating the separate tier-classifier
LLM call and producing richer embeddings.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from semantic_rts.llm.prompts import SUMMARIZER_V2, SUMMARIZER_V2_TEMPLATE

if TYPE_CHECKING:
    from semantic_rts.kb.test_parser import TestMethod
    from semantic_rts.llm.client import GeminiClient

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def summarize_test(
    tm: "TestMethod",
    client: "GeminiClient",
    project_path: Path | None = None,
) -> tuple[str, list[str], int, list[str], str]:
    """Return (summary, concepts, tier, tested_methods, condition) via a single LLM call.

    Falls back gracefully on parse failure — never raises.
    """
    from semantic_rts.kb.sut_linker import get_sut_context

    sut_ctx = ""
    if project_path is not None:
        try:
            sut_ctx = get_sut_context(tm, project_path)
        except Exception:
            pass  # SUT linking is best-effort

    sut_block = f"\n{sut_ctx}\n" if sut_ctx else ""

    prompt = SUMMARIZER_V2_TEMPLATE.format(
        class_fqn=tm.class_fqn,
        method_name=tm.method,
        sut_context=sut_block,
        source=tm.source,
    )

    for attempt in range(2):
        if attempt == 1:
            prompt += "\n\nIMPORTANT: Respond with valid JSON only — no markdown, no explanation."

        try:
            result = client.chat(prompt, version_tag=SUMMARIZER_V2)
            parsed = _parse_json(result["text"])
            if parsed:
                summary = str(parsed.get("summary", "")).strip()
                condition = str(parsed.get("condition", "")).strip()
                tested_methods = [str(m) for m in parsed.get("tested_methods", [])]
                concepts = [str(c) for c in parsed.get("concepts", [])]
                tier_raw = parsed.get("tier", 3)
                try:
                    tier = int(tier_raw)
                    tier = max(1, min(5, tier))
                except (TypeError, ValueError):
                    tier = 3
                if summary:
                    return summary, concepts, tier, tested_methods, condition
        except Exception as exc:
            logger.warning(
                "Summarizer error for %s (attempt %d): %s",
                tm.test_id, attempt + 1, exc,
            )

    logger.warning("Summarizer gave up for %s; using fallback.", tm.test_id)
    return f"Tests {tm.class_simple}.{tm.method}", [], 3, [], ""


def _parse_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None
