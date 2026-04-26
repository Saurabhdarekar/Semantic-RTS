"""Two-stage tier classifier: rule pass → LLM fallback → default Tier 3."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Literal

from semantic_rts.llm.prompts import TIER_CLASSIFIER_V1, TIER_CLASSIFIER_V1_TEMPLATE

if TYPE_CHECKING:
    from semantic_rts.config import Config, TierKeywords
    from semantic_rts.kb.test_parser import TestMethod
    from semantic_rts.llm.client import GeminiClient

logger = logging.getLogger(__name__)

TierSource = Literal["rule", "llm", "default"]

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def classify_tier_rule(
    tm: "TestMethod",
    tier_keywords: "TierKeywords",
) -> tuple[int, TierSource] | None:
    """Rule-based tier classification.

    Returns (tier, "rule") if any keyword matches, else None.
    Tier 1 takes precedence over Tier 2, etc. (first match wins).
    """
    haystack = " ".join([
        tm.class_simple,
        tm.method,
        tm.summary,
        *tm.concepts,
    ]).lower()

    for tier_num in range(1, 6):
        keywords = tier_keywords.for_tier(tier_num)
        if any(kw.lower() in haystack for kw in keywords):
            return (tier_num, "rule")

    return None


def _classify_tier_llm(tm: "TestMethod", client: "GeminiClient") -> tuple[int, TierSource]:
    """LLM fallback tier classification. Returns (tier, "llm") or (3, "default")."""
    prompt = TIER_CLASSIFIER_V1_TEMPLATE.format(
        class_fqn=tm.class_fqn,
        summary=tm.summary,
        concepts=", ".join(tm.concepts),
    )

    for attempt in range(2):
        if attempt == 1:
            prompt += '\n\nRespond with valid JSON only: {"tier": <1-5>, "reason": "<brief>"}'
        try:
            result = client.chat(prompt, version_tag=TIER_CLASSIFIER_V1)
            text = result["text"].strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            match = _JSON_RE.search(text)
            if match:
                data = json.loads(match.group())
                tier = int(data.get("tier", 3))
                if 1 <= tier <= 5:
                    return (tier, "llm")
        except Exception as exc:
            logger.warning("Tier LLM error for %s (attempt %d): %s", tm.test_id, attempt + 1, exc)

    logger.warning("Tier classifier gave up for %s; defaulting to Tier 3.", tm.test_id)
    return (3, "default")


def classify_tier(
    tm: "TestMethod",
    config: "Config",
    client: "GeminiClient | None" = None,
) -> tuple[int, TierSource]:
    """Two-stage classifier: rule pass → LLM fallback → default Tier 3."""
    result = classify_tier_rule(tm, config.kb.tier_keywords)
    if result is not None:
        return result

    if client is not None:
        return _classify_tier_llm(tm, client)

    return (3, "default")
