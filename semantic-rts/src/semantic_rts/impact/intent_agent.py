"""LLM-based change-intent summarizer (Phase 2, step 5)."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from semantic_rts.llm.prompts import (
    INTENT_MERGE_V1,
    INTENT_MERGE_V1_TEMPLATE,
    INTENT_V1,
    INTENT_V1_TEMPLATE,
)

if TYPE_CHECKING:
    from semantic_rts.config import Config
    from semantic_rts.llm.client import GeminiClient

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_CHARS_PER_TOKEN = 4

_TEST_DIR_MARKERS = ("/test/", "/tests/", "Test.java", "Tests.java", "TestCase.java")


def _is_test_file(path: str) -> bool:
    return any(m in path for m in _TEST_DIR_MARKERS)


@dataclass
class IntentResult:
    intent_summary: str
    concepts: list[str] = field(default_factory=list)
    risk_areas: list[str] = field(default_factory=list)
    intent_failed: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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


def _call_intent_llm(
    prompt: str,
    client: "GeminiClient",
    version_tag: str,
) -> IntentResult | None:
    for attempt in range(2):
        if attempt == 1:
            prompt += "\n\nIMPORTANT: Respond with valid JSON only — no markdown, no explanation."
        try:
            result = client.chat(prompt, version_tag=version_tag)
            parsed = _parse_json(result["text"])
            if parsed:
                summary = str(parsed.get("intent_summary", "")).strip()
                concepts = [str(c) for c in parsed.get("concepts", [])]
                risk_areas = [str(r) for r in parsed.get("risk_areas", ["other"])]
                if summary:
                    return IntentResult(
                        intent_summary=summary,
                        concepts=concepts,
                        risk_areas=risk_areas,
                    )
        except Exception as exc:
            logger.warning("Intent LLM error (attempt %d): %s", attempt + 1, exc)
    return None


def _single_intent_call(
    diff_text: str,
    files_changed: list[str],
    methods_changed: list[str],
    client: "GeminiClient",
    max_diff_chars: int,
    method_sigs: list[str] | None = None,
) -> IntentResult | None:
    diff_snippet = diff_text[:max_diff_chars]
    if len(diff_text) > max_diff_chars:
        diff_snippet += "\n... [diff truncated]"

    sig_block = ""
    if method_sigs:
        sig_block = "Changed method signatures:\n" + "\n".join(f"  - {s}" for s in method_sigs) + "\n\n"

    prompt = INTENT_V1_TEMPLATE.format(
        file_list=", ".join(files_changed) or "(none)",
        method_list=", ".join(methods_changed) or "(none)",
        sig_block=sig_block,
        diff=diff_snippet,
    )
    return _call_intent_llm(prompt, client, INTENT_V1)


def _chunked_intent_call(
    diff_text: str,
    files_changed: list[str],
    methods_changed: list[str],
    client: "GeminiClient",
    max_diff_chars: int,
) -> IntentResult | None:
    """Split diff by file, call intent per file, then merge."""
    import unidiff

    try:
        patch = unidiff.PatchSet(diff_text)
    except Exception:
        return _single_intent_call(diff_text, files_changed, methods_changed, client, max_diff_chars)

    partial_intents: list[str] = []

    for pf in patch:
        file_diff = str(pf)
        file_methods = [m for m in methods_changed if pf.path in m or pf.path.split("/")[-1] in m]
        partial = _single_intent_call(file_diff, [pf.path], file_methods, client, max_diff_chars)
        if partial:
            partial_intents.append(
                f"File: {pf.path}\n"
                f"Summary: {partial.intent_summary}\n"
                f"Concepts: {', '.join(partial.concepts)}"
            )

    if not partial_intents:
        return None

    merge_prompt = INTENT_MERGE_V1_TEMPLATE.format(
        partial_intents="\n\n".join(partial_intents)
    )
    return _call_intent_llm(merge_prompt, client, INTENT_MERGE_V1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_intent(
    diff_text: str,
    files_changed: list[str],
    methods_changed: list[str],
    client: "GeminiClient",
    config: "Config",
    project_root: "str | None" = None,
) -> IntentResult:
    """Infer developer intent from a diff via LLM.

    Strips test file changes before sending to the LLM so the intent focuses
    on production code. Chunks large diffs by file and merges partial results.
    Falls back to a rule-based summary on any LLM failure.
    """
    import unidiff

    # Strip test file hunks from diff — focus LLM on production code changes
    prod_files = [f for f in files_changed if not _is_test_file(f)]
    try:
        patch = unidiff.PatchSet(diff_text)
        prod_diff = "".join(str(pf) for pf in patch if not _is_test_file(pf.path))
        if not prod_diff:
            prod_diff = diff_text  # fallback: keep original if everything was test files
    except Exception:
        prod_diff = diff_text
        prod_files = files_changed

    max_chars = config.impact.diff_max_tokens * _CHARS_PER_TOKEN

    # Extract signatures of changed methods for richer LLM context
    method_sigs: list[str] = []
    if project_root:
        try:
            from pathlib import Path
            from semantic_rts.kb.sut_linker import _extract_method_signatures
            method_names = {m.split(".")[-1] for m in methods_changed if "." in m}
            for f in prod_files:
                prod_path = Path(project_root) / f
                if prod_path.exists():
                    sigs = _extract_method_signatures(prod_path, method_names, cap=6)
                    method_sigs.extend(sigs)
                    if len(method_sigs) >= 12:
                        break
        except Exception:
            pass

    if len(prod_diff) <= max_chars:
        result = _single_intent_call(prod_diff, prod_files, methods_changed, client, max_chars, method_sigs)
    else:
        result = _chunked_intent_call(prod_diff, prod_files, methods_changed, client, max_chars)

    if result is not None:
        return result

    # Fallback: rule-based summary (no LLM)
    logger.warning("Intent agent gave up; using fallback summary.")
    return IntentResult(
        intent_summary=f"Changes to: {', '.join(files_changed) or 'unknown files'}",
        concepts=methods_changed[:10],
        risk_areas=["other"],
        intent_failed=True,
    )
