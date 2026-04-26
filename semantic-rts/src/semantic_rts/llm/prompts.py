"""Versioned prompt templates.

All prompts are named constants so changing a prompt automatically busts
the LLM response cache (cache key includes the version tag).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Phase 1 — Test Summarizer
# ---------------------------------------------------------------------------

SUMMARIZER_V1 = "SUMMARIZER_V1"

SUMMARIZER_V1_TEMPLATE = """\
You are analyzing a unit test to produce a one-sentence semantic summary for retrieval.

Test class: {class_fqn}
Test method: {method_name}

Source:
```java
{source}
```

Output JSON only:
{{
  "summary": "<one sentence describing what behavior this test verifies, in plain English>",
  "concepts": ["<3-7 short keywords or feature names>"]
}}"""

# ---------------------------------------------------------------------------
# Phase 1 — Tier Classifier
# ---------------------------------------------------------------------------

TIER_CLASSIFIER_V1 = "TIER_CLASSIFIER_V1"

TIER_CLASSIFIER_V1_TEMPLATE = """\
Classify this test into a safety tier:
- Tier 1 (CRITICAL): security, authentication, authorization, payment, privacy, cryptography, data integrity
- Tier 2 (HIGH): persistence, transactions, concurrency, public API contracts
- Tier 3 (MEDIUM): business logic, services, controllers
- Tier 4 (LOW): utilities, formatting, helpers
- Tier 5 (TRIVIAL): getters/setters/toString/equals/hashCode, simple object construction

Test class: {class_fqn}
Summary: {summary}
Concepts: {concepts}

Output JSON only: {{"tier": <1-5>, "reason": "<brief>"}}"""

# ---------------------------------------------------------------------------
# Phase 2 — Impact Analyst
# ---------------------------------------------------------------------------

INTENT_V1 = "INTENT_V1"

INTENT_V1_TEMPLATE = """\
You are analyzing a Git diff to infer the developer's intent for test selection purposes.

Files changed: {file_list}
Changed methods: {method_list}

Diff:
```diff
{diff}
```

Output JSON only:
{{
  "intent_summary": "<2-3 sentences describing what behavior is changing and why>",
  "concepts": ["<3-10 keywords/feature names that should match relevant tests>"],
  "risk_areas": ["<security|persistence|api|ui|util|other>"]
}}"""

INTENT_MERGE_V1 = "INTENT_MERGE_V1"

INTENT_MERGE_V1_TEMPLATE = """\
Merge these per-file intent analyses into a single unified intent for the whole diff.

Per-file intents:
{partial_intents}

Output JSON only:
{{
  "intent_summary": "<2-3 sentences>",
  "concepts": ["<merged keyword list, deduplicated>"],
  "risk_areas": ["<security|persistence|api|ui|util|other>"]
}}"""
