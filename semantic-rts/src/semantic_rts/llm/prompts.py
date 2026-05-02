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

# V2: merged summarize + tier + SUT context + richer output schema
SUMMARIZER_V2 = "SUMMARIZER_V2"

SUMMARIZER_V2_TEMPLATE = """\
You are analyzing a unit test to produce a rich semantic description for a test selection system.
Your output will be embedded in a vector database to match code changes to relevant tests — be precise and specific.

Test class: {class_fqn}
Test method: {method_name}
{sut_context}
Test source:
```java
{source}
```

Output JSON only:
{{
  "summary": "<one sentence: what specific production behavior does this test verify>",
  "condition": "<the exact scenario or condition under test, e.g. 'null input', 'empty list', 'concurrent modification', 'boundary value'>",
  "tested_methods": ["<ClassName.methodName for each production method this test directly exercises>"],
  "concepts": ["<3-7 domain keywords relevant to this test>"],
  "tier": <integer 1-5: 1=security/auth/payment/crypto, 2=persistence/transactions/concurrency, 3=business-logic/services/API, 4=utilities/formatting/parsing, 5=trivial getters/setters/toString>
}}"""

# V3: adds topology_scope field; keep V2 intact so its cached responses remain valid
SUMMARIZER_V3 = "SUMMARIZER_V3"

SUMMARIZER_V3_TEMPLATE = """\
You are analyzing a unit test to produce a rich semantic description for a test selection system.
Your output will be embedded in a vector database to match code changes to relevant tests — be precise and specific.

Test class: {class_fqn}
Test method: {method_name}
{sut_context}
Test source:
```java
{source}
```

Output JSON only:
{{
  "summary": "<one sentence: what specific production behavior does this test verify>",
  "condition": "<the exact scenario or condition under test, e.g. 'null input', 'empty list', 'concurrent modification', 'boundary value'>",
  "tested_methods": ["<ClassName.methodName for each production method this test directly exercises>"],
  "concepts": ["<3-7 domain keywords relevant to this test>"],
  "tier": <integer 1-5: 1=security/auth/payment/crypto, 2=persistence/transactions/concurrency, 3=business-logic/services/API, 4=utilities/formatting/parsing, 5=trivial getters/setters/toString>,
  "topology_scope": "<unit|integration|system: unit=tests one class in isolation using mocks or minimal deps, integration=tests real interaction between 2+ classes with real dependencies, system=tests end-to-end flow spanning multiple subsystems>"
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
{commit_block}\
{sig_block}Diff:
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
