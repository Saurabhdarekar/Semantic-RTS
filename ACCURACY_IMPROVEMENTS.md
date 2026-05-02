# Semantic-RTS: Accuracy Improvement Implementation Plan

**Project:** CS 527 Group 7 — Semantic-Aware Regression Test Selection  
**Scope:** Improvements to recall and precision across all pipeline phases  
**Excluded:** B2 (co-failure correlation), D1 (bidirectional retrieval)  
**Already implemented — do not re-implement:** Safety Bridge, threshold-based pool fetching,
IDF method boost, SUT linking, disk-based LLM cache

Read the entire plan before writing any code. Implement in the order listed.

---

## Corrections Applied vs. Previous Draft

The following bugs were present in the previous draft and are fixed here:

1. `classify_change_type` used `for pf in ...` placeholder and referenced `pf.target_file` which does not exist on `FileChange` — fixed to use `parsed.raw_diff` directly
2. `_build_bm25` relied on `self._rows` which is always empty after `VectorStore.load()` because `save()` does not write rows to `index.meta.json` — fixed by accepting `tested_methods_map` as an explicit parameter
3. `retrieve()` accessed private `kb._test_ids` directly — fixed to use `kb.bm25_scores()` which returns a dict keyed by test_id
4. Topology multiplier in `ranker.py` called `.get("topology_scope", "unit")` on a `list[str]` from `tested_methods_map` — fixed by using a separate `topology_map: dict[str, str]`
5. `Reason` literal type not updated to include `"fixture_bypass"` — fixed
6. `AnalysisTrace` dataclass not updated for `low_confidence_retrieval` — fixed
7. `select()` new parameters not formally specified — fixed with full updated signature
8. `eval/runner.py` not listed as a file to update even though it needs to build new maps and pass new args — fixed
9. `SUMMARIZER_V2_TEMPLATE` referred to inconsistently (sometimes V2, sometimes V3) — standardised to V3 throughout with a new version constant
10. `micro_diff_max_lines: 0` ablation does not disable the bypass because the check uses `change_type == "micro_fix"` not the config value — fixed to use `micro_diff_bypass_enabled: false`
11. `import re, math` in sensitivity scorer imports `math` but never uses it — removed
12. `_package_proximity_score` could throw uncaught `StopIteration` when "java" is not in path — wrapped in try/except
13. Cross-encoder takes `list[SelectedTest]` but must run before `select()` on `list[Candidate]` — fixed type and placement to `retriever.py`
14. Config Pydantic models in `config.py` were not updated, only the YAML — full model updates now specified
15. Step 11 (config) was listed last but config fields must exist before any code that references them — moved to step 0
16. `cli.py` not listed as a file to update despite needing `--commit-message` option — added

---

## Overview of Changes by File

| File | Changes |
|---|---|
| `config.py` | New fields in `ImpactConfig`, `RetrievalConfig`, `SelectorConfig` Pydantic models |
| `config/default.yaml` | Default values for all new config fields |
| `kb/embed_format.py` | **NEW FILE** — shared canonical embedding format function |
| `kb/sensitivity.py` | **NEW FILE** — static test sensitivity scorer |
| `kb/test_parser.py` | Add `fixture_classes`, `sensitivity_score`, `topology_scope` to `TestMethod` |
| `kb/builder.py` | Call shared embed format; compute sensitivity; extract fixtures |
| `kb/summarizer.py` | Use SUMMARIZER_V3; parse `topology_scope` from response |
| `kb/vector_store.py` | Add `build_bm25(tested_methods_map)` and `bm25_scores()` methods |
| `llm/prompts.py` | Add `SUMMARIZER_V3` / `SUMMARIZER_V3_TEMPLATE`; update `INTENT_V1_TEMPLATE` |
| `impact/diff_parser.py` | Add `change_type` to `ParsedDiff`; implement `classify_change_type()` |
| `impact/intent_agent.py` | Micro-diff bypass; commit message injection; updated function signature |
| `impact/retriever.py` | Token overlap; package proximity; BM25 blend; negative pass; adaptive threshold; optional cross-encoder |
| `selector/ranker.py` | Updated `select()` signature; fixture bypass; topology multiplier; sensitivity multiplier; use effective threshold |
| `eval/runner.py` | Build `fixture_map`, `sensitivity_map`, `topology_map`; call `build_bm25`; pass new args to `select()` |
| `cli.py` | `--commit-message` option on `select`; pass `change_type` to `analyze_intent` |
| `config/ablations/` | New ablation YAML files for each new feature |

---

## Phase 0: Config First

All new config fields must exist before any other code is written.

### 0.1 Update Pydantic Models in `config.py`

```python
class ImpactConfig(BaseModel):
    skip_intent_agent: bool = False
    diff_max_tokens: int = 8000
    micro_diff_bypass_enabled: bool = True   # NEW
    micro_diff_max_lines: int = 5            # NEW


class RetrievalConfig(BaseModel):
    top_k: int = 30
    similarity_threshold: float = 0.55
    token_overlap_weight: float = 0.12       # NEW
    package_proximity_weight: float = 0.08   # NEW
    bm25_weight: float = 0.20               # NEW
    negative_pass_enabled: bool = True       # NEW
    negative_pass_penalty: float = 0.25     # NEW
    adaptive_threshold_enabled: bool = True  # NEW
    adaptive_threshold_min: float = 0.30     # NEW


class SelectorConfig(BaseModel):
    safety_bridge: SafetyBridgeConfig = Field(default_factory=SafetyBridgeConfig)
    precision_filter: PrecisionFilterConfig = Field(default_factory=PrecisionFilterConfig)
    max_selected: int = 100
    topology_multiplier: float = 1.20                                    # NEW
    topology_trigger_change_types: list[str] = Field(                   # NEW
        default_factory=lambda: ["api_change", "refactoring"]
    )
    topology_trigger_packages: list[str] = Field(                       # NEW
        default_factory=lambda: ["util", "common", "base", "core", "shared", "helper"]
    )
    sensitivity_multiplier_enabled: bool = True                          # NEW
    cross_encoder_enabled: bool = False                                  # NEW
    cross_encoder_top_n: int = 40                                        # NEW
```

### 0.2 Update `config/default.yaml`

```yaml
impact:
  skip_intent_agent: false
  diff_max_tokens: 8000
  micro_diff_bypass_enabled: true   # NEW
  micro_diff_max_lines: 5           # NEW

retrieval:
  top_k: 30
  similarity_threshold: 0.55
  token_overlap_weight: 0.12        # NEW
  package_proximity_weight: 0.08    # NEW
  bm25_weight: 0.20                 # NEW
  negative_pass_enabled: true       # NEW
  negative_pass_penalty: 0.25       # NEW
  adaptive_threshold_enabled: true  # NEW
  adaptive_threshold_min: 0.30      # NEW

selector:
  topology_multiplier: 1.20                                # NEW
  topology_trigger_change_types: [api_change, refactoring] # NEW
  topology_trigger_packages: [util, common, base, core, shared, helper] # NEW
  sensitivity_multiplier_enabled: true                     # NEW
  cross_encoder_enabled: false                             # NEW
  cross_encoder_top_n: 40                                  # NEW
  safety_bridge:
    always_include_tier_1: true
    tier_2_threshold: 0.40
  precision_filter:
    tier_5_min: 0.65
    tier_4_min: 0.50
  max_selected: 100
```

---

## Phase 1: Knowledge Base Construction Improvements

### 1.1 Shared Symmetric Embedding Format

**Problem:** `format_for_embedding()` in `builder.py` and `format_query()` in `retriever.py`
produce differently structured strings, misaligning query and document vectors in FAISS space.

**Action:** Create `kb/embed_format.py` as a new file. Both `builder.py` and `retriever.py`
must import from it — never format embedding strings inline.

```python
# kb/embed_format.py  (NEW FILE)

"""
Canonical embedding format shared between Phase 1 (document) and Phase 2 (query).

IMPORTANT: Any change to the output of format_for_embedding() invalidates all existing
FAISS indexes. If you modify this function, force a full KB rebuild with `srts build --force`.
"""


def format_for_embedding(
    summary: str,
    methods: list[str],
    concepts: list[str],
    risk_areas: list[str] | None = None,
    condition: str | None = None,
    class_simple: str | None = None,
) -> str:
    parts = [f"Behavior: {summary}."]
    if condition:
        parts.append(f"Condition: {condition}.")
    if methods:
        simple_names = [m.split(".")[-1] for m in methods if "." in m]
        if not simple_names:
            simple_names = list(methods)
        parts.append(f"Methods involved: {', '.join(simple_names)}.")
    if concepts:
        parts.append(f"Concepts: {', '.join(concepts)}.")
    if risk_areas:
        parts.append(f"Risk areas: {', '.join(risk_areas)}.")
    if class_simple:
        parts.append(f"Test class: {class_simple}.")
    return " ".join(parts)
```

In `builder.py`, replace the body of the existing `format_for_embedding(tm: TestMethod)`
function:

```python
from semantic_rts.kb.embed_format import format_for_embedding as _canonical_format

def format_for_embedding(tm: TestMethod) -> str:
    return _canonical_format(
        summary=tm.summary,
        methods=tm.tested_methods,
        concepts=tm.concepts,
        condition=tm.condition,
        class_simple=tm.class_simple,
    )
```

In `retriever.py`, replace `format_query()`:

```python
from semantic_rts.kb.embed_format import format_for_embedding as _canonical_format

def format_query(intent: "IntentResult", methods_changed: list[str] | None = None) -> str:
    return _canonical_format(
        summary=intent.intent_summary,
        methods=methods_changed or [],
        concepts=intent.concepts,
        risk_areas=intent.risk_areas,
        # Do not pass condition or class_simple for query vectors
    )
```

---

### 1.2 Fixture Class Extraction

**Problem:** JUnit `@Before`/`@BeforeClass`/`@Rule` methods instantiate shared objects that
all tests in the class depend on. If a changed class appears as a fixture, every test in that
class is affected regardless of embedding score. This is completely unhandled today.

**Files:** `kb/test_parser.py`

Add field to `TestMethod` dataclass:

```python
fixture_classes: list[str] = field(default_factory=list)
```

Add these at module level in `test_parser.py`:

```python
_SETUP_ANNOTATIONS = {
    "Before", "BeforeClass", "After", "AfterClass",
    "BeforeEach", "AfterEach", "BeforeAll", "AfterAll",
    "Rule", "ClassRule",
}
_NEW_INSTANCE_RE = re.compile(r'\bnew\s+([A-Z][a-zA-Z0-9_]*)\s*\(')


def _extract_fixture_classes(source_lines: list[str], type_decl) -> list[str]:
    """Return class simple names instantiated inside setup/teardown methods."""
    fixture_classes: list[str] = []
    for member in (type_decl.body or []):
        if not isinstance(member, javalang.tree.MethodDeclaration):
            continue
        ann_names = {a.name for a in (member.annotations or [])}
        if not (ann_names & _SETUP_ANNOTATIONS):
            continue
        start = (member.position.line - 1) if member.position else 0
        source = _extract_method_source(source_lines, start)
        for m in _NEW_INSTANCE_RE.finditer(source):
            cls = m.group(1)
            if cls not in fixture_classes:
                fixture_classes.append(cls)
    return fixture_classes
```

In `_parse_file()`, after building the list of `TestMethod` objects for a class, call:

```python
fixtures = _extract_fixture_classes(source_lines, type_decl)
for tm in methods:   # list of TestMethod objects for this class only
    tm.fixture_classes = fixtures
```

The field is persisted to `tests.jsonl` automatically via `asdict` (it is a dataclass field).
No changes to `builder.py` serialisation are needed.

---

### 1.3 Test Sensitivity Scoring

**Problem:** All tests are treated as equally valuable retrieval targets regardless of how
likely they are to fail due to a code change.

**Files:** `kb/test_parser.py`, `kb/builder.py`, new file `kb/sensitivity.py`

Add field to `TestMethod` dataclass:

```python
sensitivity_score: float = 0.5
```

Create `kb/sensitivity.py`:

```python
# kb/sensitivity.py  (NEW FILE)

import re

_MOCK_RE = re.compile(r'\b(mock|when|verify|stub|Mockito)\b', re.IGNORECASE)
_BOUNDARY_RE = re.compile(
    r'\b(0|1|-1|null|""|empty|boundary|limit|Integer\.MAX_VALUE|Integer\.MIN_VALUE)\b'
)
_EXCEPTION_RE = re.compile(r'assertThrows|expected\s*=|ExpectedException')


def compute_sensitivity(source: str) -> float:
    """
    Estimate how sensitive a test is to logic changes using static AST heuristics.
    Returns a float in [0.1, 1.0]. Higher = more likely to catch real regressions.
    """
    score = 0.0

    # More distinct assertion types = more sensitive
    distinct_asserts = len(set(re.findall(r'\bassert\w+', source)))
    score += min(distinct_asserts * 0.15, 0.45)

    # Mocks reduce sensitivity — test is isolated from real production behavior
    if _MOCK_RE.search(source):
        score -= 0.15

    # Boundary value testing increases sensitivity to off-by-one errors
    if _BOUNDARY_RE.search(source):
        score += 0.15

    # Exception assertion increases sensitivity to behavioral changes
    if _EXCEPTION_RE.search(source):
        score += 0.15

    # Longer tests tend to cover more behavior paths
    lines = sum(1 for line in source.splitlines() if line.strip())
    score += min(lines / 100.0, 0.15)

    return max(0.1, min(1.0, score))
```

In `builder.py`, import and call this immediately after `parse_test_methods()`, before the
LLM enrichment loop (zero API cost):

```python
from semantic_rts.kb.sensitivity import compute_sensitivity

# After: all_methods = parse_test_methods(files, project_path)
for tm in all_methods:
    tm.sensitivity_score = compute_sensitivity(tm.source)
```

---

### 1.4 Topology Scope Tagging

**Problem:** Changes to low-level utilities should surface integration and system-level tests,
not only unit tests. Without topology tagging, the ranker cannot distinguish test scope.

**Files:** `llm/prompts.py`, `kb/summarizer.py`, `kb/test_parser.py`

Add field to `TestMethod` dataclass:

```python
topology_scope: str = "unit"   # "unit" | "integration" | "system"
```

In `llm/prompts.py`, add a new version and template. Do **not** modify `SUMMARIZER_V2` or
`SUMMARIZER_V2_TEMPLATE` — keep them so existing cached responses remain valid.

```python
SUMMARIZER_V3 = "SUMMARIZER_V3"

SUMMARIZER_V3_TEMPLATE = """\
You are analyzing a unit test to produce a rich semantic description for a test selection system.

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
  "condition": "<exact scenario under test, e.g. 'null input', 'concurrent modification'>",
  "tested_methods": ["<ClassName.methodName for each production method directly exercised>"],
  "concepts": ["<3-7 domain keywords>"],
  "tier": <1-5: 1=security/auth/payment/crypto, 2=persistence/transactions/concurrency, 3=business-logic/api, 4=utilities/parsing, 5=trivial getters/setters>,
  "topology_scope": "<unit|integration|system: unit=tests one class in isolation using mocks or minimal deps, integration=tests real interaction between 2+ classes with real dependencies, system=tests end-to-end flow spanning multiple subsystems>"
}}"""
```

In `kb/summarizer.py`, switch to `SUMMARIZER_V3` / `SUMMARIZER_V3_TEMPLATE` and extract
the new field:

```python
from semantic_rts.llm.prompts import SUMMARIZER_V3, SUMMARIZER_V3_TEMPLATE

# In summarize_test(), use SUMMARIZER_V3_TEMPLATE for the prompt.
# Update the return tuple to include topology_scope:
# -> tuple[str, list[str], int, list[str], str, str]
#    (summary, concepts, tier, tested_methods, condition, topology_scope)

topology_raw = str(parsed.get("topology_scope", "unit")).strip().lower()
topology_scope = topology_raw if topology_raw in {"unit", "integration", "system"} else "unit"
```

Update `_enrich_llm()` in `builder.py` to unpack the new return value and assign
`tm.topology_scope`.

**Note:** This requires a KB rebuild. Existing KBs default to `topology_scope = "unit"` for
all tests, which is safe but removes the topology boost. Rebuild when time permits.

---

## Phase 2: Intent Analysis Improvements

### 2.1 Change Type Classifier

**Problem:** Every diff goes through identical intent processing regardless of whether it is
a 1-line operator fix, a signature change, or a large refactoring. These need different
retrieval strategies.

**Files:** `impact/diff_parser.py`

Add field to `ParsedDiff` dataclass:

```python
change_type: str = "general"
# Values: "micro_fix" | "api_change" | "refactoring" | "new_behavior" | "config_change" | "general"
```

Add at module level:

```python
# Matches Java method declaration lines in unified diff output
_SIG_CHANGE_RE = re.compile(
    r'^[+-]\s*(public|protected|private|static)\s+\w[\w<>\[\]]*\s+\w+\s*\(',
    re.MULTILINE,
)


def classify_change_type(parsed: "ParsedDiff") -> str:
    """Classify a parsed diff using rule-based heuristics. No LLM call."""
    java_changes = [fc for fc in parsed.file_changes if fc.path.endswith(".java")]
    if not java_changes:
        return "config_change"

    has_new_files = any(fc.is_new_file for fc in java_changes)
    has_sig_change = bool(_SIG_CHANGE_RE.search(parsed.raw_diff))

    raw_lines = parsed.raw_diff.splitlines()
    added = sum(1 for l in raw_lines if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in raw_lines if l.startswith("-") and not l.startswith("---"))
    total_delta = added + removed

    if has_new_files:
        return "new_behavior"
    if has_sig_change:
        return "api_change"
    if total_delta <= 5:
        return "micro_fix"
    if total_delta > 30 and not has_sig_change:
        return "refactoring"
    return "general"
```

At the end of `parse_unified_diff()`, after constructing the `ParsedDiff` object, add:

```python
result.change_type = classify_change_type(result)
return result
```

---

### 2.2 Micro-Diff Bypass

**Problem:** For tiny diffs (≤5 changed lines, no signature change), the LLM produces vague
summaries that embed poorly. Direct use of method/class name tokens is a better query.

**Files:** `impact/intent_agent.py`

Update `analyze_intent()` with two new optional parameters (existing call sites work
unchanged since both have defaults):

```python
def analyze_intent(
    diff_text: str,
    files_changed: list[str],
    methods_changed: list[str],
    client: "GeminiClient",
    config: "Config",
    project_root: "str | None" = None,
    change_type: str = "general",        # NEW — pass parsed.change_type
    commit_message: str | None = None,   # NEW — see 2.3
) -> IntentResult:

    # --- Micro-diff bypass (zero LLM calls) ---
    if config.impact.micro_diff_bypass_enabled and change_type == "micro_fix":
        from pathlib import Path as _Path
        method_names = [m.split(".")[-1] for m in methods_changed if "." in m]
        file_stems = [_Path(f).stem for f in files_changed]
        summary = f"Small logic change in {', '.join(file_stems) or 'unknown file'}."
        if method_names:
            summary += f" Affected methods: {', '.join(method_names)}."
        if commit_message:
            first_line = commit_message.splitlines()[0][:120]
            summary += f" Commit: {first_line}."
        return IntentResult(
            intent_summary=summary,
            concepts=method_names[:10],
            risk_areas=["other"],
            intent_failed=False,
        )

    # ... rest of existing function body unchanged ...
```

Update call site in `eval/runner.py` (`run_bug`):

```python
intent = analyze_intent(
    bug.diff,
    parsed.files_changed,
    methods,
    client,
    config,
    project_root=str(bug.fixed_dir),
    change_type=parsed.change_type,   # NEW
    commit_message=None,              # populate per 2.3
)
```

---

### 2.3 Commit Message Injection

**Problem:** Diffs contain the *what* but rarely the *why*. Commit messages / bug descriptions
dramatically improve intent summarisation for otherwise opaque constant-change diffs.

**Files:** `impact/intent_agent.py`, `llm/prompts.py`, `eval/runner.py`, `cli.py`

Update `INTENT_V1_TEMPLATE` in `prompts.py` to include an optional commit block. This is an
additive change — the template still works with an empty string for `commit_block`:

```python
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
```

In `_single_intent_call()` in `intent_agent.py`, add `commit_message` parameter and build
`commit_block = f"Commit message: {commit_message}\n\n" if commit_message else ""`. Pass
`commit_block` into the template format call.

In `eval/runner.py`, add a helper to fetch the Defects4J bug description once per bug and
cache it:

```python
_bug_desc_cache: dict[tuple, str | None] = {}

def _get_bug_description(project: str, bug_id: int) -> str | None:
    key = (project, bug_id)
    if key in _bug_desc_cache:
        return _bug_desc_cache[key]
    try:
        result = subprocess.run(
            ["defects4j", "info", "-p", project, "-b", str(bug_id)],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            lines = result.stdout.splitlines()
            for i, line in enumerate(lines):
                if "summary" in line.lower() and i + 1 < len(lines):
                    desc = lines[i + 1].strip() or None
                    _bug_desc_cache[key] = desc
                    return desc
    except Exception:
        pass
    _bug_desc_cache[key] = None
    return None
```

Pass the result as `commit_message` to `analyze_intent()`.

In `cli.py`, add to the `select` command:
```python
@click.option("--commit-message", default=None,
              help="Commit message or PR description to improve intent analysis.")
```

---

## Phase 3: Retrieval Improvements

### 3.1 CamelCase Token Overlap Scoring

**Problem:** `testValidatePassword` and `validatePassword` are semantically identical by name
but may land far apart in general-purpose embedding space. This is the most common category
of obvious retrieval miss.

**Files:** `impact/retriever.py`

Add at module level:

```python
import re
from pathlib import Path

_CAMEL_SPLIT_RE = re.compile(r'([A-Z]+)([A-Z][a-z])|([a-z\d])([A-Z])')


def _camel_tokens(name: str) -> set[str]:
    """Split camelCase/PascalCase into lowercase tokens. Drops single-char tokens."""
    spaced = _CAMEL_SPLIT_RE.sub(r'\1\3 \2\4', name)
    return {t.lower() for t in re.split(r'[\s_\-]+', spaced) if len(t) > 1}


def _token_overlap_score(
    test_id: str,
    changed_methods: list[str],
    changed_files: list[str],
) -> float:
    """Jaccard token overlap between test name tokens and changed entity tokens."""
    method_part = test_id.split("::")[-1] if "::" in test_id else test_id
    class_part = test_id.split("::")[0].split(".")[-1] if "::" in test_id else ""
    test_tokens = _camel_tokens(method_part) | _camel_tokens(class_part)
    if not test_tokens:
        return 0.0

    best = 0.0
    for method in changed_methods:
        simple = method.split(".")[-1] if "." in method else method
        change_tokens = _camel_tokens(simple)
        if not change_tokens:
            continue
        union = test_tokens | change_tokens
        best = max(best, len(test_tokens & change_tokens) / len(union))

    # File stem match at reduced weight (0.6x) — less precise than method name
    for f in changed_files:
        stem_tokens = _camel_tokens(Path(f).stem)
        if not stem_tokens:
            continue
        union = test_tokens | stem_tokens
        best = max(best, 0.6 * len(test_tokens & stem_tokens) / len(union))

    return best
```

In `retrieve()`, after IDF boost and before final sort:

```python
overlap_w = config.retrieval.token_overlap_weight   # 0.12
for i, (score, test_id) in enumerate(boosted):
    overlap = _token_overlap_score(test_id, methods_changed, files_changed)
    boosted[i] = (score + overlap * overlap_w, test_id)
```

---

### 3.2 Package Proximity Scoring

**Problem:** Tests co-located in the same Java package as changed code are structurally more
likely to be affected. This is a free structural signal not currently used.

**Files:** `impact/retriever.py`

```python
def _package_proximity_score(test_id: str, changed_files: list[str]) -> float:
    """Score by Java package prefix overlap between the test class and changed files."""
    if "::" not in test_id:
        return 0.0
    test_fqn = test_id.split("::")[0]
    test_pkg_parts = test_fqn.split(".")[:-1]   # drop class name

    best = 0.0
    for changed_file in changed_files:
        parts = Path(changed_file).with_suffix("").parts
        try:
            java_idx = next(i for i, p in enumerate(parts) if p == "java")
            changed_pkg_parts = list(parts[java_idx + 1 : -1])   # drop class name
        except StopIteration:
            continue   # path does not contain a "java" directory marker

        if not changed_pkg_parts:
            continue
        matching = sum(1 for a, b in zip(test_pkg_parts, changed_pkg_parts) if a == b)
        max_len = max(len(test_pkg_parts), len(changed_pkg_parts), 1)
        best = max(best, matching / max_len)

    return best
```

Apply in `retrieve()` after token overlap:

```python
prox_w = config.retrieval.package_proximity_weight   # 0.08
for i, (score, test_id) in enumerate(boosted):
    prox = _package_proximity_score(test_id, files_changed)
    boosted[i] = (boosted[i][0] + prox * prox_w, test_id)
```

---

### 3.3 BM25 Hybrid Search

**Problem:** Dense FAISS embeddings fail at exact keyword matching. A test that explicitly
lists the changed method in its `tested_methods` field should always score near the top.

**Files:** `kb/vector_store.py`, `impact/retriever.py`, `pyproject.toml`

Add `rank-bm25` to `pyproject.toml` dependencies.

In `vector_store.py`:

```python
class VectorStore:
    def __init__(self, dim: int = 768) -> None:
        # ... existing init ...
        self._bm25 = None   # built separately via build_bm25()

    def build_bm25(self, tested_methods_map: dict[str, list[str]]) -> None:
        """
        Build BM25 index over test identity tokens.

        Must be called explicitly after load() because tested_methods data
        is stored in tests.jsonl, not in index.meta.json.

        Args:
            tested_methods_map: {test_id: [ClassName.method, ...]} from tests.jsonl
        """
        from rank_bm25 import BM25Okapi
        import re as _re

        corpus = []
        for test_id in self._test_ids:
            tokens = _re.split(r'[.:_\s]+', test_id.lower())
            for method in tested_methods_map.get(test_id, []):
                tokens.extend(_re.split(r'[._\s]+', method.lower()))
            corpus.append(tokens)

        self._bm25 = BM25Okapi(corpus) if corpus else None

    def bm25_scores(self, query_tokens: list[str]) -> dict[str, float]:
        """
        Return {test_id: normalised_score} for all tests.
        Scores are in [0.0, 1.0]. Returns empty dict if BM25 not built.
        """
        if self._bm25 is None or not self._test_ids:
            return {}
        import numpy as np
        raw = self._bm25.get_scores(query_tokens)    # numpy array
        max_score = float(raw.max())
        if max_score <= 0:
            return {}
        return dict(zip(self._test_ids, (raw / max_score).tolist()))
```

In `eval/runner.py` (`run_eval`), after loading the store, call:

```python
store.build_bm25(tested_methods_map)   # tested_methods_map already loaded from jsonl
```

In `retrieve()`, after all other per-test boosts and before the final sort:

```python
# Build BM25 query from method and file tokens
bm25_query: list[str] = []
for m in methods_changed:
    bm25_query.extend(m.lower().split("."))
for f in files_changed:
    bm25_query.extend(re.split(r'[./\\]+', Path(f).stem.lower()))

bm25_w = config.retrieval.bm25_weight          # 0.20
bm25_map = kb.bm25_scores(bm25_query)          # dict[str, float]

for i, (score, test_id) in enumerate(boosted):
    bm25 = bm25_map.get(test_id, 0.0)
    boosted[i] = ((1.0 - bm25_w) * score + bm25_w * bm25, test_id)
```

---

### 3.4 Negative Reasoning Pass

**Problem:** The pipeline has no "definitely cannot fail" signal. Tier 4/5 tests with zero
naming, package, or method relationship to the change consume budget slots without helping
recall.

**Files:** `impact/retriever.py`

After all scoring (token overlap + package proximity + BM25 + IDF) and before the final
sort, apply penalty to clearly unrelated low-tier tests. Tier 1 and 2 are never touched.

```python
if config.retrieval.negative_pass_enabled:
    penalty = config.retrieval.negative_pass_penalty   # 0.25
    for i, (score, test_id) in enumerate(boosted):
        tier = kb.tier_for_id(test_id)
        if tier not in (4, 5):
            continue
        tok = _token_overlap_score(test_id, methods_changed, files_changed)
        bm25 = bm25_map.get(test_id, 0.0)
        pkg = _package_proximity_score(test_id, files_changed)
        if tok == 0.0 and bm25 < 0.05 and pkg < 0.10:
            boosted[i] = (score * penalty, test_id)
```

---

### 3.5 Uncertainty-Aware Adaptive Threshold

**Problem:** A fixed threshold silently returns near-zero useful candidates when the KB has
poor coverage. The pipeline appears to work while actually failing.

**Files:** `impact/retriever.py`

Update `AnalysisTrace` dataclass (in `retriever.py`):

```python
@dataclass
class AnalysisTrace:
    # ... all existing fields ...
    low_confidence_retrieval: bool = False   # NEW
    effective_threshold: float = 0.0        # NEW — actual threshold used this call
```

In `retrieve()`, after sorting `boosted`, compute the effective threshold:

```python
effective_threshold = config.retrieval.similarity_threshold

if config.retrieval.adaptive_threshold_enabled and boosted:
    top_score = boosted[0][0]
    if top_score < 0.45:
        effective_threshold = max(
            config.retrieval.adaptive_threshold_min,
            top_score - 0.10,
        )
        logger.warning(
            "Low-confidence retrieval (best score=%.3f). "
            "Widening threshold %.2f -> %.2f",
            top_score, config.retrieval.similarity_threshold, effective_threshold,
        )
        low_confidence = True
    else:
        low_confidence = False
else:
    low_confidence = False
```

Include in the returned trace and pass `effective_threshold` back to the caller so `select()`
can use it instead of re-reading from config.

---

### 3.6 Optional Cross-Encoder Reranking

This runs on `list[Candidate]` **inside `retrieve()`**, before returning to the caller.
It is a pre-selection filter, not a post-selection one. Add to `retriever.py`:

```python
def _cross_encode_filter(
    candidates: list["Candidate"],
    intent_summary: str,
    kb_summaries: dict[str, str],    # test_id -> summary string
    client: "GeminiClient",
    top_n: int,
) -> list["Candidate"]:
    """Binary YES/NO LLM pass over top_n candidates. Non-top-n are always kept."""
    top = candidates[:top_n]
    rest = candidates[top_n:]
    keep: list["Candidate"] = []
    for candidate in top:
        summary = kb_summaries.get(candidate.test_id, candidate.test_id)
        prompt = (
            f"Code change summary: {intent_summary[:400]}\n"
            f"Test description: {summary}\n"
            "Could this code change plausibly cause this test to fail? "
            "Answer YES or NO only."
        )
        try:
            result = client.chat(prompt, version_tag="CROSS_ENCODER_V1")
            if "YES" in result["text"].upper():
                keep.append(candidate)
        except Exception:
            keep.append(candidate)   # on error, keep the candidate (safe default)
    return keep + rest
```

At the end of `retrieve()`, before building the trace, call this only when enabled:

```python
if config.selector.cross_encoder_enabled and client is not None:
    candidates = _cross_encode_filter(
        candidates,
        intent.intent_summary,
        kb_summaries,          # dict loaded from tests.jsonl in run_eval
        client,
        config.selector.cross_encoder_top_n,
    )
```

The `client` and `kb_summaries` parameters must be threaded through to `retrieve()`.
Keep `cross_encoder_enabled: false` in `default.yaml`. Enable only for experiments.

---

## Phase 4: Selection & Reranking Improvements

### 4.1 Updated `select()` Signature

All new parameters are optional with safe defaults for backward compatibility.

```python
# selector/ranker.py

# Update Reason type first:
Reason = Literal["safety_bridge_t1", "safety_bridge_t2", "semantic_match", "fixture_bypass"]

def select(
    candidates: list["Candidate"],
    all_kb_tests: list[tuple[str, int]],
    config: "Config",
    ablation_flags: "AblationFlags | None" = None,
    # New optional parameters:
    fixture_map: "dict[str, list[str]] | None" = None,
    sensitivity_map: "dict[str, float] | None" = None,
    topology_map: "dict[str, str] | None" = None,
    change_type: str = "general",
    files_changed: "list[str] | None" = None,
    similarity_threshold: "float | None" = None,   # effective threshold from retriever
) -> SelectionTrace:
```

Add `from pathlib import Path` at the top of `ranker.py`.

---

### 4.2 Build Lookup Maps in `eval/runner.py`

In `run_eval()`, build all four maps in a single pass over `tests.jsonl`:

```python
tested_methods_map: dict[str, list[str]] = {}
fixture_map: dict[str, list[str]] = {}
sensitivity_map: dict[str, float] = {}
topology_map: dict[str, str] = {}
kb_summaries: dict[str, str] = {}          # for cross-encoder

jsonl_path = kb_path / "tests.jsonl"
if jsonl_path.exists():
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                tid = row["test_id"]
                tested_methods_map[tid] = row.get("tested_methods", [])
                fixture_map[tid] = row.get("fixture_classes", [])
                sensitivity_map[tid] = float(row.get("sensitivity_score", 0.5))
                topology_map[tid] = row.get("topology_scope", "unit")
                kb_summaries[tid] = row.get("summary", "")
            except Exception:
                pass
```

Pass `fixture_map`, `sensitivity_map`, `topology_map` to `run_bug()` and then to `select()`.

---

### 4.3 Fixture Class Bypass

In `select()`, after the Safety Bridge block and before the semantic relevance loop:

```python
# --- Fixture bypass ---
if fixture_map:
    changed_class_simples = {Path(f).stem for f in (files_changed or [])}
    for test_id, tier in all_kb_tests:
        if test_id in scored:
            continue
        if changed_class_simples & set(fixture_map.get(test_id, [])):
            scored[test_id] = (0.75, "fixture_bypass", tier)
```

In the Precision Filter drop loop, guard `fixture_bypass` same as `safety_bridge_t1`:

```python
if reason in ("safety_bridge_t1", "fixture_bypass"):
    continue   # never drop these entries
```

---

### 4.4 Effective Threshold from Retriever

Replace the hardcoded config read in the semantic relevance loop:

```python
# Use effective threshold from retriever if provided; fall back to config
sim_thresh = (
    similarity_threshold
    if similarity_threshold is not None
    else config.retrieval.similarity_threshold
)

for c in candidates:
    if c.score >= sim_thresh and c.test_id not in scored:
        scored[c.test_id] = (c.score, "semantic_match", c.tier)
```

---

### 4.5 Topology-Aware Score Multiplier

After all entries are in `scored` and before the Precision Filter:

```python
# --- Topology multiplier ---
trig_types = set(config.selector.topology_trigger_change_types)
trig_pkgs = config.selector.topology_trigger_packages

should_boost_topology = (
    change_type in trig_types
    or any(
        pkg in f.lower()
        for f in (files_changed or [])
        for pkg in trig_pkgs
    )
)

if should_boost_topology and topology_map:
    multiplier = config.selector.topology_multiplier
    for test_id in list(scored.keys()):
        old_score, reason, tier = scored[test_id]
        if reason == "safety_bridge_t1":
            continue
        if topology_map.get(test_id, "unit") in ("integration", "system"):
            scored[test_id] = (old_score * multiplier, reason, tier)
```

---

### 4.6 Sensitivity Score Multiplier

After the topology multiplier and before the Precision Filter:

```python
# --- Sensitivity multiplier ---
if config.selector.sensitivity_multiplier_enabled and sensitivity_map:
    sensitivity_weight = 0.08
    for test_id in list(scored.keys()):
        old_score, reason, tier = scored[test_id]
        if reason == "safety_bridge_t1":
            continue
        sensitivity = sensitivity_map.get(test_id, 0.5)
        # sensitivity=1.0 → +4% boost; sensitivity=0.1 → -3.2% penalty
        adjusted = old_score * (1.0 + sensitivity_weight * (sensitivity - 0.5))
        scored[test_id] = (adjusted, reason, tier)
```

---

## Phase 5: Ablation Config Files

Create in `config/ablations/`:

```yaml
# no_bm25.yaml
retrieval:
  bm25_weight: 0.0

# no_token_overlap.yaml
retrieval:
  token_overlap_weight: 0.0
  package_proximity_weight: 0.0

# no_micro_bypass.yaml
impact:
  micro_diff_bypass_enabled: false

# no_negative_pass.yaml
retrieval:
  negative_pass_enabled: false

# no_topology.yaml
selector:
  topology_multiplier: 1.0

# no_sensitivity.yaml
selector:
  sensitivity_multiplier_enabled: false

# with_cross_encoder.yaml
selector:
  cross_encoder_enabled: true
```

---

## Implementation Order

| Step | What | Notes |
|---|---|---|
| 0 | `config.py` + `config/default.yaml` | Must exist before any code references new fields |
| 1 | `kb/embed_format.py` (new file) | All embedding paths depend on this |
| 2 | `kb/builder.py` (use shared format) | No KB rebuild yet |
| 3 | `impact/diff_parser.py` (change type + classifier) | Required by step 4 |
| 4 | `impact/intent_agent.py` (bypass + commit msg) | Depends on step 3 |
| 5 | `impact/retriever.py` (token overlap + package proximity) | Standalone |
| 6 | `kb/vector_store.py` (BM25 methods) | New public methods on existing class |
| 7 | `impact/retriever.py` (BM25 + negative pass + adaptive threshold + cross-encoder stub) | Depends on step 6 |
| **8** | **Smoke test** — `scripts/run_eval_smoke.sh` (Chart bugs 1–5) | Verify recall ≥ 0.7 before KB changes |
| 9 | `kb/test_parser.py` (3 new fields) | KB data structure changes |
| 10 | `kb/sensitivity.py` (new file) + `kb/builder.py` (sensitivity + fixtures) | Depends on step 9 |
| 11 | `llm/prompts.py` + `kb/summarizer.py` (SUMMARIZER_V3 + topology) | Depends on step 9 |
| **12** | **Rebuild KB** for at least Chart project | Needed for steps 9–11 to take effect |
| 13 | `eval/runner.py` (4-map build, `build_bm25`, pass new args) | Depends on step 12 |
| 14 | `selector/ranker.py` (full updates) | Depends on step 13 |
| 15 | `cli.py` (commit-message option, change_type threading) | Wires everything for interactive use |
| 16 | `config/ablations/*.yaml` | After all features confirmed working |

---

## Testing Checklist

**Steps 0–1:**
- `test_embed_format_symmetric` — builder and retriever paths with same inputs produce identical strings
- `test_embed_format_no_optionals` — works with only `summary`, `methods`, `concepts`

**Step 3:**
- `test_change_type_micro_fix` — ≤5 changed lines, no sig → `"micro_fix"`
- `test_change_type_api_change` — modified method signature → `"api_change"`
- `test_change_type_new_behavior` — new file added → `"new_behavior"`
- `test_change_type_config` — no Java files changed → `"config_change"`
- `test_change_type_general` — 15-line change, no sig → `"general"`

**Step 4:**
- `test_micro_bypass_skips_llm` — `change_type="micro_fix"`, mock client, assert zero `client.chat` calls
- `test_micro_bypass_contains_method_name` — returned summary contains the changed method name
- `test_general_change_calls_llm` — `change_type="general"` still calls the LLM

**Step 5:**
- `test_token_overlap_exact` — `testValidatePassword` vs `validatePassword` → ≥ 0.8
- `test_token_overlap_zero` — `testDatabaseConnection` vs `validatePassword` → 0.0
- `test_package_proximity_same_package` — test and file in same package → ≈ 1.0
- `test_package_proximity_unrelated` — unrelated packages → 0.0

**Step 6–7:**
- `test_bm25_build_empty_map` — `build_bm25({})` completes without error
- `test_bm25_method_token_scores_high` — test whose ID contains query token scores higher than unrelated test
- `test_bm25_returns_empty_when_not_built` — `bm25_scores(...)` returns `{}` before `build_bm25` called
- `test_negative_pass_penalises_tier5` — Tier 5 test with no overlap gets score × 0.25
- `test_negative_pass_ignores_tier1` — Tier 1 test never penalised
- `test_adaptive_threshold_widens` — top score 0.35 → effective threshold ≤ 0.30

**Steps 9–11:**
- `test_fixture_extraction_setup_method` — `@Before` method with `new Auth()` → `fixture_classes=["Auth"]`
- `test_fixture_extraction_no_setup` — class with no setup → `fixture_classes=[]`
- `test_sensitivity_mocks_lower_score` — Mockito usage → `sensitivity_score < 0.5`
- `test_sensitivity_boundary_raises_score` — null/0 checks → score boost applied
- `test_topology_scope_integration_parsed` — SUMMARIZER_V3 JSON with `"integration"` → field set correctly

**Steps 13–14:**
- `test_fixture_bypass_always_included` — test with matching fixture class included even at FAISS score 0.0
- `test_fixture_bypass_survives_precision_filter` — fixture_bypass entry never dropped
- `test_topology_multiplier_applies_on_api_change` — integration-scoped test boosted × 1.2 for `api_change` diff
- `test_sensitivity_multiplier_boosts_high_sensitivity` — `sensitivity=1.0` → score × 1.04
- `test_effective_threshold_used_over_config` — passing `similarity_threshold=0.30` includes tests at 0.32
