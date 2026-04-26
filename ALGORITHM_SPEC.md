# Semantic-Aware RTS — Algorithm Specification

**Companion to:** `IMPLEMENTATION_PLAN.md`
**Audience:** Claude Code (precise enough to implement from)
**Status:** Draft — open for refinement. Each section ends with **🔧 Open design questions** for discussion.

---

## How to read this doc

Each phase is specified as:
- **I/O contract** — types of inputs, types of outputs
- **Steps** — numbered, each step is implementable as one function
- **Pseudocode** — language-agnostic; map to Python module paths from §2 of the plan
- **Edge cases** — things that will go wrong if not handled
- **Determinism notes** — what makes the step reproducible
- **🔧 Open design questions** — for us to discuss and lock down

Cross-cutting concerns (caching, rate-limiting, errors) are in §4 because they apply to every phase.

---

## 1. Phase 1 — Knowledge Base Construction

### 1.1 I/O contract

```
INPUT
  project_path: str          # path to a checked-out Defects4J project (fixed version)
  project_name: str          # e.g. "Chart"
  config: Config

OUTPUT (written to data/kb/<project_name>/)
  tests.jsonl                # one row per test method
  index.faiss                # FAISS IndexFlatIP, normalized vectors
  index.meta.json            # row index → test_id mapping + build metadata

tests.jsonl row schema:
  {
    "test_id":     str,      # "org.jfree.chart.ChartTest::testRender"
    "class_fqn":   str,
    "class_simple": str,
    "method":      str,
    "file_path":   str,      # relative to project root
    "junit":       "4" | "5",
    "source":      str,      # raw method body, may be truncated
    "source_hash": str,      # sha1 of original source (for cache key)
    "summary":     str,      # 1 sentence, from LLM
    "concepts":    list[str],
    "tier":        int,      # 1..5
    "tier_source": "rule" | "llm",
    "embedding":   list[float]  # 768-d, normalized
  }
```

### 1.2 Steps

```
ALGORITHM BuildKB(project_path, project_name, config)
  1.  test_files     ← discover_test_files(project_path)
  2.  test_methods   ← parse_test_methods(test_files)        # may emit "skipped" log entries
  3.  for each tm in test_methods:
        a. tm.source ← extract_source(tm, max_tokens=1500)
        b. cache_key ← sha1(tm.source_hash + SUMMARIZER_PROMPT_VERSION)
        c. {tm.summary, tm.concepts} ← cached_or_call(LLM_summarize, tm, key=cache_key)
        d. tm.tier, tm.tier_source ← classify_tier(tm, config)
        e. tm.embedding ← embed(format_for_embedding(tm))     # see §1.3
  4.  write_jsonl(tests.jsonl, test_methods)
  5.  index ← FAISS.IndexFlatIP(dim=768)
  6.  index.add(np.stack([normalize(t.embedding) for t in test_methods]))
  7.  index.save(); write_meta(...)
END
```

### 1.3 Sub-algorithms

#### 1.3.1 `discover_test_files(project_path)`
Walk the project's source tree and return all `.java` files under any directory matching `**/test/**` or `**/tests/**`. Skip files in `target/`, `build/`, `out/`.

#### 1.3.2 `parse_test_methods(files)` — **NEEDS CARE**
Use a Java parser (`javalang` library). For each `.java` file:
- Walk the AST. For every `MethodDeclaration`:
  - **JUnit 4 detection:** has `@Test` annotation from `org.junit` import (or unqualified `@Test` if `org.junit.Test` is imported)
  - **JUnit 5 detection:** has `@Test`, `@ParameterizedTest`, or `@RepeatedTest` from `org.junit.jupiter.api` import
  - Skip if `@Disabled` or `@Ignore` is present
  - Skip if method is `private` or `static` (rare but invalid for tests)
- Build `test_id = <package>.<class>::<method>` (use `::` to disambiguate from method-call syntax)
- For nested test classes (`@Nested`), build hierarchical class FQN: `Outer$Inner`

**Edge cases:**
- Parameterized tests in JUnit 5 — record once; `@MethodSource` providers are not separate test methods
- Abstract test classes — skip (concrete subclasses will be parsed)
- Tests inheriting from a parent test class without overriding — currently won't be detected; **log and accept this gap in v1**

#### 1.3.3 `extract_source(test_method, max_tokens=1500)`
Concatenate, in order:
1. Class-level Javadoc (if present)
2. Class name + class-level annotations
3. Method-level Javadoc (if present)
4. Method body, including `@DisplayName` / `@ParameterizedTest` annotations

Truncate from the **end** of the method body if total exceeds budget. Use a rough 4-chars-per-token estimate; we don't need a real tokenizer here.

Hash before truncation. The hash is over the canonical source (we want cache hits even if we change `max_tokens` later — so hash the untruncated source).

#### 1.3.4 `LLM_summarize(test_method)` → `{summary, concepts}`
- Prompt: `SUMMARIZER_V1` from §15.1 of the plan
- `temperature=0`, `max_output_tokens=200`
- Parse output as JSON; on parse failure, retry once with a "respond in valid JSON only" reminder
- On second failure, fall back to: `summary = "Tests {class_simple}.{method}"`, `concepts = []`, log warning

#### 1.3.5 `classify_tier(test_method, config)` → `(tier, source)`
Two-stage classifier:

```
def classify_tier(tm, config):
    # Stage 1: rule pass
    haystack = (tm.class_simple + " " +
                tm.method + " " +
                tm.summary + " " +
                " ".join(tm.concepts)).lower()

    for tier_num in [1, 2, 3, 4, 5]:
        keywords = config.kb.tier_keywords[f"tier_{tier_num}"]
        if any(kw in haystack for kw in keywords):
            return (tier_num, "rule")

    # Stage 2: LLM fallback
    cache_key = sha1(tm.summary + tm.class_simple + TIER_PROMPT_VERSION)
    result = cached_or_call(LLM_classify_tier, tm, key=cache_key)
    return (result.tier, "llm")
```

**Rule precedence:** Tier 1 wins over Tier 2 wins over … Tier 5. So a test that mentions both "auth" (tier 1) and "format" (tier 4) is tier 1.

**LLM fallback default:** if LLM returns invalid JSON twice, default to Tier 3 (medium). Log it.

#### 1.3.6 `format_for_embedding(test_method)` → str
```
"summary: {summary} | tier: {tier} | class: {class_simple} | concepts: {concepts_joined}"
```
Why this shape: keeps the embedded text compact (~30-60 tokens), gives the embedder consistent slots, and lets us experiment with what to include without changing the embedder.

#### 1.3.7 `embed(text)` → list[float]
Call `text-embedding-005` via Gemini. Returns 768-d vector. **Normalize to unit length** before adding to FAISS (so `IndexFlatIP` computes cosine similarity).

### 1.4 Determinism

- All LLM calls at `temperature=0`
- Pinned `chat_model` and `embedding_model` versions in config
- All LLM responses cached on disk by `sha1(input + prompt_version)`
- File walk order sorted alphabetically (no `os.walk` nondeterminism)
- FAISS save → byte-identical files for the same input

### 1.5 Resumability

- Read existing `tests.jsonl` if present
- For each `test_method`, look up by `test_id`. If `source_hash` matches, skip the LLM calls and reuse the row
- This means: re-running `srts build` after a partial failure picks up where it left off
- Force rebuild via `--force` flag

### 🔧 Open design questions for Phase 1

1. **Inheritance gap.** Tests defined in abstract parents won't be discovered. Acceptable for v1, or do we need to expand?
2. **Test source budget.** 1,500 tokens — is that enough for long integration tests? Or should we bump to 2,500 and accept higher token cost?
3. **What goes into the embedding text?** Right now: `summary | tier | class | concepts`. Alternatives: include `method_name` (helps with name-based matching), or drop `tier` (avoids tier influencing semantic similarity). Worth A/B-testing in M11 sensitivity sweep.
4. **Tier keywords are English-only.** Defects4J is all English, so this is fine — but should we add common Java naming patterns (e.g., `*Test`, `*IT`, `*Spec`) to the rule layer?
5. **Source hashing scope.** Hash the method only, or include class-level Javadoc + imports? Including them means a class-level rename invalidates every test's cache; not including them means a Javadoc edit doesn't refresh the summary. Plan default: hash method body only.

---

## 2. Phase 2 — Impact Analysis & Retrieval

### 2.1 I/O contract

```
INPUT
  diff: str                  # unified diff text
  kb: KnowledgeBase          # loaded from data/kb/<project>/
  config: Config

OUTPUT
  candidates: list[Candidate]
  Candidate {
    test_id:    str
    score:      float        # cosine sim, [-1, 1]; will be in [0, 1] in practice
    tier:       int
    rank:       int          # 0-indexed position in top-K
  }
  trace: AnalysisTrace       # for debugging + report's worked example
  AnalysisTrace {
    diff_hash:        str
    files_changed:    list[str]
    methods_changed:  list[str]
    intent_summary:   str
    intent_concepts:  list[str]
    risk_areas:       list[str]
    query_embedding:  list[float]
    top_k:            int
    raw_results:      list[Candidate]
  }
```

### 2.2 Steps

```
ALGORITHM AnalyzeImpact(diff, kb, config)
  1.  parsed   ← parse_unified_diff(diff)                # using `unidiff`
  2.  files    ← parsed.modified_files
  3.  methods  ← extract_changed_methods(parsed)         # AST-walk on changed Java files
  4.  cache_key ← sha1(diff + INTENT_PROMPT_VERSION)
  5.  intent   ← cached_or_call(LLM_intent, diff, files, methods, key=cache_key)
  6.  query_text  ← format_query(intent)
  7.  query_emb   ← normalize(embed(query_text))
  8.  scores, ids ← kb.index.search(query_emb, k=config.retrieval.top_k)
  9.  candidates  ← [Candidate(kb.test_id_at(i), s, kb.tier_at(i), rank)
                    for rank, (i, s) in enumerate(zip(ids, scores))]
  10. trace ← build_trace(...)
  11. return candidates, trace
END
```

### 2.3 Sub-algorithms

#### 2.3.1 `parse_unified_diff(diff)` → ParsedDiff
- Use `unidiff.PatchSet(diff)`.
- Filter to `.java` files only (skip docs, configs, build files — they don't have tests in our KB).
- For each file, record: path before, path after (renames), list of hunk ranges.

#### 2.3.2 `extract_changed_methods(parsed)` → list[str] — **NEEDS CARE**
For each modified `.java` file in the diff:
1. Read the **post-change** version of the file from disk (the diff is applied; we work on the result).
2. Parse with `javalang`. Build a list of `(method_name, start_line, end_line)`.
3. For each hunk in the diff, find the method whose `[start_line, end_line]` contains the hunk's modified line range. Report that method.
4. Deduplicate. Output: list of `ClassName.methodName` strings.

**Edge cases:**
- Hunk spans multiple methods → report all of them
- Hunk is between methods (e.g. import added at top) → report `<class-level>` placeholder
- File is new (added in this diff) → report all methods in the new file
- File is deleted → report all methods that were in the pre-change version (load from `git show`)
- Parse failure → log, fall back to "could not parse, file: <path>"; intent agent still gets the raw diff

#### 2.3.3 `LLM_intent(diff, files, methods)` → IntentResult
- Prompt: `INTENT_V1` from §15.3 of the plan
- `temperature=0`, `max_output_tokens=400`
- If diff > 8K tokens, **chunk by file**:
  - One LLM call per file: `LLM_intent_per_file(file_path, file_hunks)` → partial intent
  - One LLM call to merge: `LLM_intent_merge(partial_intents)` → final intent
  - Cache per-file intents and merged intent separately
- On JSON parse failure, retry once. On second failure: `intent_summary = "Changes to: " + ", ".join(files)`, `concepts = methods`, `risk_areas = ["other"]`.

#### 2.3.4 `format_query(intent)` → str
```
"intent: {intent_summary} | concepts: {concepts_joined} | risk: {risk_areas_joined}"
```
Mirrors the embedding format from Phase 1 — keeps query and document representations compatible.

### 2.4 Determinism

- `temperature=0`; same diff → same intent
- Cache keyed on `sha1(diff + INTENT_PROMPT_VERSION)`
- FAISS top-K is deterministic on identical query + index

### 2.5 What gets returned even when things go wrong

- If LLM intent fails completely: still produce candidates by embedding the raw diff (truncated to first 4K chars). Mark `trace.intent_failed = True`. The Selector will still run.
- If diff parsing fails: report empty `methods`, set `intent_summary = "unknown change"`. Phase 3's Safety Bridge still includes Tier-1 tests, so we don't return an empty selection.

### 🔧 Open design questions for Phase 2

1. **Pre-change vs post-change parsing.** I'm parsing the post-change file. For deletions, I fall back to pre-change. Should we *always* parse both and union the methods? This would catch cases where a method is moved.
2. **`top_k = 30` enough?** For a project with 700 tests, that's 4%. We might miss a relevant test that's at rank 31. Two mitigations: (a) bump to 50; (b) use a similarity threshold instead of fixed K. Plan currently does both — Phase 3 filters at threshold 0.55. Worth confirming.
3. **Diff chunking threshold.** 8K tokens — is that too aggressive? Gemini Flash-Lite's context is 1M tokens; we could send much bigger diffs. But longer diffs mean more LLM noise in the intent. Sensitivity test in M11.
4. **Risk areas free-text vs enum.** The prompt asks for `["security|persistence|api|ui|util|other"]`. Should we let it be free-form (and embed those words) or stick to a closed enum? Closed enum is easier to test but loses signal.
5. **Should the query format include the changed-method names verbatim?** Right now we only embed the LLM's summary. Adding raw method names might help string-match on test names. Worth A/B-testing.

---

## 3. Phase 3 — Risk-Aware Selection

### 3.1 I/O contract

```
INPUT
  candidates: list[Candidate]   # from Phase 2
  kb: KnowledgeBase
  config: Config
  ablation_flags: AblationFlags # for ablation runs

OUTPUT
  selected: list[SelectedTest]
  SelectedTest {
    test_id:    str
    score:      float           # final ranking score
    tier:       int
    reason:     str             # "safety_bridge_t1" | "semantic_match" | "safety_bridge_t2"
  }
  trace: SelectionTrace         # per-test {kept|dropped, reason}

AblationFlags {
  safety_bridge_enabled: bool   # default True
  precision_filter_enabled: bool # default True
  tiers_enabled: bool            # default True; if False, all tests are tier 3
}
```

### 3.2 Steps

```
ALGORITHM Select(candidates, kb, config, ablation_flags)
  1.  scored ← {}                                       # test_id → (score, reason)
  2.  IF ablation_flags.safety_bridge_enabled:
        # Tier 1: always include
        for test in kb.all_tests():
          if effective_tier(test, ablation_flags) == 1:
            scored[test.id] = (1.0, "safety_bridge_t1")
        # Tier 2: include if score above threshold
        for c in candidates:
          if effective_tier(c, ablation_flags) == 2:
            if c.score >= config.selector.safety_bridge.tier_2_threshold:
              scored[c.test_id] = (c.score, "safety_bridge_t2")

  3.  # Semantic relevance: include any candidate above similarity threshold
      for c in candidates:
        if c.score >= config.retrieval.similarity_threshold:
          if c.test_id not in scored:
            scored[c.test_id] = (c.score, "semantic_match")

  4.  IF ablation_flags.precision_filter_enabled:
        scored = {
          tid: (s, r) for tid, (s, r) in scored.items()
          if not should_drop(kb.test(tid), s, ablation_flags, config)
        }

  5.  # Cap by max_selected, keep highest-scored
      sorted_items ← sort_by_score_desc(scored.items())
      capped ← sorted_items[:config.selector.max_selected]

  6.  selected ← [SelectedTest(tid, s, kb.tier(tid), r) for tid, (s, r) in capped]
  7.  trace ← build_selection_trace(candidates, scored, capped)
  8.  return selected, trace
END

FUNCTION effective_tier(test, ablation_flags):
  return 3 if not ablation_flags.tiers_enabled else test.tier

FUNCTION should_drop(test, score, ablation_flags, config):
  tier = effective_tier(test, ablation_flags)
  return (tier == 5 and score < config.selector.precision_filter.tier_5_min) or
         (tier == 4 and score < config.selector.precision_filter.tier_4_min)
```

### 3.3 Reason codes (in priority order)

When a test is included, `reason` is set to the **first** reason that applied:
1. `safety_bridge_t1` — kept because tier 1 + Safety Bridge enabled
2. `safety_bridge_t2` — kept because tier 2 + score above tier-2 threshold
3. `semantic_match` — kept because score above similarity threshold

When a test is dropped from `scored` by the Precision Filter, the trace records:
- `dropped_precision_t5` — tier 5 with score below threshold
- `dropped_precision_t4` — tier 4 with score below threshold

When a test is in `scored` but dropped by the cap:
- `dropped_cap` — beyond `max_selected`

This trace is what powers Figure 7 (tier distribution) and Figure 8 (failure cases) in the report.

### 3.4 Determinism

Pure Python, no I/O, no LLM. Deterministic given identical inputs. **Sort by score, then by `test_id` lexicographic** as a tiebreaker — this guarantees reproducible ordering when scores are equal.

### 3.5 Edge cases

- **Empty candidates list.** Safety Bridge still includes Tier-1 tests. Final selection is non-empty as long as the project has any Tier-1 test.
- **No Tier-1 tests in project.** Possible (e.g., Chart has no security tests). Selection falls back to semantic matches only. Log a warning at KB build time so the team knows the Safety Bridge is inactive for that project.
- **All candidates score below similarity threshold.** Selection is empty (modulo Safety Bridge). Log warning. The user can lower threshold via config to investigate.
- **`max_selected` < `|tier-1 tests|`.** Safety Bridge would be capped! Fix: when capping, **never drop a `safety_bridge_t1` entry**. Apply the cap only to `semantic_match` and `safety_bridge_t2` entries. This is a hard invariant — unit test it.

### 🔧 Open design questions for Phase 3

1. **Tier-1 inclusion is "always."** This means every commit triggers every Tier-1 test, regardless of relevance. If there are 50 Tier-1 tests in a project, that's a 50-test floor on the selection size. Acceptable? Alternative: include Tier-1 tests only when the diff's `risk_areas` contains a critical area (security/persistence). Tradeoff: simpler-but-conservative vs smarter-but-more-failure-modes.
2. **Score combination.** Right now Safety Bridge and Precision Filter are gates, not weights. We could instead compute a single weighted score: `final_score = α × semantic_score + β × tier_priority_bonus` and rank-cap on that. Cleaner but harder to explain. Plan keeps the gate model — easier to ablate, clearer in the report.
3. **Cap vs no cap.** `max_selected = 100` is somewhat arbitrary. For a 700-test project, 100 is ~14%. For a 7,000-test project, it's 1.4%. Maybe make this a **fraction of total** (e.g., 15%) rather than absolute? Or remove the cap entirely and trust the thresholds.
4. **Precision filter on Tier 4 vs Tier 5.** Currently we drop Tier 5 below 0.65 and Tier 4 below 0.50. These thresholds are guesses. Sensitivity sweep in M11 should validate them.
5. **Should `safety_bridge_t2` count toward the cap?** Right now yes. If we want to guarantee critical tests aren't dropped, the answer is no for both Tier 1 and Tier 2. Need to decide.

---

## 4. Cross-Cutting Concerns

### 4.1 Caching

**One disk cache for all LLM calls.**

```
data/cache/
  llm/
    <sha1>.json     # response cached by sha1(prompt_text + model + version_tag)
  intents/
    <sha1>.json     # diff intent results
  embeddings/
    <sha1>.json     # embedding vectors (8-byte float32 array → JSON, list[float])
```

**Cache key construction:**
```
def cache_key(prompt_text, model, version_tag):
    h = hashlib.sha1()
    h.update(prompt_text.encode("utf-8"))
    h.update(b"\x00")
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(version_tag.encode("utf-8"))
    return h.hexdigest()
```

**Why include the model name in the key:** so switching from `gemini-2.5-flash-lite` to `gemini-2.5-pro` invalidates the cache automatically.

**Why include a `version_tag`:** the prompt template name + version (e.g. `"SUMMARIZER_V1"`). Bumping the prompt to V2 invalidates V1 cache automatically.

**Cache lookup contract:**
```
def cached_or_call(fn, *args, key, force=False):
    if not force and cache_exists(key):
        return cache_load(key)
    result = fn(*args)
    cache_save(key, result)
    return result
```

**Replay mode:** an env var `SRTS_CACHE_ONLY=1` makes any cache miss raise an error instead of hitting the API. Use this in CI and in the eval-replay flow — guarantees no accidental network calls.

**Committing the cache:** the cache for the official eval set should be committed to the repo (or attached as a release artifact). Total size estimate: ~700 tests/project × 17 projects × ~2KB/response ≈ 24 MB. Manageable.

### 4.2 Rate limiting

**Two token-bucket limiters per provider:** one for RPM (requests per minute), one for RPD (requests per day).

```
class TokenBucket:
    def __init__(self, rate, per_seconds):
        self.tokens = rate
        self.rate = rate
        self.per_seconds = per_seconds
        self.last_refill = time.monotonic()

    def acquire(self, n=1):
        # block until n tokens available
        while True:
            self._refill()
            if self.tokens >= n:
                self.tokens -= n
                return
            sleep_for = (n - self.tokens) * (self.per_seconds / self.rate)
            time.sleep(min(sleep_for, 1.0))

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.rate, self.tokens + elapsed * (self.rate / self.per_seconds))
        self.last_refill = now
```

**Concurrency:** use a `ThreadPoolExecutor(max_workers=5)` for LLM calls. The bucket is shared across workers — each worker acquires before calling.

**Limits (Gemini free tier, with safety margin):**
- RPM: 14 (limit is 15)
- RPD: 1,400 (limit is 1,500)

When the daily limit is hit, stop the current run and write a checkpoint. Resume next day with `--resume`.

### 4.3 Error handling

**Three error categories, three policies.**

| Category | Examples | Policy |
|---|---|---|
| **Transient** | Network timeout, 429 rate limit, 5xx | Exponential backoff, retry 3 times (1s, 2s, 4s), then escalate to category 3 |
| **Recoverable** | LLM returns invalid JSON, embedding returns wrong dim | Retry once with stricter prompt; if still bad, fall back (see per-step "what gets returned even when things go wrong") |
| **Fatal** | API key invalid, daily quota exhausted, out of disk | Write checkpoint, log structured error, exit non-zero |

**Retry decorator:**
```python
@retry_with_backoff(max_retries=3, exceptions=(TimeoutError, RateLimitError, ServerError))
def llm_call(prompt): ...
```

**Logging:** all LLM calls go through one wrapper that logs `{method, latency_ms, input_tokens, output_tokens, cached, error}` to `data/logs/llm_calls.jsonl`. This is what feeds the cost-vs-benefit figure in the report.

### 4.4 Configuration

Single `Config` object loaded from `config/default.yaml`, overridable per-run via:
- `SRTS_CONFIG=config/ablations/no_bridge.yaml` env var
- `--config` CLI flag
- `--set llm.chat_model=gemini-2.5-pro` for one-off overrides

Ablation configs live in `config/ablations/`:
- `no_bridge.yaml` — sets `selector.safety_bridge.always_include_tier_1: false`, `tier_2_threshold: 1.0` (effectively disabled)
- `no_llm.yaml` — sets a flag the Phase 2 code reads to skip the intent agent
- `no_tier.yaml` — sets `tier_keywords` to all empty + a flag the selector reads to use tier 3 for all tests

Each ablation file is a delta on top of `default.yaml`, merged via deep-merge.

### 4.5 Logging & observability

- **Structured logs:** JSONL to stdout + `data/logs/`.
- **Per-bug eval row** appended to `data/eval/results.csv` immediately after each bug completes — so a crash mid-run doesn't lose progress.
- **Progress bar:** `tqdm` over bugs, with running cost estimate displayed.

### 🔧 Open design questions for cross-cutting

1. **Cache invalidation when only the embedding format changes.** Right now the embedding cache is keyed on `sha1(text)`. If we change `format_for_embedding(...)`, the text changes, so the cache invalidates correctly. But if we change *which* embedding model we use, do we want fully separate caches? (Plan keys on `text + model`, so yes — automatic.) Confirming this is the right behavior.
2. **Concurrency level.** 5 workers with a 14 RPM bucket means most workers block most of the time. Is concurrency even helping here? Maybe drop to 1 worker and simplify the locking model.
3. **`SRTS_CACHE_ONLY` for the report run.** Recommend making this the **default** when running `scripts/run_eval_full.sh` after an initial pass — guarantees the report is rendered from cache and any "live" cost is a bug.
4. **Logging volume.** Per-LLM-call log entries for 700 tests × 17 projects = ~12K rows. Fine. Per-test selection traces × bugs can blow up — 700 candidates × 835 bugs × 6 methods ≈ 3.5M rows. Plan default: write traces to per-bug JSON files, not the central log.

---

## 5. Discussion Hooks

Things I think we should talk through before Claude Code starts coding. These are the places where a small design decision compounds across the whole project.

1. **Tier-1 Safety Bridge always-on vs conditional.** §3 Q1 above. This is the single biggest design lever — it's most of the recall in our results, and it's also the one easiest for a reviewer to question ("you're just running every critical test"). Worth resolving before M3.

2. **What `format_for_embedding` includes.** §1 Q3 above. This affects every retrieval result. Cheap to change; expensive to leave undecided.

3. **Per-project KB only.** Plan locks this in — but if we ever want to do "run on a fresh project without rebuilding," we'd want a cross-project KB. Out of scope for v1; flag in §12 threats.

4. **`top_k` vs threshold as the primary retrieval gate.** Right now both are active. Keeping both works but means three knobs (`top_k`, `similarity_threshold`, plus tier thresholds) interact. Simpler model: take top-K only, drop the threshold. Trades robustness for clarity.

5. **Static-RTS baseline.** §12 of the plan documents the file-reachability fallback. A simpler one is "if the diff touches `src/main/java/foo/Bar.java`, run any test in `src/test/java/foo/BarTest.java` (name match)." Easier to implement, weaker as a baseline. Worth deciding which to ship.

Let me know which of these you want to discuss first, or if there's anything missing.
