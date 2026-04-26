# Semantic-Aware RTS — Implementation Plan

**Project:** CS 527 Group 7 — Semantic-Aware Regression Test Selection Using LLMs
**Authors:** Ankit Chavan, Saurabh Darekar
**Stack decisions:** Python 3.11+, LangChain, FAISS (vector store), Gemini 2.5 Flash-Lite (LLM + embeddings via free tier), Defects4J v2.0
**Strategy:** Build on 1–2 small Defects4J projects first (Chart, Lang), then scale to the full 17-project benchmark.

---

## 1. Executive Summary

We are building an RTS (Regression Test Selection) pipeline that, given a code diff, picks a small but safe subset of tests to run instead of the full suite. The novelty is using LLM-generated semantic summaries of tests + diffs, retrieved via FAISS vector similarity, and ranked with a "Safety Bridge" that protects critical tests from being filtered out.

The pipeline has **three phases**, mirroring the proposal:

1. **Phase 1 — Knowledge Base Construction (offline, per project):** summarize every test, assign a safety tier, embed, store in FAISS.
2. **Phase 2 — Impact Analysis (per commit):** read the diff, summarize the change intent with an LLM, embed it, retrieve top-K similar tests.
3. **Phase 3 — Risk-Aware Selection (per commit):** apply Safety Bridge + Precision Filter rules to produce the final test subset.

Evaluation runs all three baselines (Retest-All, STARTS, our pipeline) over Defects4J bugs and reports recall + selection rate.

---

## 2. Repository Structure

```
semantic-rts/
├── README.md
├── pyproject.toml                  # uv / poetry; pinned deps
├── .env.example                    # GOOGLE_API_KEY=...
├── config/
│   ├── default.yaml                # model names, top_k, weights, paths
│   └── projects.yaml               # which Defects4J projects/bugs to run
├── src/
│   └── semantic_rts/
│       ├── __init__.py
│       ├── cli.py                  # entrypoint: `srts build`, `srts select`, `srts eval`
│       ├── config.py               # pydantic settings loader
│       ├── llm/
│       │   ├── client.py           # Gemini wrapper w/ retry + rate-limit
│       │   ├── prompts.py          # all prompt templates (versioned strings)
│       │   └── embeddings.py       # text-embedding-005 wrapper
│       ├── kb/                     # Phase 1
│       │   ├── test_parser.py      # parse JUnit tests from a project
│       │   ├── summarizer.py       # LLM call → test summary
│       │   ├── tier_classifier.py  # rule-based + LLM tier assignment
│       │   ├── vector_store.py     # FAISS wrapper (build/save/load)
│       │   └── builder.py          # orchestrator for Phase 1
│       ├── impact/                 # Phase 2
│       │   ├── diff_parser.py      # git diff → structured changes
│       │   ├── intent_agent.py     # LLM call → change-intent summary
│       │   └── retriever.py        # FAISS top-K + scores
│       ├── selector/               # Phase 3
│       │   ├── safety_bridge.py    # rule engine
│       │   ├── precision_filter.py
│       │   └── ranker.py           # combined scoring
│       ├── defects4j/
│       │   ├── checkout.py         # `defects4j checkout` wrapper
│       │   ├── diff_extractor.py   # buggy↔fixed diff
│       │   └── ground_truth.py     # known failing tests
│       ├── baselines/
│       │   ├── retest_all.py
│       │   └── starts_runner.py    # wraps STARTS jar
│       └── eval/
│           ├── metrics.py          # recall, selection rate, F1, latency
│           ├── runner.py           # bug-by-bug eval loop
│           └── report.py           # CSV + Markdown summary
├── scripts/
│   ├── setup_defects4j.sh          # installs Defects4J once
│   ├── run_phase1_chart.sh
│   ├── run_eval_smoke.sh           # 5-bug sanity check
│   └── run_eval_full.sh
├── tests/
│   ├── unit/                       # pure-Python logic
│   ├── integration/                # real Gemini calls (gated by env)
│   └── fixtures/
│       ├── mini_project/           # tiny synthetic test suite
│       └── sample_diffs/
├── data/                           # gitignored
│   ├── kb/                         # FAISS indexes per project
│   └── eval/                       # results CSVs
└── docs/
    ├── architecture.md
    ├── prompts.md                  # prompt change log
    └── eval_protocol.md
```

---

## 3. Prerequisites & One-Time Setup

These are the manual steps a human (not Claude Code) does **before** development begins.

### 3.1 Local environment
- Java 8 and Java 11 installed (Defects4J needs both — bug projects vary)
- `git`, `svn`, `perl` (Defects4J dependencies)
- Python 3.11+
- ~10 GB free disk for Defects4J project checkouts

### 3.2 Defects4J
```bash
git clone https://github.com/rjust/defects4j
cd defects4j
cpanm --installdeps .
./init.sh
export PATH=$PATH:$(pwd)/framework/bin
defects4j info -p Chart   # smoke test
```

### 3.3 Gemini API key
- Create a free key at https://aistudio.google.com/app/apikey (no credit card required for the free tier)
- Free tier on Gemini 2.5 Flash-Lite: ~1,500 requests/day, sufficient for prototype phase
- Save in `.env` as `GOOGLE_API_KEY=...`

### 3.4 STARTS (baseline)
- STARTS is a Maven plugin. Most Defects4J projects build with Ant, not Maven — we'll need a wrapper that converts the project's classpath/source layout. See §10 for the fallback plan.

---

## 4. Configuration (config/default.yaml)

```yaml
llm:
  provider: gemini
  chat_model: gemini-2.5-flash-lite
  embedding_model: text-embedding-005
  max_retries: 3
  rate_limit_rpm: 14            # stay under 15 RPM free-tier limit
  rate_limit_rpd: 1400          # stay under 1500 RPD

vector_store:
  type: faiss
  index_kind: IndexFlatIP       # cosine sim via normalized vectors; small-scale
  embedding_dim: 768            # text-embedding-005 default

kb:
  summary_max_tokens: 200
  tier_keywords:
    tier_1: [security, auth, password, crypto, privacy, payment]
    tier_2: [persistence, database, transaction, concurrency]
    tier_3: [api, controller, service]
    tier_4: [util, helper, format]
    tier_5: [getter, setter, toString, equals, hashCode]

retrieval:
  top_k: 30
  similarity_threshold: 0.55

selector:
  safety_bridge:
    always_include_tier_1: true
    always_include_tier_2_if_score_above: 0.40
  precision_filter:
    drop_tier_5_if_score_below: 0.65
    drop_tier_4_if_score_below: 0.50
  max_selected: 100

paths:
  defects4j_home: ${DEFECTS4J_HOME}
  kb_dir: data/kb
  eval_dir: data/eval
```

---

## 5. Phase 1 — Knowledge Base Construction

### 5.1 Inputs / Outputs
- **Input:** a checked-out Defects4J project at the **fixed** version (use the fixed tree as the canonical test suite for a project)
- **Output:** `data/kb/<project>/` containing:
  - `tests.jsonl` — one row per test with `{test_id, class, method, source, summary, tier}`
  - `index.faiss` — FAISS vector index
  - `index.meta.json` — id ↔ row mapping

### 5.2 Algorithm

1. **Discover tests.** Walk the project's test source root. A "test" is any method annotated `@Test` (JUnit 4) or `@org.junit.jupiter.api.Test` (JUnit 5). Record fully qualified name `pkg.Class::method`.
2. **Extract source.** Pull the method body + class-level Javadoc + class-level imports. Truncate to ~1,500 tokens to stay within prompt budget.
3. **Summarize.** Call Gemini with the prompt in §5.3. Get a 1–2 sentence purpose summary.
4. **Assign tier.** Two-step classifier:
   - **Rule pass:** match the test's class name + summary against `tier_keywords` from config. If matched, assign that tier.
   - **LLM fallback:** for tests not matched by rules, ask Gemini to classify into Tier 1–5 with a short rubric. Cache result.
5. **Embed.** Use `text-embedding-005` on the string `"{summary}\nTIER:{tier}\nCLASS:{class_simple_name}"`. Normalize for cosine similarity.
6. **Index.** Add to FAISS `IndexFlatIP`. Save index + metadata.

### 5.3 Prompt — Test Summarizer (versioned `v1`)

> You are analyzing a unit test to produce a one-sentence semantic summary for retrieval.
>
> Test class: `{class_fqn}`
> Test method: `{method_name}`
>
> Source:
> ```java
> {source}
> ```
>
> Output JSON only:
> ```
> { "summary": "<one sentence describing what behavior this test verifies, in plain English>",
>   "concepts": ["<3-7 short keywords or feature names>"] }
> ```

### 5.4 Prompt — Tier Classifier (versioned `v1`)

> Classify this test into a safety tier:
> - Tier 1 (CRITICAL): security, authentication, authorization, payment, privacy, cryptography, data integrity
> - Tier 2 (HIGH): persistence, transactions, concurrency, public API contracts
> - Tier 3 (MEDIUM): business logic, services, controllers
> - Tier 4 (LOW): utilities, formatting, helpers
> - Tier 5 (TRIVIAL): getters/setters/toString/equals/hashCode, simple object construction
>
> Test class: `{class_fqn}`
> Summary: `{summary}`
> Concepts: `{concepts}`
>
> Output JSON: `{ "tier": <1-5>, "reason": "<brief>" }`

### 5.5 Engineering notes
- **Idempotent.** A `--resume` flag should skip tests already in `tests.jsonl`. Use file hash + method signature as cache key.
- **Concurrency.** Cap at 5 concurrent LLM calls; respect the RPM limit using a token-bucket limiter.
- **Cost guard.** Print running token + request totals; abort if exceeds `--max-requests` arg.
- **Determinism.** Set Gemini `temperature=0` for both summarizer and classifier.

---

## 6. Phase 2 — Impact Analysis & Retrieval

### 6.1 Inputs / Outputs
- **Input:** a unified diff (string) and a project KB
- **Output:** ranked candidate list `[{test_id, score, tier}, ...]` of length `top_k`

### 6.2 Algorithm

1. **Parse the diff.** Use `unidiff` to extract changed files, hunk locations, and changed methods (heuristic: walk up from a hunk to the nearest `class`/`method` declaration).
2. **Summarize change intent.** Single LLM call with the diff + extracted changed method names. Returns natural-language description and concept keywords (prompt §6.3).
3. **Embed the intent.** Same embedder as Phase 1.
4. **Retrieve.** FAISS top-K cosine similarity. Return rows with attached scores.

### 6.3 Prompt — Impact Analyst (versioned `v1`)

> You are analyzing a Git diff to infer the developer's intent for test selection purposes.
>
> Files changed: `{file_list}`
> Changed methods: `{method_list}`
>
> Diff:
> ```diff
> {diff}
> ```
>
> Output JSON only:
> ```
> { "intent_summary": "<2-3 sentences describing what behavior is changing and why>",
>   "concepts": ["<3-10 keywords/feature names that should match relevant tests>"],
>   "risk_areas": ["<security|persistence|api|ui|util|other>"] }
> ```

### 6.4 Engineering notes
- **Diff size cap.** If the diff exceeds 8K tokens, chunk by file and merge intent summaries. (Rare for Defects4J bugs.)
- **Cache by diff hash.** Same diff → same intent. Save under `data/cache/intents/`.

---

## 7. Phase 3 — Risk-Aware Selection

This is **pure Python** — no LLM call. Fast, deterministic, easy to unit-test.

### 7.1 Algorithm

```
INPUT: candidates (top_k from Phase 2), config
selected = set()

# Safety Bridge — never miss a critical test
for test in all_tests_in_kb:
    if test.tier == 1 and config.safety_bridge.always_include_tier_1:
        selected.add(test.id)

for test in candidates:
    if test.tier == 2 and test.score >= config.safety_bridge.always_include_tier_2_if_score_above:
        selected.add(test.id)

# Semantic relevance — include candidates above threshold
for test in candidates:
    if test.score >= config.retrieval.similarity_threshold:
        selected.add(test.id)

# Precision Filter — drop low-value matches
selected = {
    t for t in selected
    if not (t.tier == 5 and t.score < config.precision_filter.drop_tier_5_if_score_below)
    and not (t.tier == 4 and t.score < config.precision_filter.drop_tier_4_if_score_below)
}

# Cap size
selected = top_n_by_score(selected, config.selector.max_selected)
```

### 7.2 Engineering notes
- All thresholds live in config — never hardcode. Eval phase will sweep these.
- Emit a per-test trace `{kept|dropped, reason}` for debugging and ablation.

---

## 8. Defects4J Integration

### 8.1 Bug iteration

For each `(project, bug_id)` pair:
1. `defects4j checkout -p {project} -v {bug_id}f -w /tmp/{project}_{bug_id}_fixed`
2. `defects4j checkout -p {project} -v {bug_id}b -w /tmp/{project}_{bug_id}_buggy`
3. Generate diff: `diff -ruN {fixed}/src {buggy}/src` (or use `defects4j export -p src.diff`)
4. Get ground truth: `defects4j export -p tests.trigger -w /tmp/{project}_{bug_id}_buggy`
5. Run our pipeline on the diff against the project's KB → selected tests
6. Compute metrics (§9)

**Important:** the KB is built once **per project** (using the `f` version of bug 1 as canonical, or the master branch). It is *not* rebuilt per bug — that would be cheating against the baselines and unrealistic.

### 8.2 Recommended starter projects

| Project   | Bugs | Why                                                  |
|-----------|------|------------------------------------------------------|
| **Chart** | 26   | Small, ~700 tests, straightforward Ant build         |
| **Lang**  | 65   | Pure utility code, large clean test suite            |
| **Math**  | 106  | Bigger; good stress test once Chart + Lang work      |

Start with **Chart**. Its size keeps Phase 1 LLM cost under ~$0.10 even on the paid tier.

---

## 9. Evaluation

### 9.1 Metrics (per bug)

- **Recall (Safety):** `|selected ∩ failing_tests| / |failing_tests|`. Target: ≥ 0.95 mean.
- **Selection Rate (Efficiency):** `|selected| / |all_tests|`. Target: ≤ 0.30 mean.
- **Precision (informational):** `|selected ∩ failing_tests| / |selected|`
- **End-to-end latency:** Phase 1 (amortized) + Phase 2 + Phase 3 wall time
- **LLM cost:** total input + output tokens, dollar estimate

### 9.2 Aggregation

- Per project: mean ± std for each metric
- Across baselines: paired comparison Retest-All vs STARTS vs Ours
- Significance: Wilcoxon signed-rank test on per-bug recall and selection rate
- Tier ablation: re-run with Safety Bridge disabled; expect recall to drop on Tier-1-relevant bugs

### 9.3 Output

`data/eval/results.csv` with columns:
`project, bug_id, method, recall, selection_rate, precision, latency_ms, cost_usd, n_failing, n_selected, n_total`

Plus `data/eval/summary.md` rendered from the CSV — tables and a few plots.

---

## 10. Risks & Mitigations

| Risk                                           | Mitigation                                                                                                                              |
|-----------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| **Free-tier rate limits during full eval**     | Token-bucket limiter; checkpoint every 10 bugs; can switch to paid Flash-Lite (~$5–10 total for full benchmark) if needed              |
| **STARTS doesn't run on Ant-only projects**    | Fallback baseline: file-level static RTS (any test whose source/imports transitively reach a changed class). Document the substitution. |
| **LLM-summarized intent misses subtle bugs**   | Safety Bridge always includes Tier-1 tests regardless of score. Ablation will quantify how often this rescues recall.                   |
| **Embedding quality on code-heavy text**       | Embed the **summary**, not raw code. If recall is poor, swap to a code-aware embedder (e.g., `voyage-code-3` — small extra cost).      |
| **Determinism / reproducibility**              | `temperature=0`, pinned model versions, cache all LLM responses by input hash, commit `data/cache/` for the eval set                   |
| **Test name parsing edge cases (parameterized, nested, JUnit 5)** | Unit-test the parser with fixtures for each style; for unsupported styles, log + skip rather than crash             |
| **Diff parsing for renames/moves**             | Use `git diff -M --find-renames`; treat renamed files as both removed + added contexts                                                  |

---

## 11. Build Order (Milestones)

A linear sequence Claude Code can follow. Each milestone ends with a passing test or runnable command.

### M1 — Skeleton (1 evening)
- `pyproject.toml`, package layout, CLI entrypoint stubs, `.env` loading, config loader
- **Done when:** `srts --help` works; `pytest` runs (zero tests, zero failures)

### M2 — LLM + embedding wrappers (1 evening)
- `llm/client.py`, `llm/embeddings.py` with retry/rate-limit/caching
- Integration test (gated by `RUN_LIVE_TESTS=1`) calls Gemini once and checks shape
- **Done when:** integration test passes against live API

### M3 — Phase 1 on a tiny fixture (1 evening)
- Synthetic mini-project under `tests/fixtures/mini_project/` with 5 hand-written tests
- `kb/builder.py` end-to-end on the fixture; produces `tests.jsonl` + FAISS index
- **Done when:** unit test verifies all 5 tests are summarized + tiered + indexed

### M4 — Phase 2 + 3 (1–2 evenings)
- Diff parser, intent agent, retriever, selector
- Unit tests for selector logic with mocked candidate lists
- **Done when:** `srts select --diff some.diff --kb fixture` returns plausible test ids

### M5 — Defects4J adapter (1 evening)
- Checkout/diff/ground-truth helpers; smoke test on Chart bug 1
- **Done when:** can generate diff and ground truth for one bug programmatically

### M6 — End-to-end on Chart (1 evening)
- Build KB for Chart once; run pipeline on 5 Chart bugs; print recall/selection rate
- **Done when:** results CSV populated for 5 bugs

### M7 — Baselines (1–2 evenings)
- Retest-All trivially returns all tests. STARTS wrapper or static-fallback baseline.
- **Done when:** all three methods produce comparable rows in results CSV

### M8 — Full eval on Chart + Lang (overnight run)
- Run all bugs in Chart and Lang; generate `summary.md`
- **Done when:** report has aggregated metrics + significance test results

### M9 — Scale to remaining projects (incremental)
- Per-project KB build; eval run; merge into master CSV
- Track cost; switch to paid tier if hitting rate limits

### M10 — Ablations + writeup
- Disable Safety Bridge → measure recall delta
- Sweep `top_k` ∈ {10, 20, 30, 50}; sweep `similarity_threshold`
- Write up findings, push final report

---

## 12. Testing Strategy

- **Unit tests** (no network): config loader, diff parser, tier rule classifier, selector logic, FAISS save/load, prompt template formatting. Target ≥ 80% coverage on `selector/`, `kb/tier_classifier.py`, `impact/diff_parser.py`.
- **Integration tests** (live LLM, gated by env var): one summarizer call, one impact-analyst call, one embedding call. Each asserts schema, not content.
- **End-to-end smoke** (`scripts/run_eval_smoke.sh`): 5 bugs from Chart, full pipeline, asserts `recall ≥ 0.5` and `selection_rate ≤ 0.6` (loose thresholds — just to catch breakage in CI).
- **Fixtures:** snapshot LLM responses for the mini-project so unit tests don't need API access.

---

## 13. Reproducibility Checklist

- [ ] Pinned `pyproject.toml` with exact versions
- [ ] `.env.example` documents every required variable
- [ ] All prompts are versioned strings in `llm/prompts.py` (e.g., `SUMMARIZER_V1`)
- [ ] LLM responses cached by input hash — eval is replayable without API
- [ ] Random seeds set wherever any nondeterminism remains
- [ ] `scripts/run_eval_full.sh` reproduces every number in the final report
- [ ] `data/cache/` committed for the official eval set (or hosted as a release artifact)
- [ ] README has a "reproduce in 4 commands" section

---

## 14. What to Hand to Claude Code

When you start the Claude Code session, give it:

1. **This file** (`IMPLEMENTATION_PLAN_v1.md`)
2. **The proposal PDF** (for reference on Safety Bridge / Precision Filter wording)
3. A **starting prompt** like:

   > Read `IMPLEMENTATION_PLAN_v1.md`. Build Milestone 1 (M1 — Skeleton). Use `uv` for dependency management. After scaffolding, show me `srts --help` and the pytest output. Stop and wait for my approval before starting M2.

   Going milestone-by-milestone with explicit stop points keeps the agent on rails and lets you review each chunk before it builds on top of it.

4. **Your `GOOGLE_API_KEY`** in `.env` so M2's integration test passes.

---

## 15. Open Questions to Resolve Before Coding

These are decisions worth making now rather than mid-build:

1. **Embedding model.** `text-embedding-005` is cheap and strong, but `voyage-code-3` is code-aware. Recommendation: start with `text-embedding-005`; revisit only if recall is below target.
2. **JUnit 4 vs 5.** Defects4J projects mix both. Confirm the parser handles both styles before M3.
3. **Per-project vs cross-project KB.** Plan above is per-project (each project has its own FAISS index). Cross-project would require shared vocabulary — not recommended for v1.
4. **STARTS substitute.** If STARTS is too painful to wire to Ant projects, do we accept a simpler static baseline (file-level transitive reach)? Plan above says yes; flag in writeup.
5. **Reporting "test failure" granularity.** Defects4J `tests.trigger` lists fault-triggering tests. Are we counting recall at method or class level? Plan above uses **method**. Confirm.
