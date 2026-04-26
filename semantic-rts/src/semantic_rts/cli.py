"""CLI entrypoint: srts build | select | eval | baseline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click

from semantic_rts import __version__
from semantic_rts.config import load_config


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(__version__, prog_name="srts")
@click.option(
    "--config", "config_path",
    default=None, metavar="PATH",
    help="Path to config YAML (default: config/default.yaml).",
)
@click.option(
    "--overlay", "overlay_path",
    default=None, metavar="PATH",
    help="Ablation/override YAML merged on top of --config.",
)
@click.option(
    "--set", "overrides",
    multiple=True, metavar="KEY=VALUE",
    help="One-off config override, e.g. --set llm.chat_model=gemini-2.5-pro.",
)
@click.pass_context
def main(ctx: click.Context, config_path: Optional[str], overlay_path: Optional[str], overrides: tuple[str, ...]) -> None:
    """Semantic-Aware Regression Test Selection (CS 527 Group 7)."""
    ctx.ensure_object(dict)
    parsed_overrides = {}
    for item in overrides:
        if "=" not in item:
            raise click.BadParameter(f"--set expects KEY=VALUE, got: {item!r}")
        k, v = item.split("=", 1)
        parsed_overrides[k] = v
    ctx.obj["config"] = load_config(config_path, overlay_path, parsed_overrides or None)


# ---------------------------------------------------------------------------
# srts build  (Phase 1 — Knowledge Base Construction)
# ---------------------------------------------------------------------------

@main.command()
@click.option("--project", required=True, help="Defects4J project name, e.g. Chart.")
@click.option(
    "--project-path", required=True, type=click.Path(exists=True),
    help="Path to the checked-out project (fixed version).",
)
@click.option("--resume/--force", default=True, show_default=True,
              help="Resume from existing tests.jsonl, or force full rebuild.")
@click.option("--max-requests", default=None, type=int,
              help="Abort if LLM request count exceeds this limit.")
@click.pass_context
def build(ctx: click.Context, project: str, project_path: str, resume: bool, max_requests: Optional[int]) -> None:
    """Phase 1: Build the semantic knowledge base for a project."""
    cfg = ctx.obj["config"]
    click.echo(f"[build] project={project}  path={project_path}  resume={resume}")
    click.echo(f"        kb_dir={cfg.paths.kb_dir}  model={cfg.llm.chat_model}")
    from semantic_rts.kb.builder import build_kb
    build_kb(project_path, project, cfg, resume=resume, max_requests=max_requests)
    click.echo("[build] Done.")


# ---------------------------------------------------------------------------
# srts select  (Phase 2 + 3)
# ---------------------------------------------------------------------------

@main.command()
@click.option(
    "--diff", required=True, type=click.Path(exists=True),
    help="Path to the unified diff file.",
)
@click.option(
    "--kb", required=True, type=click.Path(exists=True),
    help="Path to the project KB directory (output of `srts build`).",
)
@click.option("--output", default=None, help="Write selected test IDs to this file (stdout if omitted).")
@click.pass_context
def select(ctx: click.Context, diff: str, kb: str, output: Optional[str]) -> None:
    """Phase 2+3: Retrieve and rank tests for a given diff."""
    import json
    from pathlib import Path

    from semantic_rts.impact.diff_parser import extract_changed_methods, parse_unified_diff
    from semantic_rts.impact.intent_agent import analyze_intent
    from semantic_rts.impact.retriever import retrieve
    from semantic_rts.kb.vector_store import VectorStore
    from semantic_rts.llm.client import GeminiClient
    from semantic_rts.llm.embeddings import GeminiEmbedder
    from semantic_rts.selector.ranker import select as run_select

    cfg = ctx.obj["config"]
    diff_text = Path(diff).read_text(encoding="utf-8")
    kb_path = Path(kb)

    click.echo(f"[select] Loading KB from {kb_path} ...")
    store = VectorStore.load(kb_path)
    click.echo(f"[select] KB has {store.size} tests.")

    click.echo("[select] Parsing diff ...")
    parsed = parse_unified_diff(diff_text)
    methods = extract_changed_methods(parsed)
    click.echo(f"[select] Files changed: {parsed.files_changed}")
    click.echo(f"[select] Methods changed: {methods}")

    click.echo("[select] Analyzing intent ...")
    client = GeminiClient(cfg)
    embedder = GeminiEmbedder(cfg)
    intent = analyze_intent(diff_text, parsed.files_changed, methods, client, cfg)
    click.echo(f"[select] Intent: {intent.intent_summary[:120]}")

    click.echo("[select] Retrieving candidates ...")
    candidates, trace = retrieve(
        intent, store, embedder, cfg,
        diff_hash=parsed.diff_hash,
        files_changed=parsed.files_changed,
        methods_changed=methods,
    )
    click.echo(f"[select] {len(candidates)} candidates retrieved.")

    click.echo("[select] Applying Safety Bridge + Precision Filter ...")
    selection = run_select(candidates, store.all_tests(), cfg)

    selected_ids = [t.test_id for t in selection.selected]
    click.echo(f"[select] Selected {len(selected_ids)} tests.")

    result_lines = "\n".join(selected_ids)
    if output:
        Path(output).write_text(result_lines + "\n", encoding="utf-8")
        click.echo(f"[select] Written to {output}")
    else:
        click.echo("\n--- Selected tests ---")
        for t in selection.selected:
            click.echo(f"  {t.test_id}  (score={t.score:.3f}, tier={t.tier}, reason={t.reason})")


# ---------------------------------------------------------------------------
# srts eval  (Evaluation loop)
# ---------------------------------------------------------------------------

@main.command()
@click.option("--project", required=True, help="Defects4J project name, e.g. Chart.")
@click.option(
    "--kb", required=True, type=click.Path(exists=True),
    help="Path to the pre-built KB directory (output of `srts build`).",
)
@click.option("--bug-start", default=1, show_default=True, type=int)
@click.option("--bug-end", required=True, type=int, help="Inclusive upper bound on bug IDs.")
@click.option(
    "--work-dir", default=None,
    help="Root directory for bug checkouts (default: config paths.work_dir).",
)
@click.option(
    "--d4j-home", default=None, envvar="DEFECTS4J_HOME",
    help="Path to Defects4J installation (default: $DEFECTS4J_HOME).",
)
@click.option("--output-dir", default=None, help="Override eval output directory.")
@click.pass_context
def eval(
    ctx: click.Context,
    project: str,
    kb: str,
    bug_start: int,
    bug_end: int,
    work_dir: Optional[str],
    d4j_home: Optional[str],
    output_dir: Optional[str],
) -> None:
    """Run the full evaluation loop over a range of Defects4J bugs."""
    from pathlib import Path

    from semantic_rts.eval.runner import run_eval

    cfg = ctx.obj["config"]
    kb_path = Path(kb)
    wd = Path(work_dir) if work_dir else Path(cfg.paths.work_dir)
    d4j = Path(d4j_home) if d4j_home else (Path(cfg.paths.defects4j_home) if cfg.paths.defects4j_home else None)
    out = Path(output_dir) if output_dir else Path(cfg.paths.eval_dir)

    bug_ids = list(range(bug_start, bug_end + 1))
    click.echo(f"[eval] project={project}  bugs={bug_start}–{bug_end}  kb={kb_path}")
    click.echo(f"       work_dir={wd}  d4j_home={d4j}  output={out}")

    results = run_eval(
        project=project,
        bug_ids=bug_ids,
        config=cfg,
        kb_path=kb_path,
        work_dir=wd,
        d4j_home=d4j,
        output_dir=out,
    )

    n = len(results)
    click.echo(f"[eval] Done. {n}/{len(bug_ids)} bugs completed.")
    if results:
        avg_recall = sum(r.recall for r in results) / n
        avg_sr = sum(r.selection_rate for r in results) / n
        n_safe = sum(1 for r in results if r.recall == 1.0)
        click.echo(
            f"[eval] avg_recall={avg_recall:.3f}  avg_sel_rate={avg_sr:.3f}"
            f"  safe(recall=1)={n_safe}/{n}"
        )


# ---------------------------------------------------------------------------
# srts baseline  (M7 — Retest-All and file-level static)
# ---------------------------------------------------------------------------

@main.command()
@click.option("--project", required=True, help="Defects4J project name, e.g. Chart.")
@click.option(
    "--method", required=True,
    type=click.Choice(["retest_all", "file_level_static"]),
    help="Baseline method to run.",
)
@click.option("--bug-start", default=1, show_default=True, type=int)
@click.option("--bug-end", required=True, type=int, help="Inclusive upper bound on bug IDs.")
@click.option(
    "--work-dir", default=None,
    help="Root directory for bug checkouts (default: config paths.work_dir).",
)
@click.option(
    "--d4j-home", default=None, envvar="DEFECTS4J_HOME",
    help="Path to Defects4J installation (default: $DEFECTS4J_HOME).",
)
@click.option("--output-dir", default=None, help="Override eval output directory.")
@click.pass_context
def baseline(
    ctx: click.Context,
    project: str,
    method: str,
    bug_start: int,
    bug_end: int,
    work_dir: Optional[str],
    d4j_home: Optional[str],
    output_dir: Optional[str],
) -> None:
    """Run a baseline RTS method (retest_all or file_level_static) over Defects4J bugs."""
    from semantic_rts.eval.baselines import run_baseline_eval

    cfg = ctx.obj["config"]
    wd = Path(work_dir) if work_dir else Path(cfg.paths.work_dir)
    d4j = Path(d4j_home) if d4j_home else (Path(cfg.paths.defects4j_home) if cfg.paths.defects4j_home else None)
    out = Path(output_dir) if output_dir else Path(cfg.paths.eval_dir)

    bug_ids = list(range(bug_start, bug_end + 1))
    click.echo(f"[baseline] project={project}  method={method}  bugs={bug_start}–{bug_end}")
    click.echo(f"           work_dir={wd}  output={out}")

    results = run_baseline_eval(
        project=project,
        bug_ids=bug_ids,
        method=method,
        work_dir=wd,
        d4j_home=d4j,
        output_dir=out,
    )

    n = len(results)
    click.echo(f"[baseline] Done. {n}/{len(bug_ids)} bugs completed.")
    if results:
        avg_recall = sum(r.recall for r in results) / n
        avg_sr = sum(r.selection_rate for r in results) / n
        n_safe = sum(1 for r in results if r.recall == 1.0)
        click.echo(
            f"[baseline] avg_recall={avg_recall:.3f}  avg_sel_rate={avg_sr:.3f}"
            f"  safe(recall=1)={n_safe}/{n}"
        )
