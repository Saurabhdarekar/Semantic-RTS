#!/usr/bin/env python3
"""Run all ablation studies over the Cli Defects4J project and print a summary table.

Usage (from the semantic-rts directory, with venv activated):
    python run_ablations.py

Results are written to data/eval/ablations/<name>_results.csv and a summary
table is printed to stdout.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Ensure we can import semantic_rts even without pip install -e
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from semantic_rts.config import load_config
from semantic_rts.eval.runner import run_eval, write_csv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT = "Cli"
BUG_START = 1
BUG_END = 40
KB_PATH = Path("data/kb/Cli")
WORK_DIR = Path("/tmp/d4j_eval")
D4J_HOME = Path(os.environ.get("DEFECTS4J_HOME", "")) if os.environ.get("DEFECTS4J_HOME") else None
OUTPUT_BASE = Path("data/eval/ablations")

ABLATIONS: list[tuple[str, str | None]] = [
    ("baseline",          None),
    ("no_bm25",           "config/ablations/no_bm25.yaml"),
    ("no_bridge",         "config/ablations/no_bridge.yaml"),
    ("no_llm",            "config/ablations/no_llm.yaml"),
    ("no_micro_bypass",   "config/ablations/no_micro_bypass.yaml"),
    ("no_negative_pass",  "config/ablations/no_negative_pass.yaml"),
    ("no_sensitivity",    "config/ablations/no_sensitivity.yaml"),
    ("no_tier",           "config/ablations/no_tier.yaml"),
    ("no_token_overlap",  "config/ablations/no_token_overlap.yaml"),
    ("no_topology",       "config/ablations/no_topology.yaml"),
]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_ablation(name: str, overlay: str | None) -> dict:
    print(f"\n{'='*60}")
    print(f"  Ablation: {name}")
    print(f"{'='*60}")

    cfg = load_config(overlay_path=overlay)
    output_dir = OUTPUT_BASE / name
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_eval(
        project=PROJECT,
        bug_ids=list(range(BUG_START, BUG_END + 1)),
        config=cfg,
        kb_path=KB_PATH,
        work_dir=WORK_DIR,
        d4j_home=D4J_HOME,
        output_dir=output_dir,
        method=name,
    )

    n = len(results)
    if n == 0:
        return {"name": name, "n": 0, "recall": 0.0, "sel_rate": 0.0, "safe": 0}

    avg_recall = sum(r.recall for r in results) / n
    avg_sr = sum(r.selection_rate for r in results) / n
    n_safe = sum(1 for r in results if r.recall == 1.0)
    avg_latency = sum(r.latency_ms for r in results) / n

    print(f"  → avg_recall={avg_recall:.3f}  avg_sel_rate={avg_sr:.3f}  safe={n_safe}/{n}  latency={avg_latency:.0f}ms")
    return {
        "name": name,
        "n": n,
        "recall": avg_recall,
        "sel_rate": avg_sr,
        "safe": n_safe,
        "latency_ms": avg_latency,
    }


def print_table(rows: list[dict]) -> None:
    print("\n")
    print("=" * 72)
    print(f"  Ablation Study Results — {PROJECT} bugs {BUG_START}–{BUG_END}")
    print("=" * 72)
    header = f"  {'Ablation':<22}  {'Recall':>8}  {'Sel Rate':>9}  {'Safe':>8}  {'Latency':>9}"
    print(header)
    print("  " + "-" * 68)

    baseline = next((r for r in rows if r["name"] == "baseline"), None)
    for row in rows:
        if row["n"] == 0:
            print(f"  {row['name']:<22}  {'N/A':>8}  {'N/A':>9}  {'N/A':>8}  {'N/A':>9}")
            continue
        n = row["n"]
        recall_str = f"{row['recall']:.3f}"
        sr_str = f"{row['sel_rate']:.3f}"
        safe_str = f"{row['safe']}/{n}"
        lat_str = f"{row['latency_ms']:.0f}ms"

        # Show delta vs baseline for non-baseline rows
        if baseline and row["name"] != "baseline" and baseline["n"] > 0:
            dr = row["recall"] - baseline["recall"]
            ds = row["sel_rate"] - baseline["sel_rate"]
            recall_str += f" ({dr:+.3f})"
            sr_str += f" ({ds:+.3f})"

        print(f"  {row['name']:<22}  {recall_str:>16}  {sr_str:>17}  {safe_str:>8}  {lat_str:>9}")

    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Semantic-RTS ablation studies")
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help="Run only these ablation names (e.g. --only baseline no_bm25)")
    parser.add_argument("--skip", nargs="+", metavar="NAME",
                        help="Skip these ablation names")
    args = parser.parse_args()

    to_run = ABLATIONS
    if args.only:
        to_run = [(n, o) for n, o in ABLATIONS if n in args.only]
    if args.skip:
        to_run = [(n, o) for n, o in to_run if n not in args.skip]

    summary_rows: list[dict] = []
    for name, overlay in to_run:
        row = run_ablation(name, overlay)
        summary_rows.append(row)

    print_table(summary_rows)

    # Save combined summary CSV
    summary_path = OUTPUT_BASE / "ablation_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        f.write("ablation,n_bugs,avg_recall,avg_sel_rate,n_safe,avg_latency_ms\n")
        for row in summary_rows:
            f.write(f"{row['name']},{row['n']},{row['recall']:.4f},{row['sel_rate']:.4f},{row['safe']},{row.get('latency_ms',0):.0f}\n")
    print(f"Summary CSV written to {summary_path}")
