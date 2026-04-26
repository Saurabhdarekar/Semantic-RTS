#!/usr/bin/env bash
# Full evaluation run. Set SRTS_CACHE_ONLY=1 for the report run.
set -euo pipefail

export SRTS_CACHE_ONLY="${SRTS_CACHE_ONLY:-0}"
PROJECTS=("Chart" "Lang" "Math" "Time" "Closure")

for project in "${PROJECTS[@]}"; do
  echo "[eval] === $project ==="
  srts eval --project "$project"
done

echo "[eval] Generating summary report ..."
python -c "
from semantic_rts.eval.report import generate_report
from pathlib import Path
generate_report(Path('data/eval/results.csv'), Path('data/eval'))
"

echo "[eval] Done. See data/eval/summary.md"
