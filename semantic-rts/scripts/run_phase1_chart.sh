#!/usr/bin/env bash
# Build the knowledge base for Chart (fixed version of bug 1).
set -euo pipefail

CHART_PATH="${1:-/tmp/Chart_1_fixed}"

if [ ! -d "$CHART_PATH" ]; then
  echo "[phase1] Checking out Chart-1f to $CHART_PATH ..."
  defects4j checkout -p Chart -v 1f -w "$CHART_PATH"
fi

echo "[phase1] Building KB for Chart ..."
srts build --project Chart --project-path "$CHART_PATH"
echo "[phase1] Done."
