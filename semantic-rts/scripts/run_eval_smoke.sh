#!/usr/bin/env bash
# Smoke test: 5 Chart bugs, all three methods. Loose recall ≥ 0.5 threshold.
set -euo pipefail

srts eval --project Chart --bug-start 1 --bug-end 5

echo "[smoke] Checking recall ≥ 0.5 and selection_rate ≤ 0.6 ..."
python - <<'PY'
import csv, sys
rows = list(csv.DictReader(open("data/eval/results.csv")))
ours = [r for r in rows if r["method"] == "ours"]
recalls = [float(r["recall"]) for r in ours]
rates   = [float(r["selection_rate"]) for r in ours]
if not recalls:
    print("No results found.", file=sys.stderr); sys.exit(1)
mean_recall = sum(recalls) / len(recalls)
mean_rate   = sum(rates)   / len(rates)
print(f"mean recall={mean_recall:.3f}  mean selection_rate={mean_rate:.3f}")
if mean_recall < 0.5:
    print("FAIL: recall below 0.5", file=sys.stderr); sys.exit(1)
if mean_rate > 0.6:
    print("FAIL: selection_rate above 0.6", file=sys.stderr); sys.exit(1)
print("OK")
PY
