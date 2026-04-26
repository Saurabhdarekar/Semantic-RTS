#!/usr/bin/env bash
# One-time Defects4J setup. Run once before any eval.
set -euo pipefail

D4J_DIR="${1:-$HOME/defects4j}"

echo "[setup] Cloning Defects4J into $D4J_DIR ..."
git clone https://github.com/rjust/defects4j "$D4J_DIR"
cd "$D4J_DIR"

echo "[setup] Installing Perl dependencies ..."
cpanm --installdeps .

echo "[setup] Initializing Defects4J ..."
./init.sh

echo "[setup] Done. Add to your shell profile:"
echo "  export DEFECTS4J_HOME=$D4J_DIR"
echo "  export PATH=\$PATH:\$DEFECTS4J_HOME/framework/bin"
echo ""
echo "[setup] Smoke test:"
echo "  defects4j info -p Chart"
