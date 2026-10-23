#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  final-ppo.sh — Launch the dashboard instantly (no training, no evaluation)
#
#  Use this any time after the first full run. Models and results must exist.
#  If they don't exist, run ./start-ppo.sh first.
#
#  Usage:
#    chmod +x final-ppo.sh
#    ./final-ppo.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
PYTHON="$VENV_DIR/bin/python"
MODEL_A="models/agent_a_ppo.zip"
MODEL_B="models/agent_b_ppo.zip"
SUMMARY="results/summary.json"

echo ""
echo "====================================================================="
echo "  PPO Autonomous Driving -- Sensor Failure Robustness Study"
echo "  Launching dashboard (skip training + evaluation)"
echo "====================================================================="
echo ""

# Check venv exists
if [[ ! -f "$PYTHON" ]]; then
  echo "ERROR: Virtual environment not found."
  echo "       Run ./start-ppo.sh first to set everything up."
  exit 1
fi

# Check models exist
if [[ ! -f "$MODEL_A" ]] || [[ ! -f "$MODEL_B" ]]; then
  echo "ERROR: One or both models are missing."
  echo "       Run ./start-ppo.sh first to train the agents."
  exit 1
fi

# Check results exist
if [[ ! -f "$SUMMARY" ]]; then
  echo "WARNING: No evaluation results found (results/summary.json missing)."
  echo "         You can run evaluation from the browser after launch."
fi

source "$VENV_DIR/bin/activate"

PORT=5000
URL="http://127.0.0.1:${PORT}"

echo "  Dashboard starting at: $URL"
echo "  Press Ctrl+C to stop."
echo ""

# Open browser (non-blocking)
(sleep 2 && {
  if command -v xdg-open &>/dev/null; then
    xdg-open "$URL" 2>/dev/null || true
  elif command -v open &>/dev/null; then
    open "$URL" 2>/dev/null || true
  fi
}) &

exec "$PYTHON" app.py
