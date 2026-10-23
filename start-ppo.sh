#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────────────────
#  start-ppo.sh — One-shot launcher for the PPO Autonomous Driving project
#
#  What this script does (in order):
#    1. Creates the Python virtual environment (.venv) if it doesn't exist
#    2. Installs all dependencies from requirements.txt
#    3. Trains Agent A (200k timesteps) if model not already present
#    4. Trains Agent B (200k timesteps) if model not already present
#    5. Runs the full evaluation (30 eps x 4 failure rates) if not done
#    6. Launches the Flask frontend at http://127.0.0.1:5000
#    7. Opens the browser automatically
#
#  Skip flags (set env vars to skip steps):
#    SKIP_TRAIN=1   ./start-ppo.sh   -- skips training, loads existing models
#    SKIP_EVAL=1    ./start-ppo.sh   -- skips evaluation, uses existing results
#    QUICK=1        ./start-ppo.sh   -- uses 10k timesteps + 10 eval episodes
#
#  Usage:
#    chmod +x start-ppo.sh
#    ./start-ppo.sh
# ───────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Script location (project root) ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Config ────────────────────────────────────────────────────────────────────
VENV_DIR=".venv"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

TRAIN_TIMESTEPS=${QUICK:+10000};    TRAIN_TIMESTEPS=${TRAIN_TIMESTEPS:-200000}
EVAL_EPISODES=${QUICK:+10};         EVAL_EPISODES=${EVAL_EPISODES:-30}
SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

MODEL_A="models/agent_a_ppo.zip"
MODEL_B="models/agent_b_ppo.zip"
SUMMARY="results/summary.json"

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "====================================================================="
echo "  PPO Autonomous Driving -- Sensor Failure Robustness Study"
echo "====================================================================="
echo ""
echo "  Research question:"
echo "  Does sensor-failure training improve autonomous-driving robustness?"
echo ""
if [[ "${QUICK:-0}" == "1" ]]; then
  echo "  [QUICK mode: using 10k training steps & 10 eval episodes]"
fi
echo ""

# ── Step 1: Virtual environment ───────────────────────────────────────────────
echo "-- Step 1/6: Python virtual environment"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "   Creating virtual environment in $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
  echo "   Done."
else
  echo "   Virtual environment already exists."
fi

source "$VENV_DIR/bin/activate"

# ── Step 2: Install dependencies ─────────────────────────────────────────────
echo ""
echo "-- Step 2/6: Installing dependencies"
echo "   Running pip install -r requirements.txt..."
"$PIP" install --quiet --upgrade pip
"$PIP" install --quiet -r requirements.txt
echo "   Dependencies installed."

"$PYTHON" -c "
import highway_env, stable_baselines3, gymnasium, flask, numpy, pandas, matplotlib, PIL
print('   All imports verified.')
"

# ── Step 3 & 4: Train agents ──────────────────────────────────────────────────
echo ""
echo "-- Step 3-4/6: Training Agents A and B"

if [[ "$SKIP_TRAIN" == "1" ]]; then
  echo "   SKIP_TRAIN=1: skipping training."
  if [[ ! -f "$MODEL_A" ]] || [[ ! -f "$MODEL_B" ]]; then
    echo "   ERROR: Models not found but SKIP_TRAIN=1 is set."
    echo "   Either train first or remove SKIP_TRAIN=1."
    exit 1
  fi
else
  mkdir -p models results results/graphs

  # Training uses SDL dummy since it's headless (no render needed)
  export SDL_VIDEODRIVER=dummy
  export SDL_AUDIODRIVER=dummy

  # Agent A
  if [[ -f "$MODEL_A" ]]; then
    echo "   Agent A model already exists -- skipping training."
  else
    echo "   Training Agent A (${TRAIN_TIMESTEPS} timesteps)..."
    "$PYTHON" train.py --agent A --timesteps "$TRAIN_TIMESTEPS"
    echo "   Agent A trained and saved."
  fi

  # Agent B
  if [[ -f "$MODEL_B" ]]; then
    echo "   Agent B model already exists -- skipping training."
  else
    echo "   Training Agent B (${TRAIN_TIMESTEPS} timesteps, dropout=0.10)..."
    "$PYTHON" train.py --agent B --timesteps "$TRAIN_TIMESTEPS" --dropout-prob 0.10
    echo "   Agent B trained and saved."
  fi

  # Unset dummy driver so Flask/live demo uses real display
  unset SDL_VIDEODRIVER
  unset SDL_AUDIODRIVER
fi

# ── Step 5: Evaluation ────────────────────────────────────────────────────────
echo ""
echo "-- Step 5/6: Running evaluation"

if [[ "$SKIP_EVAL" == "1" ]]; then
  echo "   SKIP_EVAL=1: skipping evaluation."
elif [[ -f "$SUMMARY" ]]; then
  echo "   Evaluation results already exist -- skipping."
  echo "   Delete $SUMMARY to force re-evaluation."
else
  echo "   Evaluating both agents at 0%, 5%, 10%, 20% failure rates..."
  echo "   (${EVAL_EPISODES} episodes per condition x 8 conditions)"
  # Evaluation also headless
  SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
    "$PYTHON" evaluate.py --episodes "$EVAL_EPISODES" --failure-rates 0 0.05 0.10 0.20
  echo "   Evaluation complete. Results saved to results/."
fi

# ── Step 6: Launch Flask ──────────────────────────────────────────────────────
echo ""
echo "-- Step 6/6: Launching the dashboard"
echo ""

PORT=5000
URL="http://127.0.0.1:${PORT}"

echo "  Dashboard starting at: $URL"
echo "  Press Ctrl+C to stop."
echo ""

# Open browser (non-blocking, best-effort)
(sleep 2 && {
  if command -v xdg-open &>/dev/null; then
    xdg-open "$URL" 2>/dev/null || true
  elif command -v open &>/dev/null; then
    open "$URL" 2>/dev/null || true
  fi
}) &

# Run Flask in foreground (live demo uses real display for rendering)
exec "$PYTHON" app.py
