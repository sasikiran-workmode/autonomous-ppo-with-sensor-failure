# How to Run

This repository deliberately does **not** include the local `.venv/` virtual environment or Python cache files (`__pycache__/`). Create them on your own machine after cloning.

## Prerequisites

- Python 3.9 or newer
- `pip`
- Linux, macOS, or Windows with a Bash-compatible terminal (Git Bash or WSL on Windows)

## 1. Clone the repository

```bash
git clone git@github.com:sasikiran-workmode/autonomous-ppo-with-sensor-failure.git
cd autonomous-ppo-with-sensor-failure
```

## 2. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## 3. Run the dashboard with the included trained models

The repository includes the final trained agents and saved evaluation results, so the dashboard can start without training again:

```bash
SKIP_TRAIN=1 SKIP_EVAL=1 ./start-ppo.sh
```

Open the address printed by the script, normally <http://127.0.0.1:5000>.

To stop the dashboard, return to the terminal and press `Ctrl+C`.

## Optional: retrain and evaluate from scratch

Training both PPO agents on CPU takes about 90 minutes. To run the entire pipeline:

```bash
./start-ppo.sh
```

Useful options:

```bash
SKIP_TRAIN=1 ./start-ppo.sh          # evaluate existing models, then start dashboard
SKIP_EVAL=1 ./start-ppo.sh           # train models, then start dashboard
QUICK=1 ./start-ppo.sh               # short smoke-test training and evaluation
```

## Run individual steps manually

```bash
# Train the baseline agent
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  .venv/bin/python train.py --agent A --timesteps 200000

# Train the sensor-robust agent
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  .venv/bin/python train.py --agent B --timesteps 200000

# Run the evaluation
.venv/bin/python evaluate.py

# Start the local web application
.venv/bin/python app.py
```

## Generated files

Running the project may create `.venv/`, `__pycache__/`, log files, and intermediate files in `models/checkpoints/`. They are local generated artifacts and are intentionally ignored by Git. Final models in `models/` and the saved results in `results/` are included in the repository.
