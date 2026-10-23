# Implementation Status — PPO Autonomous Driving Sensor-Failure Study

Last checked: 2026-10-23

This file is an implementation-status record, replacing the original planning checklist. It was checked against the current source code, saved models/results, and read-only Flask smoke tests.

**Legend:** `[x]` implemented or artifact present; `[~]` implemented but not fully run/validated in this workspace; `[ ]` not implemented or still needs verification.

## Checked evidence

- [x] All Python files parse successfully.
- [x] Flask smoke tests returned HTTP 200 for `/`, `/api/status`, `/api/results`, `/api/results/graphs`, `/api/results/episodes`, and `/api/explain/result`.
- [x] Final model files and metadata exist for both Agent A and Agent B.
- [x] Saved CSV, JSON summary, and five PNG graphs exist.
- [~] The saved evaluation contains **1 episode per condition** (8 rows total), not the planned 30 episodes per condition. Its results must therefore be treated as a quick run, not statistically meaningful full-study evidence.

## Project setup

- [x] Project structure exists: configuration, environment, dropout wrapper, training, evaluation, metrics, live simulation, explanation, Flask backend, frontend, models, results, and documentation.
- [x] `requirements.txt` specifies Highway-Env, Stable-Baselines3, Gymnasium, Flask, NumPy, Pandas, Matplotlib, Pillow, Torch, Pygame, and Shimmy.
- [x] Local virtual environments and Python caches are intentionally ignored by `.gitignore`.
- [x] `README.md` and `howtorun.md` document installation, training, evaluation, and dashboard launch.

## Environment and configuration

- [x] Shared kinematic observation configuration uses shape `(5, 5)` with the expected `presence`, `x`, `y`, `vx`, and `vy` features.
- [x] Uses four lanes, 50 traffic vehicles, discrete meta-actions, 40-second episodes, and normalized rewards.
- [x] Uses `highway-fast-v0` for faster CPU simulation; the configuration now uses a simulation frequency of 5 rather than the original plan's 15.
- [x] Shared paths, seeds, default evaluation rates, PPO settings, and derived observation constants are centralised in `config.py`.
- [x] Environment factories exist for baseline training, Agent B safety-oriented training, evaluation, and RGB-array rendering.
- [x] Agent A rejects non-zero training dropout.
- [x] Agent B deliberately differs from Agent A: harsher collision penalty, lower speed reward, lane-change penalty, longer-horizon discount factor, and curriculum training. It is no longer an identical-hyperparameter comparison.
- [~] No standalone automated environment-validation test verifies observation dtype/action count/reset determinism.

## Sensor dropout

- [x] `SensorDropoutWrapper` is a Gymnasium `ObservationWrapper` with a dedicated seeded RNG.
- [x] It copies observations before modification, never masks row 0 (the ego vehicle), masks at most one present non-ego vehicle, and preserves the underlying observation space.
- [x] Dropout probability is validated and can be changed at runtime for curriculum use.
- [~] There is no automated unit-test suite for the wrapper's edge cases or its expected dropout frequency.
- [~] `reset(seed=...)` passes the seed to the wrapped environment but does not currently reseed the wrapper's own RNG, despite its docstring suggesting that it does.

## Training

### Agent A — baseline

- [x] `train.py` provides CLI arguments for agent, timesteps, and seed.
- [x] Agent A trains with zero dropout, saves `models/agent_a_ppo.zip`, writes metadata, and performs a short deterministic post-save sanity check.
- [x] A saved 200,000-timestep Agent A model and metadata are present (seed 42, dropout 0.0).

### Agent B — sensor-robust policy

- [x] Agent B trains through a four-stage dropout curriculum: 0%, 5%, 10%, then 15%.
- [x] Agent B uses its own safety-oriented reward configuration and PPO discount factor.
- [x] Phase checkpoints and resume-from-phase support are implemented locally; checkpoints are intentionally excluded from Git.
- [x] Agent B saves `models/agent_b_ppo.zip`, metadata, and performs the same post-save sanity check.
- [x] A saved 200,000-timestep Agent B model and curriculum metadata are present (seed 42, requested maximum dropout 0.10).
- [~] The code default for Agent B is 500,000 timesteps, but the committed model metadata records 200,000 timesteps.
- [~] Training reward curves are not persisted, so convergence cannot be independently checked from repository artifacts.

## Evaluation and metrics

- [x] Pure Python/NumPy metric functions calculate collision rate, collision-free rate, average and standard-deviation reward, average speed, and average duration.
- [x] Evaluation loads both models, fails clearly if either is absent, evaluates deterministic policies, uses matching seed sequences across agents for each failure rate, records collision/speed/steps, and writes CSV and JSON outputs.
- [x] Evaluation covers the planned failure rates: 0%, 5%, 10%, and 20%.
- [x] Five comparison graphs are implemented and present: collision rate, collision-free rate, reward, speed, and duration.
- [x] `results/summary.json` has all eight agent/failure-rate condition entries and precomputed comparisons for explanation.
- [~] The current saved output is only 8 episode records (one per condition); rerun with `python evaluate.py --episodes 30 --failure-rates 0 0.05 0.10 0.20` before drawing conclusions.
- [~] No automated assertion checks CSV/summary agreement, graph values, or invalid failure-rate input handling.

## Flask backend and dashboard

- [x] The app binds to `127.0.0.1:5000`, serves the root page, and uses Flask's static-file handling.
- [x] Status, evaluation start/progress, results, graph listing/serving, episode-data, explanation start/result, and demo start/step/stop/status endpoints are implemented.
- [x] Long-running evaluation and explanation requests run in background threads, with polling endpoints and a consistent `{success, data, error}` JSON envelope.
- [x] Missing-model, missing-result, duplicate-evaluation, and common demo/explanation errors return clear JSON errors.
- [x] The UI contains the planned header, live dual-agent demo, controls, live stats, evaluation progress, metrics table, graph grid, and explanation panel.
- [x] The frontend polls the appropriate backend endpoints and handles request errors.
- [x] Responsive dark/glass-style visual design, distinct agent panels, controls, transitions, and animations are implemented.
- [~] Browser-level interactive testing of all controls has not been performed as part of this check.

## Live simulation

- [x] `LiveSimulationManager` loads both policies, starts matched-seed RGB-array environments, steps each agent independently, tracks episode data, resets ended episodes, and closes resources on stop.
- [x] Frames are converted to base64 JPEG data URLs for the browser.
- [x] Render failures return a labelled placeholder frame instead of crashing the dashboard.
- [~] A full live-render session was not run during this check, so host-specific Pygame/display behavior still needs manual validation.

## OpenCode result explanation

- [x] `explain_results.py` validates the JSON summary, verifies that OpenCode is available, invokes it with the summary file and a restricted interpretation prompt, handles errors/timeouts, and saves a returned explanation.
- [x] The prompt explicitly prohibits inventing metrics, treating results as real-world safety proof, or using the model to control/train the agents.
- [x] Flask exposes asynchronous endpoints for starting and reading the explanation.
- [~] OpenCode availability/configuration and a successful external explanation run were not verified in this check.
- [~] The integration passes only the summary file to the CLI, but it cannot technically enforce external OpenCode permissions beyond the prompt and command arguments.

## Remaining validation work

- [ ] Run and preserve a full 30-episode-per-condition evaluation.
- [ ] Add automated tests for environment creation, dropout edge cases, metrics, and API error paths.
- [ ] Test the complete browser workflow: live demo at each failure rate, evaluate from UI, stop/restart demo, and explanation polling.
- [ ] Verify the OpenCode integration with a configured provider and retain an explanation artifact if desired.
- [ ] Record/plot training metrics if convergence evidence is required for the report.
- [ ] Review the research comparison wording: Agent B differs in both sensor-dropout curriculum and reward/PPO configuration, so any observed robustness improvement cannot be attributed to dropout training alone.

## Quick commands for remaining checks

```bash
# Full evaluation (30 episodes × 8 conditions)
.venv/bin/python evaluate.py --episodes 30 --failure-rates 0 0.05 0.10 0.20

# Launch dashboard using the included models and saved results
SKIP_TRAIN=1 SKIP_EVAL=1 ./start-ppo.sh
```
