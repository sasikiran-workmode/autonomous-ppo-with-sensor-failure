"""
train.py — Training script for Agent A (baseline) and Agent B (sensor-failure robust).

Agent A: Standard PPO with ENV_CONFIG and PPO_CONFIG (200k steps, no dropout).
Agent B: Curriculum PPO with ENV_CONFIG_B (safety rewards) and PPO_CONFIG_B (gamma=0.95).
         Trains in 4 phases with increasing sensor dropout (0% -> 5% -> 10% -> 15%).

Usage:
    python train.py --agent A --timesteps 200000
    python train.py --agent B --timesteps 500000
    python train.py --agent B --timesteps 500000 --dropout-prob 0.15
"""

import argparse
import json
import os
import sys
import time

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor

import config
from environment import create_training_env, create_training_env_b, create_env


# ── Helpers ──────────────────────────────────────────────────────────────────

def ensure_dirs():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)


def save_meta(agent_name: str, timesteps: int, seed: int, dropout_prob: float,
              extra: dict = None):
    meta = {
        "agent": agent_name,
        "timesteps": timesteps,
        "seed": seed,
        "dropout_prob": dropout_prob,
        "env_config": config.ENV_CONFIG if agent_name == "A" else {
            **config.ENV_CONFIG,
            "collision_reward":   config.ENV_CONFIG_B["collision_reward"],
            "high_speed_reward":  config.ENV_CONFIG_B["high_speed_reward"],
            "lane_change_reward": config.ENV_CONFIG_B["lane_change_reward"],
        },
        "ppo_config": config.PPO_CONFIG if agent_name == "A" else config.PPO_CONFIG_B,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra:
        meta.update(extra)
    path = config.AGENT_A_META if agent_name == "A" else config.AGENT_B_META
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[train] Metadata saved -> {path}")


def build_ppo(env, seed: int, ppo_cfg: dict) -> PPO:
    """Instantiate PPO with the given hyperparameters."""
    kwargs = dict(ppo_cfg)
    kwargs.pop("verbose", None)
    return PPO(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        verbose=1,
        **kwargs,
    )


def make_vec_env(env_factory, n_envs: int, seed: int):
    """Create a vectorized environment for parallel data collection."""
    def make_env(rank):
        def _init():
            return env_factory(seed=seed + rank)
        return _init

    try:
        vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    except Exception:
        print("[train] SubprocVecEnv failed, falling back to DummyVecEnv")
        vec_env = DummyVecEnv([make_env(0)])

    return VecMonitor(vec_env)


# ── Agent A training ─────────────────────────────────────────────────────────

def train_agent_a(timesteps: int, seed: int):
    """Standard PPO training for the baseline agent."""
    ensure_dirs()

    print(f"\n{'='*60}")
    print(f"  Training Agent A (Baseline)")
    print(f"  Timesteps    : {timesteps:,}")
    print(f"  Seed         : {seed}")
    print(f"  Dropout prob : 0.0")
    print(f"  Parallel envs: {config.N_ENVS}")
    print(f"{'='*60}\n")

    def factory(seed=seed):
        return create_training_env(agent_name="A", dropout_prob=0.0, seed=seed)

    vec_env = make_vec_env(factory, config.N_ENVS, seed)
    model = build_ppo(vec_env, seed=seed, ppo_cfg=config.PPO_CONFIG)

    t0 = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    elapsed = time.time() - t0

    print(f"\n[train] Agent A complete in {elapsed:.1f}s")
    model.save(config.AGENT_A_PATH)
    print(f"[train] Model saved -> {config.AGENT_A_PATH}.zip")
    save_meta("A", timesteps, seed, 0.0)

    vec_env.close()
    _sanity_check(config.AGENT_A_PATH, seed)
    return config.AGENT_A_PATH


# ── Agent B training (curriculum) ─────────────────────────────────────────────

def train_agent_b(timesteps: int, seed: int, max_dropout: float, start_phase: int = 1):
    """
    Curriculum training for the sensor-robust agent.

    4 phases with increasing dropout:
      Phase 1 (0-25%):   dropout = 0%   -- learn basic driving with safety rewards
      Phase 2 (25-50%):  dropout = 5%   -- introduce light sensor noise
      Phase 3 (50-75%):  dropout = 10%  -- match evaluation conditions
      Phase 4 (75-100%): dropout = 15%  -- slightly over-train for robustness

    A checkpoint is saved after every phase in models/checkpoints/.
    Use --start-phase N to resume from a previously completed phase checkpoint.
    """
    ensure_dirs()
    os.makedirs(os.path.join(config.MODEL_DIR, "checkpoints"), exist_ok=True)
    schedule = config.CURRICULUM_SCHEDULE

    print(f"\n{'='*60}")
    print(f"  Training Agent B (Sensor-Robust, Curriculum)")
    print(f"  Total timesteps : {timesteps:,}")
    print(f"  Seed            : {seed}")
    print(f"  Max dropout     : {max_dropout}")
    print(f"  Start phase     : {start_phase}")
    print(f"  Parallel envs   : {config.N_ENVS}")
    print(f"  Reward shaping  : collision=-5, speed=0.2, lane_change=-0.05")
    print(f"  Gamma           : {config.PPO_CONFIG_B['gamma']}")
    print(f"  Schedule        : {schedule}")
    print(f"{'='*60}\n")

    model = None
    total_trained = 0

    for phase_idx, (frac, dropout_prob) in enumerate(schedule):
        phase_num = phase_idx + 1

        # Calculate steps for this phase
        if phase_idx == 0:
            prev_frac = 0.0
        else:
            prev_frac = schedule[phase_idx - 1][0]
        phase_steps = int((frac - prev_frac) * timesteps)

        if phase_steps <= 0:
            continue

        # Skip completed phases if resuming
        if phase_num < start_phase:
            ckpt = os.path.join(config.MODEL_DIR, "checkpoints",
                                f"agent_b_phase{phase_num}.zip")
            if not os.path.exists(ckpt):
                print(f"[train] ERROR: Cannot skip Phase {phase_num} -- "
                      f"checkpoint not found: {ckpt}")
                sys.exit(1)
            print(f"[train] Skipping Phase {phase_num} (checkpoint found).")
            total_trained += phase_steps
            # Load the latest skipped checkpoint as base model for next phase
            vec_env_tmp = make_vec_env(
                lambda seed=seed, dp=dropout_prob: create_training_env_b(dropout_prob=dp, seed=seed),
                config.N_ENVS, seed
            )
            model = PPO.load(ckpt, env=vec_env_tmp)
            vec_env_tmp.close()
            continue

        print(f"\n--- Phase {phase_num}/{len(schedule)}: "
              f"dropout={dropout_prob*100:.0f}%, steps={phase_steps:,} ---")

        # Create environments for this phase
        def factory(seed=seed, dp=dropout_prob):
            return create_training_env_b(dropout_prob=dp, seed=seed)

        vec_env = make_vec_env(factory, config.N_ENVS, seed)

        if model is None:
            # First active phase: create new model
            model = build_ppo(vec_env, seed=seed, ppo_cfg=config.PPO_CONFIG_B)
        else:
            model.set_env(vec_env)

        t0 = time.time()
        model.learn(total_timesteps=phase_steps, progress_bar=False,
                    reset_num_timesteps=False)
        elapsed = time.time() - t0

        total_trained += phase_steps
        print(f"  Phase {phase_num} done in {elapsed:.1f}s "
              f"(total: {total_trained:,}/{timesteps:,})")

        # Save phase checkpoint so training can be resumed later
        ckpt_path = os.path.join(config.MODEL_DIR, "checkpoints",
                                 f"agent_b_phase{phase_num}")
        model.save(ckpt_path)
        print(f"  Checkpoint saved -> {ckpt_path}.zip")

        vec_env.close()

    # Save final model
    print(f"\n[train] Agent B curriculum training complete.")
    model.save(config.AGENT_B_PATH)
    print(f"[train] Model saved -> {config.AGENT_B_PATH}.zip")
    save_meta("B", timesteps, seed, max_dropout, extra={
        "training_type": "curriculum",
        "curriculum_schedule": schedule,
    })

    _sanity_check(config.AGENT_B_PATH, seed)
    return config.AGENT_B_PATH


# ── Sanity check ─────────────────────────────────────────────────────────────

def _sanity_check(model_path: str, seed: int):
    """Load model and run 5 deterministic steps to verify it works."""
    print("[train] Running quick sanity check (5 deterministic steps)...")
    check_env = create_env()
    loaded = PPO.load(model_path, env=check_env)
    obs, _ = check_env.reset(seed=seed)
    for _ in range(5):
        action, _ = loaded.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = check_env.step(action)
        if terminated or truncated:
            obs, _ = check_env.reset()
    check_env.close()
    print("[train] Sanity check passed.\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a PPO driving agent on highway-v0."
    )
    parser.add_argument(
        "--agent", required=True, choices=["A", "B", "a", "b"],
        help="Which agent to train: A (baseline) or B (sensor-failure robust)."
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Total training timesteps. Default: 200k for A, 500k for B."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed. Defaults to 42."
    )
    parser.add_argument(
        "--dropout-prob", type=float, default=None,
        dest="dropout_prob",
        help=(
            f"Max dropout probability for Agent B (default: {config.DEFAULT_DROPOUT_PROB_B}). "
            "Must not be set for Agent A."
        ),
    )

    parser.add_argument(
        "--start-phase", type=int, default=1, dest="start_phase",
        choices=[1, 2, 3, 4],
        help=(
            "Resume from this phase number (1-4). Requires phase checkpoints in "
            "models/checkpoints/agent_b_phaseN.zip. Default: 1 (start fresh)."
        ),
    )

    args = parser.parse_args()
    agent = args.agent.upper()

    # Default seeds
    if args.seed is None:
        args.seed = config.DEFAULT_SEED_A if agent == "A" else config.DEFAULT_SEED_B

    if agent == "A":
        if args.dropout_prob is not None and args.dropout_prob > 0.0:
            print("[error] Agent A must train without dropout.")
            sys.exit(1)
        ts = args.timesteps or config.DEFAULT_TRAINING_TIMESTEPS
        train_agent_a(timesteps=ts, seed=args.seed)
    else:
        dp = args.dropout_prob if args.dropout_prob is not None else config.DEFAULT_DROPOUT_PROB_B
        ts = args.timesteps or config.AGENT_B_TIMESTEPS
        train_agent_b(timesteps=ts, seed=args.seed, max_dropout=dp,
                      start_phase=args.start_phase)
