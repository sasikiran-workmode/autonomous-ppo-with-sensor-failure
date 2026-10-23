"""
evaluate.py — Evaluate both trained agents under 4 sensor-failure rates.

Runs 8 conditions (2 agents × 4 failure rates), records per-episode metrics,
saves results to CSV + JSON, and generates comparison graphs.

Usage:
    python evaluate.py
    python evaluate.py --episodes 30 --failure-rates 0 0.05 0.10 0.20
    python evaluate.py --episodes 10 --failure-rates 0 0.10   # quick test
"""

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless rendering — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from stable_baselines3 import PPO

import config
from environment import create_eval_env
from metrics import aggregate, build_summary, compute_comparisons


# ─── Helpers ──────────────────────────────────────────────────────────────────

def ensure_dirs():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.GRAPHS_DIR, exist_ok=True)


def load_models():
    """Load both agents; exit with clear message if a model is missing."""
    missing = []
    for path, label in [
        (config.AGENT_A_PATH + ".zip", "Agent A"),
        (config.AGENT_B_PATH + ".zip", "Agent B"),
    ]:
        if not os.path.exists(path):
            missing.append(f"  • {label}: {path}")

    if missing:
        print("\n[evaluate] ERROR — Required model files not found:\n")
        for m in missing:
            print(m)
        print(
            "\nPlease train the missing agents first:\n"
            "  python train.py --agent A --timesteps 200000\n"
            "  python train.py --agent B --timesteps 200000\n"
        )
        sys.exit(1)

    print("[evaluate] Loading Agent A...")
    model_a = PPO.load(config.AGENT_A_PATH)
    print("[evaluate] Loading Agent B...")
    model_b = PPO.load(config.AGENT_B_PATH)
    return model_a, model_b


def run_episode(model, env, episode_seed: int):
    """
    Run one evaluation episode.

    Returns:
        dict with keys: cumulative_reward, avg_speed, num_steps, collision.
    """
    obs, info = env.reset(seed=episode_seed)
    cumulative_reward = 0.0
    speeds = []
    steps = 0
    collision = False

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        cumulative_reward += float(reward)
        steps += 1

        # Extract speed from info dict (highway-env provides "speed" in info)
        speed = info.get("speed", 0.0)
        if speed == 0.0:
            # Fallback: denormalize vx from observation row 0
            # vx normalized range [-20, 20]; obs[0, 3] is normalised vx
            # Use abs so negative values (backward) still count
            speed = abs(float(obs[0, 3])) * 20.0  # rough de-norm
        speeds.append(float(speed))

        if info.get("crashed", False):
            collision = True

        if terminated or truncated:
            break

    return {
        "cumulative_reward": cumulative_reward,
        "avg_speed":         float(np.mean(speeds)) if speeds else 0.0,
        "num_steps":         steps,
        "collision":         collision,
    }


# ─── Graphing ─────────────────────────────────────────────────────────────────

AGENT_COLORS = {
    "A": "#1565C0",   # Blue  — Agent A
    "B": "#2E7D32",   # Green — Agent B
}

STYLE = {
    "figure.facecolor":   "white",
    "axes.facecolor":     "#F8F9FA",
    "axes.edgecolor":     "#CED4DA",
    "axes.labelcolor":    "#212529",
    "xtick.color":        "#495057",
    "ytick.color":        "#495057",
    "text.color":         "#212529",
    "grid.color":         "#DEE2E6",
    "grid.linestyle":     "--",
    "grid.alpha":         0.7,
    "legend.facecolor":   "white",
    "legend.edgecolor":   "#CED4DA",
    "font.family":        "DejaVu Sans",
}


def apply_style():
    for k, v in STYLE.items():
        plt.rcParams[k] = v


def _bar_chart(title, ylabel, metric_key, summary, failure_rates, save_path, pct=False, note=None):
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(failure_rates))
    width = 0.35
    mult = 100 if pct else 1

    vals_a = [summary.get(f"A_{fr}", {}).get(metric_key, 0) * mult for fr in failure_rates]
    vals_b = [summary.get(f"B_{fr}", {}).get(metric_key, 0) * mult for fr in failure_rates]

    bars_a = ax.bar(x - width / 2, vals_a, width, label="Agent A (Baseline)",
                    color=AGENT_COLORS["A"], alpha=0.85, zorder=3)
    bars_b = ax.bar(x + width / 2, vals_b, width, label="Agent B (Sensor-Robust)",
                    color=AGENT_COLORS["B"], alpha=0.85, zorder=3)

    for bar in bars_a:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + (0.3 if pct else h*0.01+0.05),
                f"{h:.1f}{'%' if pct else ''}", ha="center", va="bottom",
                fontsize=8.5, color="#212529")
    for bar in bars_b:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + (0.3 if pct else h*0.01+0.05),
                f"{h:.1f}{'%' if pct else ''}", ha="center", va="bottom",
                fontsize=8.5, color="#212529")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=14, color="#212529")
    ax.set_xlabel("Sensor Failure Rate", fontsize=11, color="#495057")
    ax.set_ylabel(ylabel, fontsize=11, color="#495057")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{fr*100:.0f}%" for fr in failure_rates])
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if note:
        fig.text(0.5, -0.03, note, ha="center", fontsize=8.5, color="#6C757D", style="italic")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[evaluate] Graph saved → {save_path}")


def generate_graphs(summary, failure_rates):
    ensure_dirs()

    _bar_chart(
        title="Collision Rate by Agent and Sensor Failure Probability",
        ylabel="Collision Rate (%)",
        metric_key="collision_rate",
        summary=summary,
        failure_rates=failure_rates,
        save_path=os.path.join(config.GRAPHS_DIR, "collision_rate.png"),
        pct=True,
    )

    _bar_chart(
        title="Average Episode Reward by Agent and Sensor Failure Probability",
        ylabel="Average Cumulative Reward",
        metric_key="avg_reward",
        summary=summary,
        failure_rates=failure_rates,
        save_path=os.path.join(config.GRAPHS_DIR, "avg_reward.png"),
        pct=False,
    )

    _bar_chart(
        title="Average Speed by Agent and Sensor Failure Probability",
        ylabel="Average Speed (m/s)",
        metric_key="avg_speed",
        summary=summary,
        failure_rates=failure_rates,
        save_path=os.path.join(config.GRAPHS_DIR, "avg_speed.png"),
        pct=False,
    )

    _bar_chart(
        title="Average Episode Duration by Agent and Sensor Failure Probability",
        ylabel="Average Steps per Episode",
        metric_key="avg_duration",
        summary=summary,
        failure_rates=failure_rates,
        save_path=os.path.join(config.GRAPHS_DIR, "avg_duration.png"),
        pct=False,
    )

    _bar_chart(
        title="Collision-Free Rate by Agent and Sensor Failure Probability",
        ylabel="Collision-Free Rate (%)",
        metric_key="collision_free_rate",
        summary=summary,
        failure_rates=failure_rates,
        save_path=os.path.join(config.GRAPHS_DIR, "collision_free_rate.png"),
        pct=True,
    )


# ─── Main evaluation loop ─────────────────────────────────────────────────────

def run_evaluation(n_episodes: int, failure_rates: list, progress_callback=None):
    """
    Run the full 8-condition evaluation.

    Args:
        n_episodes: Episodes per condition.
        failure_rates: List of float dropout probabilities to test.
        progress_callback: Optional callable(current, total, message) for UI updates.

    Returns:
        (all_episodes: list, summary: dict)
    """
    ensure_dirs()
    model_a, model_b = load_models()

    agents = [("A", model_a), ("B", model_b)]
    all_episodes = []

    total_conditions = len(agents) * len(failure_rates)
    condition_idx = 0

    for fr in failure_rates:
        for agent_name, model in agents:
            condition_idx += 1
            label = f"Agent {agent_name} @ {fr*100:.0f}% failure"
            print(f"\n[evaluate] Condition {condition_idx}/{total_conditions}: {label}")

            if progress_callback:
                progress_callback(
                    condition_idx - 1, total_conditions,
                    f"Running {label} ({n_episodes} episodes)..."
                )

            # Each condition gets its own env instance
            env = create_eval_env(failure_rate=fr)

            for ep in range(n_episodes):
                ep_seed = 1000 * int(fr * 100) + ep  # deterministic per (fr, episode)
                try:
                    result = run_episode(model, env, episode_seed=ep_seed)
                    record = {
                        "agent":             agent_name,
                        "failure_rate":      fr,
                        "episode_idx":       ep,
                        "seed":              ep_seed,
                        "cumulative_reward": result["cumulative_reward"],
                        "avg_speed":         result["avg_speed"],
                        "num_steps":         result["num_steps"],
                        "collision":         result["collision"],
                    }
                    all_episodes.append(record)
                except Exception as e:
                    print(f"[evaluate] WARNING: Episode {ep} failed ({e}), skipping.")
                    traceback.print_exc()

            env.close()
            cond_eps = [r for r in all_episodes
                        if r["agent"] == agent_name and r["failure_rate"] == fr]
            agg = aggregate(cond_eps)
            print(f"  collision_rate={agg['collision_rate']:.2%}  "
                  f"avg_reward={agg['avg_reward']:.3f}  "
                  f"avg_speed={agg['avg_speed']:.2f} m/s  "
                  f"avg_steps={agg['avg_duration']:.1f}")

    if progress_callback:
        progress_callback(total_conditions, total_conditions, "Computing metrics...")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_episodes)
    df.to_csv(config.EPISODE_METRICS_CSV, index=False)
    print(f"\n[evaluate] Episode data → {config.EPISODE_METRICS_CSV} ({len(df)} rows)")

    # ── Compute summary ───────────────────────────────────────────────────────
    summary = build_summary(all_episodes)
    comparisons = compute_comparisons(summary)

    full_summary = {
        "experiment_config": {
            "n_episodes":    n_episodes,
            "failure_rates": failure_rates,
            "env_config":    config.ENV_CONFIG,
            "ppo_config":    config.PPO_CONFIG,
            "evaluated_at":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "results":      summary,
        "comparisons":  comparisons,
    }

    with open(config.SUMMARY_JSON, "w") as f:
        json.dump(full_summary, f, indent=2, default=str)
    print(f"[evaluate] Summary → {config.SUMMARY_JSON}")

    # ── Generate graphs ───────────────────────────────────────────────────────
    generate_graphs(summary, failure_rates)

    if progress_callback:
        progress_callback(total_conditions, total_conditions, "Done!")

    return all_episodes, full_summary


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate both PPO agents under multiple sensor-failure rates."
    )
    parser.add_argument(
        "--episodes", type=int, default=config.DEFAULT_EVAL_EPISODES,
        help=f"Episodes per condition (default: {config.DEFAULT_EVAL_EPISODES})."
    )
    parser.add_argument(
        "--failure-rates", nargs="+", type=float,
        default=config.EVAL_FAILURE_RATES,
        dest="failure_rates",
        help="List of sensor-failure probabilities to test (default: 0 0.05 0.10 0.20)."
    )
    args = parser.parse_args()

    run_evaluation(n_episodes=args.episodes, failure_rates=args.failure_rates)
