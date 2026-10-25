"""
evaluate.py — Evaluate both trained agents under 4 sensor-failure rates.

Runs 8 conditions (2 agents × 4 failure rates), records per-episode metrics,
saves results to CSV + JSON, and generates comparison graphs.    Usage:
    python evaluate.py                        # canonical 30-episode baseline (frozen once written)
    python evaluate.py --episodes 30 --failure-rates 0 0.05 0.10 0.20
    python evaluate.py --force               # regenerate the canonical baseline
    python evaluate.py --episodes 10 --failure-rates 0 0.10 --force   # quick test
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


def generate_graphs(summary, failure_rates, graphs_dir=None):
    """Generate the 5 comparison charts.

    Args:
        graphs_dir: Where to write the PNGs. Defaults to the canonical graphs
            directory; experiment runs pass their own per-run directory.
    """
    if graphs_dir is None:
        graphs_dir = config.GRAPHS_DIR
    os.makedirs(graphs_dir, exist_ok=True)

    _bar_chart(
        title="Collision Rate by Agent and Sensor Failure Probability",
        ylabel="Collision Rate (%)",
        metric_key="collision_rate",
        summary=summary,
        failure_rates=failure_rates,
        save_path=os.path.join(graphs_dir, "collision_rate.png"),
        pct=True,
    )

    _bar_chart(
        title="Average Episode Reward by Agent and Sensor Failure Probability",
        ylabel="Average Cumulative Reward",
        metric_key="avg_reward",
        summary=summary,
        failure_rates=failure_rates,
        save_path=os.path.join(graphs_dir, "avg_reward.png"),
        pct=False,
    )

    _bar_chart(
        title="Average Speed by Agent and Sensor Failure Probability",
        ylabel="Average Speed (m/s)",
        metric_key="avg_speed",
        summary=summary,
        failure_rates=failure_rates,
        save_path=os.path.join(graphs_dir, "avg_speed.png"),
        pct=False,
    )

    _bar_chart(
        title="Average Episode Duration by Agent and Sensor Failure Probability",
        ylabel="Average Steps per Episode",
        metric_key="avg_duration",
        summary=summary,
        failure_rates=failure_rates,
        save_path=os.path.join(graphs_dir, "avg_duration.png"),
        pct=False,
    )

    _bar_chart(
        title="Collision-Free Rate by Agent and Sensor Failure Probability",
        ylabel="Collision-Free Rate (%)",
        metric_key="collision_free_rate",
        summary=summary,
        failure_rates=failure_rates,
        save_path=os.path.join(graphs_dir, "collision_free_rate.png"),
        pct=True,
    )


# ─── Main evaluation loop ─────────────────────────────────────────────────────

def run_evaluation(n_episodes: int, failure_rates: list, progress_callback=None,
                   baseline: bool = False, force: bool = False):
    """
    Run the full 8-condition evaluation.

    Two output modes:
      • baseline=True  → canonical post-training baseline. Writes the canonical
        summary.json / episode_metrics.csv / graphs. These are the official
        results for the trained models and are written ONCE: an existing
        canonical dataset is never overwritten unless force=True.
      • baseline=False → user experiment (e.g. triggered from the browser).
        Writes everything to results/experiments/<run_id>/ so it can never
        clobber the canonical baseline.

    Args:
        n_episodes: Episodes per condition.
        failure_rates: List of float dropout probabilities to test.
        progress_callback: Optional callable(current, total, message) for UI updates.
        baseline: Whether this is the canonical post-training evaluation.
        force: Only for baseline=True — overwrite the existing canonical results.

    Returns:
        (all_episodes: list, summary: dict)
    """
    n_episodes = int(n_episodes)
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1 (got {n_episodes}).")
    failure_rates = sorted(float(fr) for fr in failure_rates)

    # ── Decide where outputs go ───────────────────────────────────────────────
    if baseline:
        summary_path = config.SUMMARY_JSON
        csv_path = config.EPISODE_METRICS_CSV
        graphs_dir = config.GRAPHS_DIR
        out_dir = config.RESULTS_DIR

        if not force and os.path.exists(summary_path):
            raise RuntimeError(
                f"Canonical baseline results already exist at {summary_path}.\n"
                "They are frozen and are never overwritten by evaluation runs.\n"
                "Pass force=True to explicitly replace them."
            )
    else:
        run_id = time.strftime("%Y%m%d-%H%M%S")
        out_dir = os.path.join(config.EXPERIMENTS_DIR, run_id)
        summary_path = os.path.join(out_dir, "summary.json")
        csv_path = os.path.join(out_dir, "episode_metrics.csv")
        graphs_dir = os.path.join(out_dir, "graphs")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

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
    df.to_csv(csv_path, index=False)
    print(f"\n[evaluate] Episode data → {csv_path} ({len(df)} rows)")

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

    with open(summary_path, "w") as f:
        json.dump(full_summary, f, indent=2, default=str)
    print(f"[evaluate] Summary → {summary_path}")

    # ── Generate graphs ─────────────────────────────────────────────────────
    if baseline and not force:
        # Canonical graphs are frozen: generate once, only if missing.
        _expected_graphs = [
            "collision_rate.png", "collision_free_rate.png",
            "avg_reward.png", "avg_speed.png", "avg_duration.png",
        ]
        _all_present = all(
            os.path.exists(os.path.join(config.GRAPHS_DIR, fn))
            for fn in _expected_graphs
        )
        if _all_present:
            print("[evaluate] Canonical graphs already exist — skipping regeneration.")
        else:
            generate_graphs(summary, failure_rates)
    else:
        # Experiments (and force regeneration) write their own graphs.
        generate_graphs(summary, failure_rates, graphs_dir=graphs_dir)

    if progress_callback:
        progress_callback(total_conditions, total_conditions, "Done!")

    print(f"[evaluate] Outputs written to → {out_dir}")
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
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite the existing canonical baseline results (summary.json, "
             "episode_metrics.csv, graphs). Without this flag the CLI refuses "
             "to touch existing canonical results."
    )
    args = parser.parse_args()

    run_evaluation(
        n_episodes=args.episodes,
        failure_rates=args.failure_rates,
        baseline=True,
        force=args.force,
    )
