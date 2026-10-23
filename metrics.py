"""
metrics.py — Pure-Python metric calculations for evaluation results.
No LLM involvement. All values are derived from recorded episode data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any


# ─── Episode record structure ─────────────────────────────────────────────────
# Each episode produces one record:
# {
#     "agent":              "A" or "B",
#     "failure_rate":       float (0.0, 0.05, 0.10, 0.20),
#     "episode_idx":        int,
#     "seed":               int,
#     "cumulative_reward":  float,
#     "avg_speed":          float,   # m/s, mean across episode steps
#     "num_steps":          int,
#     "collision":          bool,
# }


def calculate_collision_rate(episodes: List[Dict]) -> float:
    """Fraction of episodes containing a collision (0.0 – 1.0)."""
    if not episodes:
        return 0.0
    collisions = sum(1 for e in episodes if e.get("collision", False))
    return collisions / len(episodes)


def calculate_collision_free_rate(episodes: List[Dict]) -> float:
    """Fraction of episodes completed without any collision."""
    return 1.0 - calculate_collision_rate(episodes)


def calculate_avg_reward(episodes: List[Dict]) -> float:
    """Mean cumulative reward per episode."""
    if not episodes:
        return 0.0
    return float(np.mean([e["cumulative_reward"] for e in episodes]))


def calculate_std_reward(episodes: List[Dict]) -> float:
    """Std dev of cumulative reward per episode."""
    if not episodes:
        return 0.0
    return float(np.std([e["cumulative_reward"] for e in episodes]))


def calculate_avg_speed(episodes: List[Dict]) -> float:
    """Mean vehicle speed (m/s) averaged across all episodes."""
    if not episodes:
        return 0.0
    return float(np.mean([e["avg_speed"] for e in episodes]))


def calculate_avg_duration(episodes: List[Dict]) -> float:
    """Mean number of decision steps per episode."""
    if not episodes:
        return 0.0
    return float(np.mean([e["num_steps"] for e in episodes]))


def aggregate(episodes: List[Dict]) -> Dict[str, Any]:
    """
    Compute all metrics for a list of episode records from a single condition.

    Returns:
        dict with keys: collision_rate, collision_free_rate, avg_reward,
        std_reward, avg_speed, avg_duration, n_episodes.
    """
    return {
        "n_episodes":           len(episodes),
        "collision_rate":       round(calculate_collision_rate(episodes), 4),
        "collision_free_rate":  round(calculate_collision_free_rate(episodes), 4),
        "avg_reward":           round(calculate_avg_reward(episodes), 4),
        "std_reward":           round(calculate_std_reward(episodes), 4),
        "avg_speed":            round(calculate_avg_speed(episodes), 4),
        "avg_duration":         round(calculate_avg_duration(episodes), 2),
    }


def build_summary(all_episodes: List[Dict]) -> Dict[str, Any]:
    """
    Group episodes by (agent, failure_rate) and compute aggregate metrics.

    Returns:
        {
            "A_0.0":  {...metrics...},
            "A_0.05": {...},
            ...
        }
    """
    df = pd.DataFrame(all_episodes)
    summary = {}
    for (agent, fr), group in df.groupby(["agent", "failure_rate"]):
        key = f"{agent}_{fr}"
        summary[key] = aggregate(group.to_dict("records"))
    return summary


def percentage_change(old: float, new: float) -> float:
    """Compute (new - old) / old * 100 safely (returns 0 if old == 0)."""
    if old == 0:
        return 0.0
    return round((new - old) / abs(old) * 100, 2)


def compute_comparisons(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pre-compute numerical differences for OpenCode to interpret.
    Returns delta tables so the LLM never invents numbers.
    """
    comparisons = {}
    failure_rates = [0.0, 0.05, 0.10, 0.20]
    for fr in failure_rates:
        key_a = f"A_{fr}"
        key_b = f"B_{fr}"
        if key_a not in summary or key_b not in summary:
            continue
        ma = summary[key_a]
        mb = summary[key_b]
        comparisons[f"delta_{fr}"] = {
            "failure_rate": fr,
            "collision_rate_delta":  round(mb["collision_rate"] - ma["collision_rate"], 4),
            "avg_reward_delta":      round(mb["avg_reward"] - ma["avg_reward"], 4),
            "avg_speed_delta":       round(mb["avg_speed"] - ma["avg_speed"], 4),
            "collision_rate_pct_change": percentage_change(ma["collision_rate"], mb["collision_rate"]),
            "avg_reward_pct_change":     percentage_change(ma["avg_reward"], mb["avg_reward"]),
        }
    return comparisons
