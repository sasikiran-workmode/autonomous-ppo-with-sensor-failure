"""
environment.py — Environment factory functions.
Creates Highway-Env instances with consistent configuration for training and evaluation.
"""
import copy
import gymnasium as gym
import highway_env  # noqa: F401 — registers highway-v0

from config import ENV_CONFIG
from sensor_dropout import SensorDropoutWrapper


def create_env(render_mode=None, env_config_overrides=None):
    """
    Create a base highway-v0 environment with the shared configuration.

    Args:
        render_mode: None for headless, "rgb_array" for frame capture.
        env_config_overrides: dict of keys to override in ENV_CONFIG.

    Returns:
        A Gymnasium environment.
    """
    cfg = copy.deepcopy(ENV_CONFIG)
    if env_config_overrides:
        cfg.update(env_config_overrides)

    # highway-fast-v0 uses the same env but with fewer simulation sub-steps
    # resulting in ~15x speedup on CPU with equivalent learning dynamics.
    env = gym.make("highway-fast-v0", config=cfg, render_mode=render_mode)
    return env


def create_training_env(agent_name: str, dropout_prob: float = 0.0, seed: int = None):
    """
    Create an environment for PPO training.

    Agent A → dropout_prob must be 0.0 (no masking).
    Agent B → dropout_prob is typically 0.10.

    Args:
        agent_name: "A" or "B".
        dropout_prob: Probability of masking one NPC observation per step.
        seed: Random seed passed to env.reset().

    Returns:
        Gymnasium env (possibly wrapped with SensorDropoutWrapper).
    """
    if agent_name.upper() == "A" and dropout_prob > 0.0:
        raise ValueError(
            f"Agent A must have dropout_prob=0.0 but got {dropout_prob}. "
            "Agent A is the baseline — it trains without any sensor masking."
        )

    env = create_env()

    if dropout_prob > 0.0:
        env = SensorDropoutWrapper(env, dropout_prob=dropout_prob, seed=seed)

    return env


def create_training_env_b(dropout_prob: float = 0.0, seed: int = None):
    """
    Create a training environment for Agent B with safety-oriented rewards.

    Uses ENV_CONFIG_B (collision_reward=-5, high_speed_reward=0.2,
    lane_change_reward=-0.05) so the agent learns to prioritize safety.

    Args:
        dropout_prob: Probability of masking one NPC observation per step.
        seed: Random seed.

    Returns:
        Gymnasium env wrapped with SensorDropoutWrapper.
    """
    from config import ENV_CONFIG_B
    env = create_env(env_config_overrides={
        "collision_reward":   ENV_CONFIG_B["collision_reward"],
        "high_speed_reward":  ENV_CONFIG_B["high_speed_reward"],
        "lane_change_reward": ENV_CONFIG_B["lane_change_reward"],
    })

    if dropout_prob > 0.0:
        env = SensorDropoutWrapper(env, dropout_prob=dropout_prob, seed=seed)

    return env


def create_eval_env(failure_rate: float, seed: int = None, render_mode=None):
    """
    Create an environment for policy evaluation.

    Always wraps with SensorDropoutWrapper (even at 0% to keep code consistent).

    Args:
        failure_rate: Per-step probability of masking one NPC observation.
        seed: Random seed.
        render_mode: None or "rgb_array".

    Returns:
        Gymnasium env wrapped with SensorDropoutWrapper.
    """
    env = create_env(render_mode=render_mode)
    env = SensorDropoutWrapper(env, dropout_prob=failure_rate, seed=seed)
    return env


def create_render_env(failure_rate: float = 0.0, seed: int = None):
    """
    Convenience wrapper for the live demo: rgb_array mode + dropout wrapper.
    """
    return create_eval_env(failure_rate=failure_rate, seed=seed, render_mode="rgb_array")
