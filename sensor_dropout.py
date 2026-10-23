"""
sensor_dropout.py — Gymnasium ObservationWrapper that simulates sensor failures.

How it works:
  • At each decision step, with probability `dropout_prob`, ONE nearby vehicle's
    observation row is replaced entirely with zeros (presence + all features).
  • The ego vehicle (row 0) is NEVER masked.
  • If no nearby vehicle has presence=1 (empty traffic), the observation passes
    through unchanged.
  • The underlying simulation state is untouched: vehicles still exist, move,
    and trigger rewards/collisions normally. Only the agent's view is altered.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SensorDropoutWrapper(gym.ObservationWrapper):
    """
    Simulates per-timestep sensor failures on kinematic observations.

    Observation shape: (V, F) where
        V = vehicles_count (default 5: ego + 4 NPC slots)
        F = number of features (default 5: presence, x, y, vx, vy)

    Args:
        env: The wrapped Gymnasium environment.
        dropout_prob: Probability (0–1) that a dropout event occurs each step.
        seed: Optional integer seed for the internal RNG (for reproducibility).
    """

    def __init__(self, env: gym.Env, dropout_prob: float = 0.10, seed: int = None):
        super().__init__(env)
        if not 0.0 <= dropout_prob <= 1.0:
            raise ValueError(f"dropout_prob must be in [0, 1], got {dropout_prob}")

        self.dropout_prob = dropout_prob
        self._rng = np.random.RandomState(seed)

        # The observation_space is identical to the inner env's — shape/dtype preserved.
        # SB3 reads observation_space to determine the network input size.
        self.observation_space = env.observation_space

    # ------------------------------------------------------------------
    def observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Apply sensor dropout to a kinematic observation matrix.

        Args:
            obs: numpy array of shape (V, F), dtype float32/float64.

        Returns:
            The (possibly modified) observation array.
        """
        # Always work on a copy so we never mutate the env's internal buffer.
        obs = obs.copy()

        # Skip if no dropout or random coin flip says "no event this step".
        if self.dropout_prob <= 0.0:
            return obs
        if self._rng.random() >= self.dropout_prob:
            return obs

        # Row 0 is ego — never masked.
        # Eligible rows: rows 1..V-1 where presence feature (col 0) == 1.
        V = obs.shape[0]
        eligible = [i for i in range(1, V) if obs[i, 0] > 0.5]

        if not eligible:
            # No nearby vehicles currently in observation → nothing to hide.
            return obs

        # Choose one eligible row at random and zero it out entirely.
        chosen = self._rng.choice(eligible)
        obs[chosen, :] = 0.0

        return obs

    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        """Pass-through reset; re-seed dropout RNG if seed provided."""
        return self.env.reset(**kwargs)

    # ------------------------------------------------------------------
    def set_dropout_prob(self, prob: float):
        """Allow runtime change of dropout probability (e.g. during evaluation)."""
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"dropout_prob must be in [0, 1], got {prob}")
        self.dropout_prob = prob


def get_curriculum_dropout(current_step: int, total_steps: int, schedule: list) -> float:
    """
    Return the dropout probability for the current training step based on a
    curriculum schedule.

    Args:
        current_step: Current training timestep.
        total_steps: Total training timesteps.
        schedule: List of (fraction, dropout_prob) tuples, sorted by fraction.

    Returns:
        The dropout probability for this step.
    """
    if total_steps <= 0:
        return 0.0
    frac = current_step / total_steps
    prob = 0.0
    for cutoff, dp in schedule:
        if frac <= cutoff:
            prob = dp
            break
    else:
        prob = schedule[-1][1]
    return prob
