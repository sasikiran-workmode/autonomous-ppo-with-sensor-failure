"""
live_simulation.py — Manages two concurrent highway-env simulations for the live demo.

Renders real highway-env frames (rgb_array mode) using the system display.
Falls back to a placeholder white image if rendering fails.
"""

import base64
import io
import os
import threading
from typing import Optional, Dict, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

import config
from environment import create_render_env


class LiveSimulationManager:
    """Manages side-by-side simulations for Agent A and Agent B."""

    def __init__(self):
        self._lock = threading.RLock()
        self._running = False
        self._env_a = None
        self._env_b = None
        self._model_a: Optional[PPO] = None
        self._model_b: Optional[PPO] = None
        self._obs_a = None
        self._obs_b = None
        self._episode_reward_a = 0.0
        self._episode_reward_b = 0.0
        self._step_count = 0
        self._failure_rate = 0.0
        self._episode_a = 0
        self._episode_b = 0
        self._collision_a = False
        self._collision_b = False
        self._speed_a = 0.0
        self._speed_b = 0.0

    def _check_models(self):
        missing = []
        if not os.path.exists(config.AGENT_A_PATH + ".zip"):
            missing.append(f"Agent A ({config.AGENT_A_PATH}.zip)")
        if not os.path.exists(config.AGENT_B_PATH + ".zip"):
            missing.append(f"Agent B ({config.AGENT_B_PATH}.zip)")
        if missing:
            raise FileNotFoundError(
                "Required model files not found:\n" +
                "\n".join(f"  - {m}" for m in missing)
            )

    def start(self, failure_rate: float = 0.0, seed: int = 42):
        with self._lock:
            if self._running:
                self.stop()

            self._check_models()

            self._failure_rate = failure_rate
            self._step_count = 0
            self._episode_a = 0
            self._episode_b = 0

            self._model_a = PPO.load(config.AGENT_A_PATH)
            self._model_b = PPO.load(config.AGENT_B_PATH)

            self._env_a = create_render_env(failure_rate=failure_rate, seed=seed)
            self._env_b = create_render_env(failure_rate=failure_rate, seed=seed)

            self._obs_a, _ = self._env_a.reset(seed=seed)
            self._obs_b, _ = self._env_b.reset(seed=seed)

            # Prime the renderer with one render call to initialize it
            try:
                self._env_a.render()
                self._env_b.render()
            except Exception:
                pass

            self._episode_reward_a = 0.0
            self._episode_reward_b = 0.0
            self._collision_a = False
            self._collision_b = False
            self._speed_a = 0.0
            self._speed_b = 0.0

            self._running = True

    def stop(self):
        with self._lock:
            self._running = False
            for env in [self._env_a, self._env_b]:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
            self._env_a = None
            self._env_b = None
            self._model_a = None
            self._model_b = None

    @property
    def is_running(self) -> bool:
        return self._running

    def step(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self._running:
                return None
            try:
                return self._do_step()
            except Exception as e:
                self.stop()
                raise RuntimeError(f"Simulation step failed: {e}") from e

    def _do_step(self) -> Dict[str, Any]:
        # Agent A
        action_a, _ = self._model_a.predict(self._obs_a, deterministic=True)
        self._obs_a, reward_a, term_a, trunc_a, info_a = self._env_a.step(action_a)
        self._episode_reward_a += float(reward_a)
        self._speed_a = float(info_a.get("speed", self._speed_a))
        if info_a.get("crashed", False):
            self._collision_a = True
        if term_a or trunc_a:
            self._episode_a += 1
            self._obs_a, _ = self._env_a.reset()
            self._episode_reward_a = 0.0
            self._collision_a = False

        # Agent B
        action_b, _ = self._model_b.predict(self._obs_b, deterministic=True)
        self._obs_b, reward_b, term_b, trunc_b, info_b = self._env_b.step(action_b)
        self._episode_reward_b += float(reward_b)
        self._speed_b = float(info_b.get("speed", self._speed_b))
        if info_b.get("crashed", False):
            self._collision_b = True
        if term_b or trunc_b:
            self._episode_b += 1
            self._obs_b, _ = self._env_b.reset()
            self._episode_reward_b = 0.0
            self._collision_b = False

        # Render frames — handle None/empty returns
        frame_a = self._safe_render(self._env_a, "Agent A")
        frame_b = self._safe_render(self._env_b, "Agent B")

        self._step_count += 1

        return {
            "frame_a":      frame_a,
            "frame_b":      frame_b,
            "step":         self._step_count,
            "episode_a":    self._episode_a,
            "episode_b":    self._episode_b,
            "reward_a":     round(self._episode_reward_a, 3),
            "reward_b":     round(self._episode_reward_b, 3),
            "speed_a":      round(self._speed_a, 2),
            "speed_b":      round(self._speed_b, 2),
            "collision_a":  self._collision_a,
            "collision_b":  self._collision_b,
            "failure_rate": self._failure_rate,
        }

    def _safe_render(self, env, label: str) -> str:
        """Render and convert to base64; return placeholder on failure."""
        try:
            frame = env.render()
            if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
                return _frame_to_b64(frame)
        except Exception:
            pass
        return _placeholder_frame(label)


def _frame_to_b64(frame: np.ndarray) -> str:
    """Convert RGB numpy array to base64 JPEG string."""
    arr = np.clip(frame, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _placeholder_frame(label: str) -> str:
    """Return a white placeholder image with label text."""
    img = Image.new("RGB", (600, 150), color=(248, 249, 250))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 599, 149], outline=(200, 200, 200), width=1)
    draw.text((300, 75), f"{label} — Loading...", fill=(120, 120, 120), anchor="mm")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
