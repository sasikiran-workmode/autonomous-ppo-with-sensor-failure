"""
config.py — Central configuration for both agents and all scripts.
All shared constants, environment config, and PPO hyperparameters live here.
"""
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR = "models"
RESULTS_DIR = "results"
GRAPHS_DIR = os.path.join(RESULTS_DIR, "graphs")
AGENT_A_PATH = os.path.join(MODEL_DIR, "agent_a_ppo")   # .zip appended by SB3
AGENT_B_PATH = os.path.join(MODEL_DIR, "agent_b_ppo")
AGENT_A_META = os.path.join(MODEL_DIR, "agent_a_meta.json")
AGENT_B_META = os.path.join(MODEL_DIR, "agent_b_meta.json")
# Canonical (post-training baseline) evaluation outputs — written ONCE, never
# overwritten by browser-initiated runs.
EPISODE_METRICS_CSV = os.path.join(RESULTS_DIR, "episode_metrics.csv")
SUMMARY_JSON = os.path.join(RESULTS_DIR, "summary.json")
# User-initiated experiment runs (browser "Run Evaluation" button) go here so
# they can never clobber the canonical baseline.
EXPERIMENTS_DIR = os.path.join(RESULTS_DIR, "experiments")

# ─── Training defaults ────────────────────────────────────────────────────────
DEFAULT_SEED_A = 42
DEFAULT_SEED_B = 42          # Same seed so training variability is the only difference
DEFAULT_TRAINING_TIMESTEPS = 200_000
DEFAULT_DROPOUT_PROB_B = 0.10   # 10 % per decision timestep for Agent B

# ─── Evaluation defaults ─────────────────────────────────────────────────────
DEFAULT_EVAL_EPISODES = 30
EVAL_FAILURE_RATES = [0.0, 0.05, 0.10, 0.20]

# ─── Highway-Env configuration (shared, identical for both agents) ────────────
# vehicles_count inside "observation" controls the observation matrix rows (V).
# vehicles_count at the top level controls how many NPC cars are on the road.
ENV_CONFIG = {
    "observation": {
        "type": "Kinematics",
        # 5 features: presence, x, y, vx, vy
        "features": ["presence", "x", "y", "vx", "vy"],
        # V = 5 → observation shape (5, 5): row 0 = ego, rows 1-4 = nearest NPCs
        "vehicles_count": 5,
        "features_range": {
            "x":  [-100, 100],
            "y":  [-100, 100],
            "vx": [-20,  20],
            "vy": [-20,  20],
        },
        "absolute": False,   # relative coords → better generalisation
        "order": "sorted",   # closest NPCs first
        "normalize": True,   # values in [-1, 1]; note: presence stays 0/1
    },
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 4,
    "vehicles_count": 50,    # NPC traffic density
    "controlled_vehicles": 1,
    "initial_lane_id": None,
    "duration": 40,          # episode length in seconds
    "ego_spacing": 2,
    "vehicles_density": 1,
    # Reward weights
    "collision_reward": -1,
    "right_lane_reward": 0.1,
    "high_speed_reward": 0.4,
    "lane_change_reward": 0,
    "reward_speed_range": [20, 30],  # [m/s]
    "normalize_reward": True,
    "offroad_terminal": False,
    # Rendering
    # Lower simulation_frequency reduces physics substeps → much faster on CPU
    "simulation_frequency": 5,
    "policy_frequency": 1,
    "screen_width": 600,
    "screen_height": 150,
}

# ─── PPO hyperparameters (Agent A — baseline) ────────────────────────────────
PPO_CONFIG = {
    "learning_rate": 5e-4,   # Slightly higher than SB3 default; effective for highway-env
    "n_steps": 512,          # Smaller rollout → more frequent updates, works with n_envs>1
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.8,            # Lower discount → more reactive to immediate dangers
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,        # Small entropy bonus to encourage exploration
    "verbose": 1,
}

# Number of parallel environment workers for faster data collection
N_ENVS = 4

# ─── Agent B: Safety-oriented environment config ─────────────────────────────
# Key differences from ENV_CONFIG:
#   - collision_reward:   -5  (5x harsher crash penalty → agent learns crashing is catastrophic)
#   - high_speed_reward:  0.2 (halved speed incentive → agent prefers caution over speed)
#   - lane_change_reward: -0.05 (small penalty → discourages risky weaving under sensor noise)
import copy as _copy
ENV_CONFIG_B = _copy.deepcopy(ENV_CONFIG)
ENV_CONFIG_B["collision_reward"]   = -5
ENV_CONFIG_B["high_speed_reward"]  = 0.2
ENV_CONFIG_B["lane_change_reward"] = -0.05

# ─── Agent B: PPO hyperparameters ─────────────────────────────────────────────
# Key differences from PPO_CONFIG:
#   - gamma: 0.95 (longer horizon → agent plans further ahead, sees upcoming dangers)
PPO_CONFIG_B = dict(PPO_CONFIG)
PPO_CONFIG_B["gamma"] = 0.95

# Agent B default training timesteps (needs more to converge under noise)
AGENT_B_TIMESTEPS = 500_000

# Curriculum dropout schedule: (fraction_of_training, dropout_prob)
# Agent first learns to drive, then gradually adapts to sensor noise
CURRICULUM_SCHEDULE = [
    (0.25, 0.00),   # Phase 1: 0-25% — learn clean driving
    (0.50, 0.05),   # Phase 2: 25-50% — introduce light noise
    (0.75, 0.10),   # Phase 3: 50-75% — match evaluation condition
    (1.00, 0.15),   # Phase 4: 75-100% — over-train slightly for robustness
]

# Observation shape derived from config (used for validation)
OBS_VEHICLES = ENV_CONFIG["observation"]["vehicles_count"]   # 5
OBS_FEATURES = len(ENV_CONFIG["observation"]["features"])    # 5
OBS_SHAPE = (OBS_VEHICLES, OBS_FEATURES)                    # (5, 5)
EGO_ROW = 0       # Ego vehicle is always row 0
PRESENCE_COL = 0  # "presence" is always column 0
