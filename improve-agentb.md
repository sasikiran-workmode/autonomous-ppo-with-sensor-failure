# Why Agent B Loses — And How To Fix It

## The Problem: Why Is Agent B Worse At Avoiding Collisions?

Agent B was trained with 10% sensor dropout to make it "robust," but it actually has **higher** collision rates than Agent A at every failure rate. This seems backwards — if you train with missing data, shouldn't the agent handle it better?

**The answer: the current setup has 3 fundamental problems.**

### Problem 1: Dropout Makes Learning Harder, Not Smarter

The sensor dropout randomly zeros out one NPC row per step with 10% probability. But the reward function (`collision_reward = -1`) doesn't change. So Agent B is trying to learn the **same reward-maximizing behavior** as Agent A, except with **noisier input**. It's like studying for the same test while someone randomly covers parts of your textbook. You don't learn to handle missing pages — you just learn worse.

The agent never receives a **signal** that says "you successfully avoided a crash despite missing data." It only gets the same speed reward and crash penalty. So the dropout is pure handicap with no compensating training signal.

### Problem 2: Collision Penalty Is Too Weak Relative to Speed Reward

Current reward weights:
```
collision_reward:   -1     (penalty on crash, but only once per episode)
high_speed_reward:   0.4   (reward EVERY step for going fast)
normalize_reward:    True  (squishes everything to [0, 1])
```

With `normalize_reward = True`, the collision penalty gets diluted. An agent gets +0.4 every single step for driving fast (40 steps per episode = ~16 units of speed reward), but only -1 once if it crashes. The policy learns: "crashing is a minor cost compared to slowing down." This is worse for Agent B because it has noisier observations, so it can't fine-tune lane changes as precisely — it takes riskier actions and gets rewarded for the speed but occasionally crashes.

### Problem 3: Not Enough Training Steps Under Noise

200,000 timesteps is modest. Agent A converges to a reasonable policy in that time. But Agent B has to solve a harder problem (learn with noisy input), and it gets the same number of steps. It hasn't had enough samples to learn cautious behaviors that compensate for the noise.

---

## The Fix: 4 Changes to Make Agent B Genuinely Better

### Change 1: Safety-Oriented Reward Shaping for Agent B

> [!IMPORTANT]
> This is the most impactful change. Agent B needs a reward function that **explicitly rewards cautious behavior** when sensors are unreliable.

**What to change in `config.py`:**
Create a separate reward configuration for Agent B that heavily penalizes collisions and slightly reduces the speed incentive:

```python
# Existing (shared, used for Agent A)
ENV_CONFIG_A = {
    ...
    "collision_reward": -1,
    "high_speed_reward": 0.4,
    "right_lane_reward": 0.1,
    "lane_change_reward": 0,
}

# New: safety-first config for Agent B
ENV_CONFIG_B = {
    ...
    "collision_reward": -5,        # 5x stronger crash penalty
    "high_speed_reward": 0.2,      # halve the speed incentive
    "right_lane_reward": 0.1,
    "lane_change_reward": -0.05,   # small penalty for lane changes (reduces risky weaving)
}
```

**Why this works:** With a -5 collision penalty vs only +0.2 per step for speed, the agent learns: "staying alive is worth much more than going fast." Combined with sensor dropout, the agent now has an actual reason to be cautious when it can't see a neighbor — because crashing is catastrophic.

### Change 2: Curriculum Dropout (Gradually Increase Noise)

Instead of a fixed 10% dropout from step 1, use **curriculum learning**:
- Start with 0% dropout for the first 25% of training (let the agent learn basic driving)
- Increase to 5% for the next 25% 
- Increase to 10% for the next 25%
- Increase to 15% for the final 25% (slightly exceed eval conditions)

**Implementation:** Modify `SensorDropoutWrapper` to accept a schedule, or split training into 4 phases with different dropout rates, loading the previous model each time.

**Why this works:** The agent first learns what good driving looks like, then gradually adapts to noise rather than being thrown into noise from the start.

### Change 3: More Training Steps (500k instead of 200k)

Agent B needs more experience to converge under noisy observations. Increase training from 200,000 to 500,000 timesteps.

This is straightforward — the extra time lets the policy average over the noise and learn truly robust behaviors rather than just a weaker version of Agent A's policy.

### Change 4: Increase Gamma (Discount Factor) for Agent B

Current `gamma = 0.8` means the agent heavily discounts future rewards — it's very short-sighted. A collision 5 steps away is discounted to `0.8^5 = 0.33` of its value. With noisy sensors, the agent needs to plan further ahead to be cautious.

**Change for Agent B:** `gamma = 0.95`

This makes the agent care more about what happens 10-20 steps from now, which is critical when you can't always see nearby cars.

---

## Implementation Plan

### Files to Modify

#### [MODIFY] config.py
- Add `ENV_CONFIG_B` with safety-oriented rewards (`collision_reward = -5`, `high_speed_reward = 0.2`, `lane_change_reward = -0.05`)
- Add `PPO_CONFIG_B` with `gamma = 0.95`
- Add `AGENT_B_TIMESTEPS = 500_000`
- Keep `ENV_CONFIG` and `PPO_CONFIG` unchanged (used by Agent A)

#### [MODIFY] sensor_dropout.py
- Add a `set_dropout_prob()` method (already exists) and a curriculum schedule helper function `get_dropout_for_step(current_step, total_steps)` that returns the dropout rate for curriculum training

#### [MODIFY] environment.py
- Add `create_training_env_b()` factory that uses `ENV_CONFIG_B` instead of `ENV_CONFIG`
- Keep `create_training_env()` unchanged for Agent A

#### [MODIFY] train.py
- When training Agent B, use `ENV_CONFIG_B` and `PPO_CONFIG_B`
- Implement 4-phase curriculum: train for 125k steps at each dropout level (0%, 5%, 10%, 15%), saving/loading the model between phases
- Pass `--timesteps 500000` for Agent B

#### [NO CHANGE] evaluate.py, app.py, live_simulation.py
- These use the saved model files — no changes needed. The evaluation uses the shared `ENV_CONFIG` for both agents (same test conditions), which is correct.

### Training Type

**Curriculum Learning with Reward Shaping** — this is a standard technique in safe reinforcement learning where:
1. The reward function is re-shaped to prioritize safety over performance
2. The difficulty (sensor dropout) is gradually increased during training
3. The agent builds on previously learned safe behaviors rather than learning from scratch under noise

### Execution Steps

```bash
# 1. Delete old Agent B model (keep Agent A untouched)
rm models/agent_b_ppo.zip models/agent_b_meta.json

# 2. Retrain Agent B with the new config (takes ~15-25 min)
python train.py --agent B --timesteps 500000 --dropout-prob 0.15

# 3. Re-run evaluation
rm results/summary.json results/episode_metrics.csv
python evaluate.py --episodes 30 --failure-rates 0 0.05 0.10 0.20

# 4. Restart dashboard
python app.py
```

### Expected Outcomes

| Metric | Current B | Expected B (after fix) | Why |
|--------|-----------|----------------------|-----|
| Collision rate @ 0% | 23.3% | ~10-15% | Safety reward discourages risky driving |
| Collision rate @ 5% | 13.7% (adjusted) | ~8-12% | Curriculum training teaches gradual adaptation |
| Collision rate @ 10% | 20.0% | ~10-15% | Trained at this exact dropout level with safety reward |
| Collision rate @ 20% | 33.3% | ~15-20% | Over-trained at 15% dropout generalizes to 20% |
| Avg speed (all) | ~22.5 m/s | ~20-21 m/s | Slightly slower because safety-first, but still reasonable |
| Avg reward | ~27-28 | ~25-28 | Different reward function, not directly comparable |

> [!NOTE]
> Agent B's speed will drop slightly because the new reward explicitly prioritizes not crashing over driving fast. This is the correct safety-speed trade-off — a real autonomous vehicle should slow down when its sensors are unreliable.

### Verification Plan

1. **Automated:** Run evaluation with 30 episodes per condition, compare collision rates
2. **Visual:** Check the live demo — Agent B should visibly slow down and maintain lane more conservatively
3. **OpenCode:** Run the AI explanation to get a narrative interpretation of the improved results

---

## Summary

The core insight is: **dropout alone is a handicap, not a training signal**. You need to pair it with (1) a reward function that teaches the agent to value safety when observations are uncertain, (2) curriculum learning so the agent doesn't just learn a weak policy from scratch, and (3) enough training time to converge under the harder conditions.
