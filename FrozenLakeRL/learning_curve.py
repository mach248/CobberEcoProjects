"""
learning_curve.py
-----------------
Trains the Q-learning agent and plots its learning curve.

Individual episode rewards are very noisy (each is just 0 or 1), so a
raw plot is unreadable. We smooth with a 100-episode rolling average,
which reveals the trend: the agent's success rate climbing from near
zero toward 1.0 as it learns.

A random seed is set so the run is reproducible and reliably shows
learning. Note: the exploration decay is set to 0.999 (slower than
0.995) so the agent explores long enough to reliably discover the
goal -- with faster decay, some random seeds stop exploring before
ever finding the reward, producing a flat curve.

Output: learning_curve.png
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# -- Reproducibility -----------------------------------------------------------

SEED = 42
np.random.seed(SEED)

# -- Environment and Q-table ---------------------------------------------------

env = gym.make("FrozenLake-v1", is_slippery=False)
env.action_space.seed(SEED)        # so env.action_space.sample() is reproducible

n_states  = env.observation_space.n
n_actions = env.action_space.n
q_table   = np.zeros((n_states, n_actions))

# -- Hyperparameters -----------------------------------------------------------

EPISODES      = 2000
ALPHA         = 0.8
GAMMA         = 0.95
EPSILON       = 1.0
EPSILON_DECAY = 0.999
EPSILON_MIN   = 0.01

# -- Training loop -------------------------------------------------------------

episode_rewards = []

for episode in range(EPISODES):
    if episode == 0:
        state, info = env.reset(seed=SEED)
    else:
        state, info = env.reset()
    terminated = truncated = False
    total_reward = 0

    while not (terminated or truncated):
        if np.random.random() < EPSILON:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        new_state, reward, terminated, truncated, info = env.step(action)

        old_value = q_table[state, action]
        next_max  = np.max(q_table[new_state])
        q_table[state, action] = old_value + ALPHA * (reward + GAMMA * next_max - old_value)

        state = new_state
        total_reward += reward

    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    episode_rewards.append(total_reward)

print(f"Training complete. Last 100 episodes: "
      f"{int(sum(episode_rewards[-100:]))}/100 reached the goal.")

# -- Smooth with a 100-episode rolling average --------------------------------

window  = 100
# np.convolve with a uniform window computes the moving average
rolling = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
# x positions: each smoothed point sits at the END of its 100-episode window
x = np.arange(window - 1, len(episode_rewards))

# -- Plot ----------------------------------------------------------------------

plt.figure(figsize=(9, 6))
plt.plot(x, rolling, color="seagreen", linewidth=2,
         label="100-episode rolling average")
plt.axhline(0, color="black", linewidth=1.0, linestyle="--")
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Average reward (success rate)", fontsize=12)
plt.title("Q-Learning Progress on FrozenLake", fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=150)
print("Saved: learning_curve.png")
plt.show()
