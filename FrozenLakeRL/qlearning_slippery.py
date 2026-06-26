"""
qlearning_slippery.py
---------------------
Retrains the Q-learning agent from scratch on the STOCHASTIC version
of FrozenLake (is_slippery=True), where the ice is unpredictable: the
chosen action only sometimes moves the agent the intended way, and
often slides it sideways.

Same hyperparameters as the deterministic run:
  2000 episodes, alpha 0.8, gamma 0.95,
  epsilon 1.0 -> 0.01 decaying * 0.995 per episode.

After training, runs 100 greedy test episodes and reports the success
rate. Saves a learning curve and a policy map.

Outputs: learning_curve_slippery.png, learned_policy_slippery.png
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# -- Reproducibility -----------------------------------------------------------

SEED = 42
np.random.seed(SEED)

# -- Environment + fresh all-zero Q-table -------------------------------------

env = gym.make("FrozenLake-v1", is_slippery=True)   # <-- slippery now ON
env.action_space.seed(SEED)

q_table = np.zeros((env.observation_space.n, env.action_space.n))

# -- Hyperparameters (same as before) -----------------------------------------

EPISODES      = 2000
ALPHA         = 0.8
GAMMA         = 0.95
EPSILON       = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN   = 0.01

# -- Train ---------------------------------------------------------------------

episode_rewards = []

for episode in range(EPISODES):
    state, info = env.reset(seed=SEED) if episode == 0 else env.reset()
    terminated = truncated = False
    total_reward = 0
    while not (terminated or truncated):
        if np.random.random() < EPSILON:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        new_state, reward, terminated, truncated, info = env.step(action)
        old = q_table[state, action]
        q_table[state, action] = old + ALPHA * (reward + GAMMA * np.max(q_table[new_state]) - old)
        state = new_state
        total_reward += reward
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    episode_rewards.append(total_reward)

print("Training complete (slippery ice).")

# -- Test: 100 greedy episodes -------------------------------------------------

GOAL = 15
N_TEST = 100
successes = 0
for _ in range(N_TEST):
    state, info = env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        action = int(np.argmax(q_table[state]))
        state, reward, terminated, truncated, info = env.step(action)
    if state == GOAL:
        successes += 1

success_rate = successes / N_TEST * 100
print(f"Greedy test success rate (slippery): {successes}/{N_TEST} = {success_rate:.1f}%")

# -- Learning curve ------------------------------------------------------------

window  = 100
rolling = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
x = np.arange(window - 1, len(episode_rewards))

plt.figure(figsize=(9, 6))
plt.plot(x, rolling, color="steelblue", linewidth=2, label="100-episode rolling average")
plt.axhline(0, color="black", linewidth=1.0, linestyle="--")
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Average reward (success rate)", fontsize=12)
plt.title("Q-Learning Progress on FrozenLake (Slippery Ice)", fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig("learning_curve_slippery.png", dpi=150)
print("Saved: learning_curve_slippery.png")

# -- Policy map ----------------------------------------------------------------

START, HOLES = 0, [5, 7, 11, 12]
ARROWS = {0: "\u2190", 1: "\u2193", 2: "\u2192", 3: "\u2191"}
COLOR_NORMAL = (0.93, 0.95, 0.97); COLOR_START = (0.62, 0.79, 0.88)
COLOR_GOAL   = (0.70, 0.87, 0.66); COLOR_HOLE  = (0.92, 0.60, 0.55)

grid_colors = np.zeros((4, 4, 3))
for s in range(16):
    r, c = s // 4, s % 4
    grid_colors[r, c] = (COLOR_START if s == START else COLOR_GOAL if s == GOAL
                         else COLOR_HOLE if s in HOLES else COLOR_NORMAL)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(grid_colors)
for s in range(16):
    r, c = s // 4, s % 4
    if s == GOAL:
        ax.text(c, r, "GOAL", ha="center", va="center", fontsize=13, fontweight="bold")
    elif s in HOLES:
        ax.text(c, r, "HOLE", ha="center", va="center", fontsize=12, fontweight="bold")
    elif s == START:
        best = int(np.argmax(q_table[s]))
        ax.text(c, r - 0.18, "START", ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(c, r + 0.20, ARROWS[best], ha="center", va="center", fontsize=26)
    else:
        best = int(np.argmax(q_table[s]))
        ax.text(c, r, ARROWS[best], ha="center", va="center", fontsize=30)

ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_title("Learned Policy on FrozenLake (Slippery Ice)", fontsize=13)
plt.tight_layout()
plt.savefig("learned_policy_slippery.png", dpi=150)
print("Saved: learned_policy_slippery.png")
plt.show()
