"""
learned_policy.py
-----------------
Trains the Q-learning agent, then visualizes its learned POLICY:
for each state, the action the agent would choose (the one with the
highest Q-value), drawn as a directional arrow on the 4x4 grid.

  Action -> arrow:  0 = left, 1 = down, 2 = right, 3 = up

Special cells are color-coded:
  START (state 0), GOAL (state 15), and HOLES (states 5, 7, 11, 12).

Output: learned_policy.png
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# -- Reproducibility -----------------------------------------------------------

SEED = 42
np.random.seed(SEED)

# -- Train (reliable settings: slower epsilon decay so it learns) -------------

env = gym.make("FrozenLake-v1", is_slippery=False)
env.action_space.seed(SEED)

n_states  = env.observation_space.n
n_actions = env.action_space.n
q_table   = np.zeros((n_states, n_actions))

EPISODES, ALPHA, GAMMA = 2000, 0.8, 0.95
EPSILON, EPSILON_DECAY, EPSILON_MIN = 1.0, 0.999, 0.01

for episode in range(EPISODES):
    state, info = env.reset(seed=SEED) if episode == 0 else env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        if np.random.random() < EPSILON:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        new_state, reward, terminated, truncated, info = env.step(action)
        old = q_table[state, action]
        q_table[state, action] = old + ALPHA * (reward + GAMMA * np.max(q_table[new_state]) - old)
        state = new_state
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

print("Training complete. Building policy grid...")

# -- Identify special states ---------------------------------------------------

START = 0
GOAL  = 15
HOLES = [5, 7, 11, 12]

# Arrow symbol for each action
ARROWS = {0: "\u2190", 1: "\u2193", 2: "\u2192", 3: "\u2191"}  # left down right up

# -- Build the colored background grid -----------------------------------------

# Colors (RGB 0-1) for each cell type
COLOR_NORMAL = (0.93, 0.95, 0.97)   # light grey-blue
COLOR_START  = (0.62, 0.79, 0.88)   # blue
COLOR_GOAL   = (0.70, 0.87, 0.66)   # green
COLOR_HOLE   = (0.92, 0.60, 0.55)   # red

grid_colors = np.zeros((4, 4, 3))
for s in range(16):
    r, c = s // 4, s % 4
    if s == START:
        grid_colors[r, c] = COLOR_START
    elif s == GOAL:
        grid_colors[r, c] = COLOR_GOAL
    elif s in HOLES:
        grid_colors[r, c] = COLOR_HOLE
    else:
        grid_colors[r, c] = COLOR_NORMAL

# -- Plot ----------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(grid_colors)

for s in range(16):
    r, c = s // 4, s % 4
    if s == GOAL:
        ax.text(c, r, "GOAL", ha="center", va="center", fontsize=13, fontweight="bold")
    elif s in HOLES:
        ax.text(c, r, "HOLE", ha="center", va="center", fontsize=12, fontweight="bold")
    elif s == START:
        # START label plus the learned arrow
        best = int(np.argmax(q_table[s]))
        ax.text(c, r - 0.18, "START", ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(c, r + 0.20, ARROWS[best], ha="center", va="center", fontsize=26)
    else:
        best = int(np.argmax(q_table[s]))
        ax.text(c, r, ARROWS[best], ha="center", va="center", fontsize=30)

# Grid lines and labels
ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xticklabels(range(4)); ax.set_yticklabels(range(4))
ax.set_title("Learned Policy on FrozenLake\n(arrow = best action in each state)", fontsize=13)

plt.tight_layout()
plt.savefig("learned_policy.png", dpi=150)
print("Saved: learned_policy.png")
plt.show()
