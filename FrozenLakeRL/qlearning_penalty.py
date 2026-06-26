"""
qlearning_penalty.py
--------------------
Retrains the Q-learning agent on deterministic FrozenLake
(is_slippery=False) using a CUSTOM (shaped) reward instead of the
raw environment reward:

    +10.0  for reaching the goal
     -5.0  for falling into a hole
     -0.1  for every other step (encourages shorter paths)

Same setup otherwise: fresh zero Q-table, 2000 episodes,
alpha 0.8, gamma 0.95, epsilon 1.0 -> 0.01 decaying * 0.995.

Outputs: learning_curve_penalty.png, learned_policy_penalty.png
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# -- Reproducibility -----------------------------------------------------------

SEED = 42
np.random.seed(SEED)

# -- Environment + fresh Q-table ----------------------------------------------

env = gym.make("FrozenLake-v1", is_slippery=False)
env.action_space.seed(SEED)

q_table = np.zeros((env.observation_space.n, env.action_space.n))

# -- Hyperparameters -----------------------------------------------------------

EPISODES      = 2000
ALPHA         = 0.8
GAMMA         = 0.95
EPSILON       = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN   = 0.01

# -- Custom reward values ------------------------------------------------------

REWARD_GOAL = 10.0
REWARD_HOLE = -5.0
REWARD_STEP = -0.1

GOAL  = 15
HOLES = [5, 7, 11, 12]


def custom_reward(new_state, terminated):
    """Shape the reward based on where the agent ended up."""
    if new_state == GOAL:
        return REWARD_GOAL          # reached the goal
    elif terminated and new_state in HOLES:
        return REWARD_HOLE          # fell into a hole (episode ended, not at goal)
    else:
        return REWARD_STEP          # ordinary step


# -- Train ---------------------------------------------------------------------

episode_rewards = []   # raw goal-reached (0/1) per episode, for an honest curve

for episode in range(EPISODES):
    state, info = env.reset(seed=SEED) if episode == 0 else env.reset()
    terminated = truncated = False
    reached_goal = 0

    while not (terminated or truncated):
        if np.random.random() < EPSILON:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        new_state, raw_reward, terminated, truncated, info = env.step(action)

        # Replace the raw reward with our shaped reward
        reward = custom_reward(new_state, terminated)

        old = q_table[state, action]
        q_table[state, action] = old + ALPHA * (reward + GAMMA * np.max(q_table[new_state]) - old)
        state = new_state

    reached_goal = 1 if state == GOAL else 0
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    episode_rewards.append(reached_goal)

print("Training complete (custom reward).")

# -- Test: 100 greedy episodes -------------------------------------------------

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
print(f"Greedy test success rate (custom reward): {successes}/{N_TEST} = {success_rate:.1f}%")

# -- Learning curve (fraction reaching goal, smoothed) ------------------------

window  = 100
rolling = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
x = np.arange(window - 1, len(episode_rewards))

plt.figure(figsize=(9, 6))
plt.plot(x, rolling, color="darkorange", linewidth=2, label="100-episode rolling average")
plt.axhline(0, color="black", linewidth=1.0, linestyle="--")
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Fraction reaching the goal", fontsize=12)
plt.title("Q-Learning Progress with Custom Reward", fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig("learning_curve_penalty.png", dpi=150)
print("Saved: learning_curve_penalty.png")

# -- Policy map ----------------------------------------------------------------

START = 0
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
ax.set_title("Learned Policy with Custom Reward", fontsize=13)
plt.tight_layout()
plt.savefig("learned_policy_penalty.png", dpi=150)
print("Saved: learned_policy_penalty.png")
plt.show()
