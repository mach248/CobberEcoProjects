"""
qlearning_train.py
------------------
Trains a Q-learning agent on FrozenLake-v1 (is_slippery=False),
the CobberEcoRL management landscape.

The agent learns by trial and error: it explores the grid, occasionally
reaches the priority patch (reward 1), and uses the Bellman equation to
push that value backward into the Q-table so earlier states learn which
actions eventually lead to the goal.

Settings:
  episodes      = 2000
  alpha (LR)    = 0.8     how strongly each new experience updates the table
  gamma         = 0.95    how much future reward matters vs. immediate
  epsilon       = 1.0 -> 0.01   exploration rate, decays *0.995 per episode
"""

import numpy as np
import gymnasium as gym

# -- Environment and Q-table ---------------------------------------------------

env = gym.make("FrozenLake-v1", is_slippery=False)

n_states  = env.observation_space.n      # 16
n_actions = env.action_space.n           # 4
q_table   = np.zeros((n_states, n_actions))

# -- Hyperparameters -----------------------------------------------------------

EPISODES      = 2000
ALPHA         = 0.8      # learning rate
GAMMA         = 0.95     # discount factor
EPSILON       = 1.0      # starting exploration rate
EPSILON_DECAY = 0.995
EPSILON_MIN   = 0.01

# -- Training loop -------------------------------------------------------------

episode_rewards = []

for episode in range(EPISODES):
    state, info = env.reset()
    terminated = False
    truncated  = False
    total_reward = 0

    while not (terminated or truncated):
        # -- Choose an action: explore (random) or exploit (best known) -------
        if np.random.random() < EPSILON:
            action = env.action_space.sample()          # explore
        else:
            action = np.argmax(q_table[state])          # exploit best known

        # -- Take the action --------------------------------------------------
        new_state, reward, terminated, truncated, info = env.step(action)

        # -- Bellman update ---------------------------------------------------
        # Q(s,a) <- Q(s,a) + alpha * [ reward + gamma * max_a' Q(s',a') - Q(s,a) ]
        old_value  = q_table[state, action]
        next_max   = np.max(q_table[new_state])
        q_table[state, action] = old_value + ALPHA * (reward + GAMMA * next_max - old_value)

        # -- Move to the next state ------------------------------------------
        state = new_state
        total_reward += reward

    # -- Decay exploration after each episode ---------------------------------
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

    episode_rewards.append(total_reward)

print(f"Training complete: {EPISODES} episodes.")
print(f"Final exploration rate (epsilon): {EPSILON:.3f}")
print(f"Rewards in last 100 episodes: {int(sum(episode_rewards[-100:]))}/100 reached the goal")
