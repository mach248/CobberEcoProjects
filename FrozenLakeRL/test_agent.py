"""
test_agent.py
-------------
Trains the Q-learning agent, then evaluates it two ways:

  1. ROLLOUT: walk one greedy episode step by step, printing the path
     so you can watch the agent travel from start to goal.
  2. EVALUATION: run 100 greedy test episodes (no exploration) and
     report the success rate -- how often it reaches the goal.

"Greedy" = always pick the highest-Q action, no random exploration,
which is how you deploy a trained agent.
"""

import numpy as np
import gymnasium as gym

# -- Reproducibility -----------------------------------------------------------

SEED = 42
np.random.seed(SEED)

# -- Train (reliable settings) -------------------------------------------------

env = gym.make("FrozenLake-v1", is_slippery=False)
env.action_space.seed(SEED)

q_table = np.zeros((env.observation_space.n, env.action_space.n))
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

print("Training complete.\n")

# -- 1. Rollout: watch one greedy episode -------------------------------------

ACTION_NAMES = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
GOAL = 15

print("=" * 50)
print("ROLLOUT: watching one greedy episode")
print("=" * 50)

state, info = env.reset(seed=SEED)
terminated = truncated = False
step_num = 0
print(f"  Start: state {state}")
while not (terminated or truncated):
    action = int(np.argmax(q_table[state]))      # greedy: best known action
    new_state, reward, terminated, truncated, info = env.step(action)
    step_num += 1
    print(f"  Step {step_num}: {ACTION_NAMES[action]:<5} -> state {new_state}")
    state = new_state

if state == GOAL:
    print(f"  Reached the GOAL in {step_num} steps!\n")
else:
    print(f"  Episode ended at state {state} (did not reach goal).\n")

# -- 2. Evaluation: 100 greedy test episodes ----------------------------------

print("=" * 50)
print("EVALUATION: 100 greedy test episodes")
print("=" * 50)

N_TEST = 100
successes = 0

for _ in range(N_TEST):
    state, info = env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        action = int(np.argmax(q_table[state]))   # always greedy, no exploration
        state, reward, terminated, truncated, info = env.step(action)
    if state == GOAL:
        successes += 1

success_rate = successes / N_TEST * 100
print(f"  Episodes reaching the goal: {successes}/{N_TEST}")
print(f"  Success rate: {success_rate:.1f}%")
