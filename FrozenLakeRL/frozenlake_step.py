"""
frozenlake_step.py
------------------
Takes a single step in the environment and unpacks everything
env.step() returns. Understanding these five values is essential
before writing a Q-learning loop, because each one feeds the update.

We reset (agent at start, state 0) and take action 2 = RIGHT.
With is_slippery=False the move is deterministic.
"""

import gymnasium as gym

# -- Set up and reset ----------------------------------------------------------

env = gym.make("FrozenLake-v1", is_slippery=False)
state, info = env.reset()
print(f"Start state: {state}\n")

# -- Take ONE step: action 2 = RIGHT ------------------------------------------

action = 2   # RIGHT
new_state, reward, terminated, truncated, info = env.step(action)

print(f"Took action {action} (RIGHT) from state {state}.\n")
print("env.step() returned five values:")
print("=" * 55)
print(f"  new_state  = {new_state}")
print(f"  reward     = {reward}")
print(f"  terminated = {terminated}")
print(f"  truncated  = {truncated}")
print(f"  info       = {info}")
print("=" * 55)
