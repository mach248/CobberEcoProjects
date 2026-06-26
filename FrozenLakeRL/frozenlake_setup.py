"""
frozenlake_setup.py
-------------------
First look at a reinforcement learning environment: FrozenLake-v1.

The agent starts at the top-left of a frozen grid and must reach the
goal at the bottom-right without falling into a hole. With
is_slippery=False the ice is deterministic -- an action always moves
the agent the way you intend -- which makes the agent's behavior easy
to follow before adding randomness later.

This script just sets up the environment and prints what it looks like.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def grid_index_of(grid, target, n_cols):
    """Return the state number of the first cell matching target (e.g. 'G')."""
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == target:
                return r * n_cols + c
    return None

# -- Create the environment ----------------------------------------------------

# is_slippery=False -> deterministic moves (no sliding on the ice yet)
env = gym.make("FrozenLake-v1", is_slippery=False)

# -- Inspect the spaces --------------------------------------------------------

n_states  = env.observation_space.n   # how many grid squares the agent can be on
n_actions = env.action_space.n        # how many moves the agent can make

print(f"Number of states  (observation space): {n_states}")
print(f"Number of actions (action space)     : {n_actions}")

# What each action number means
actions = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
print("\nAction meanings:")
for number, name in actions.items():
    print(f"  {number} = {name}")

# -- Reset and see the starting state -----------------------------------------

# reset() returns a tuple: (starting_state, info_dictionary)
state, info = env.reset()
print(f"\nStarting state: {state}")
print(f"Info returned by reset(): {info}")

# -- Print the actual map and verify hole locations ---------------------------

# env.unwrapped.desc is the real grid the environment is using.
# It comes back as bytes, so we decode each cell to a normal string.
desc = env.unwrapped.desc
grid = [[cell.decode("utf-8") for cell in row] for row in desc]

# Ecological framing (CobberEcoRL): the management landscape
#   S = crew's starting patch
#   F = passable ground (safe to move through)
#   H = resource trap (looks worth treating, but drains all crew time/budget)
#   G = priority patch (the invasion that urgently needs treatment)
print("\nThe management landscape (CobberEcoRL):")
print("  S = start patch, F = passable ground,")
print("  H = resource trap, G = priority patch (goal)\n")

n_cols = len(grid[0])
traps = []
for r, row in enumerate(grid):
    row_str = "  ".join(row)
    print(f"  row {r}:  {row_str}")
    for c, cell in enumerate(row):
        state_number = r * n_cols + c          # left->right, top->bottom numbering
        if cell == "H":
            traps.append(state_number)

print(f"\nResource trap states (avoid these) : {traps}")
print(f"Priority patch (goal) state        : {grid_index_of(grid, 'G', n_cols)}")
print(f"Start patch state                  : {grid_index_of(grid, 'S', n_cols)}")
