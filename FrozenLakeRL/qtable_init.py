"""
qtable_init.py
--------------
Creates the Q-table for the FrozenLake / CobberEcoRL agent.

A Q-table stores the agent's learned estimate of how good each action
is from each state. It has one ROW per state (16) and one COLUMN per
action (4: LEFT, DOWN, RIGHT, UP). Every entry Q[state, action] is the
expected long-run value of taking that action in that state.

We start every value at zero -- the agent knows nothing yet. Learning
will gradually fill these in.
"""

import numpy as np

# -- Dimensions ----------------------------------------------------------------

n_states  = 16   # one row per grid square (state)
n_actions = 4    # one column per action: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP

# -- Create the Q-table, all zeros --------------------------------------------

q_table = np.zeros((n_states, n_actions))

# -- Confirm shape and show starting values -----------------------------------

print(f"Q-table shape: {q_table.shape}   (rows = states, columns = actions)\n")

# Header so the columns are easy to read
print(f"{'state':>6}   {'LEFT':>6} {'DOWN':>6} {'RIGHT':>6} {'UP':>6}")
print("-" * 40)
for s in range(n_states):
    left, down, right, up = q_table[s]
    print(f"{s:>6}   {left:>6.1f} {down:>6.1f} {right:>6.1f} {up:>6.1f}")
