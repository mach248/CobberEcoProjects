"""
diagnose_qlearning.py
---------------------
Quick diagnostic to find why the Q-learning agent isn't reaching the goal.
Run this and share the output.
"""

import numpy as np
import gymnasium as gym

print("gymnasium version:", gym.__version__)

env = gym.make("FrozenLake-v1", is_slippery=False)
print("is_slippery should be False -> deterministic\n")

# 1. Can a KNOWN correct path reach the goal and earn reward 1?
#    On the standard 4x4 map, a path of RIGHT/DOWN moves reaches state 15.
#    Path: DOWN, DOWN, RIGHT, RIGHT, DOWN, RIGHT  (one valid route)
known_path = [1, 1, 2, 2, 1, 2]   # 1=DOWN, 2=RIGHT
state, info = env.reset()
print(f"Reset start state: {state}")
total = 0
for a in known_path:
    new_state, reward, terminated, truncated, info = env.step(a)
    print(f"  action {a} -> state {new_state}, reward {reward}, "
          f"terminated {terminated}, truncated {truncated}")
    total += reward
    if terminated or truncated:
        break
print(f"\nTotal reward on this hand-coded path: {total}")
print("(If this reaches state 15 with reward 1.0, the environment is fine")
print(" and the problem is in the learning settings. If reward stays 0 all")
print(" the way to state 15, the reward signal itself is the issue.)")
