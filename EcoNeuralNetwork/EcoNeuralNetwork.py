"""
ann_species_area.py
-------------------
Sets up a species-area dataset for an artificial neural network (ANN)
experiment, and plots it.

The species-area relationship is non-linear: species richness rises
steeply when habitat area is small, then levels off as area grows.
That decelerating curve is a good test case for a neural network,
which can fit non-linear shapes a straight line cannot.

Output: ecological_data.png
"""

import pandas as pd
import matplotlib.pyplot as plt

# -- Build the dataset ---------------------------------------------------------

eco_data = pd.DataFrame({
    "habitat_area_ha": [0.5, 1, 2, 4, 6, 8, 12, 16, 22,
                        30, 42, 58, 75, 95, 120],
    "species_richness": [4, 6, 9, 13, 16, 19, 23, 26, 30,
                         34, 38, 41, 44, 46, 48],
})

print("eco_data:")
print(eco_data.to_string(index=False))

# -- Features and target -------------------------------------------------------

X = eco_data["habitat_area_ha"]
y = eco_data["species_richness"]

# -- Scatter plot --------------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.scatter(X, y, color="seagreen", s=60, alpha=0.8, edgecolors="black", linewidths=0.5)
plt.xlabel("Habitat Area (ha)", fontsize=12)
plt.ylabel("Species Richness", fontsize=12)
plt.title("Species-Area Relationship", fontsize=13)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig("ecological_data.png", dpi=150)
print("\nSaved: ecological_data.png")
plt.show()