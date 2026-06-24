"""
generalization_test_plot.py
---------------------------
Visualizes the leave-one-out generalization test. Both models are
trained on 14 points (the (22, 30) point removed), and the held-out
point is marked so you can see how close each model's curve comes to
predicting it.

Output: generalization_test.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

# -- Full data, then split off the held-out point -----------------------------

eco_data = pd.DataFrame({
    "habitat_area_ha": [0.5, 1, 2, 4, 6, 8, 12, 16, 22,
                        30, 42, 58, 75, 95, 120],
    "species_richness": [4, 6, 9, 13, 16, 19, 23, 26, 30,
                         34, 38, 41, 44, 46, 48],
})

held_out = (eco_data["habitat_area_ha"] == 22) & (eco_data["species_richness"] == 30)
eco_train = eco_data[~held_out].copy()

X_train = eco_train[["habitat_area_ha"]]
y_train = eco_train["species_richness"]

# -- Retrain both models on the 14 points -------------------------------------

lin = LinearRegression().fit(X_train, y_train)

nn = MLPRegressor(hidden_layer_sizes=(10, 10), activation="relu",
                  max_iter=5000, early_stopping=False, random_state=42).fit(X_train, y_train)

# -- Smooth prediction range --------------------------------------------------

area_smooth    = np.linspace(0, 130, 200)
area_smooth_df = pd.DataFrame({"habitat_area_ha": area_smooth})
lin_curve = lin.predict(area_smooth_df)
nn_curve  = nn.predict(area_smooth_df)

# -- Plot ----------------------------------------------------------------------

plt.figure(figsize=(8, 6))

# 14 training points
plt.scatter(X_train["habitat_area_ha"], y_train, color="seagreen", s=60,
            alpha=0.8, edgecolors="black", linewidths=0.5, label="Training data (14 points)")

# Linear and neural network curves
plt.plot(area_smooth, lin_curve, color="tomato", linewidth=2, label="Linear regression")
plt.plot(area_smooth, nn_curve, color="purple", linewidth=2, label="Neural network")

# The held-out point, marked clearly
plt.scatter([22], [30], color="gold", s=260, marker="*",
            edgecolors="black", linewidths=1.0, zorder=5,
            label="Held-out point (22 ha, 30 species)")

plt.xlabel("Habitat Area (ha)", fontsize=12)
plt.ylabel("Species Richness", fontsize=12)
plt.title("Generalization Test", fontsize=13)
plt.legend(fontsize=9)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig("generalization_test.png", dpi=150)
print("Saved: generalization_test.png")
plt.show()
