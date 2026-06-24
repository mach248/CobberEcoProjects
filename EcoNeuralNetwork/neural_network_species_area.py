"""
neural_network_species_area.py
------------------------------
Trains a small neural network (MLPRegressor) on the species-area data
and plots its predictions as a smooth curve.

Unlike the straight line from the linear model, the network can bend to
follow the decelerating species-area relationship.

Network: 2 hidden layers, 10 neurons each, ReLU activation, 5000 iters.

Output: neural_network_fit.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# -- Data ----------------------------------------------------------------------

eco_data = pd.DataFrame({
    "habitat_area_ha": [0.5, 1, 2, 4, 6, 8, 12, 16, 22,
                        30, 42, 58, 75, 95, 120],
    "species_richness": [4, 6, 9, 13, 16, 19, 23, 26, 30,
                         34, 38, 41, 44, 46, 48],
})

X = eco_data[["habitat_area_ha"]]      # 2D, shape (15, 1)
y = eco_data["species_richness"]

# -- Build and train the neural network ---------------------------------------

# random_state fixes the random starting weights so results are reproducible
model = MLPRegressor(
    hidden_layer_sizes=(10, 10),   # two hidden layers, 10 neurons each
    activation="relu",
    max_iter=5000,
    early_stopping=False,           # run the full 5000 iterations
    random_state=42,
)
model.fit(X, y)

print(f"Training finished after {model.n_iter_} iterations.")
print(f"Final training loss: {model.loss_:.4f}")

# -- Predict across a smooth range (same as the linear model) -----------------

area_smooth    = np.linspace(0, 130, 200)
area_smooth_df = pd.DataFrame({"habitat_area_ha": area_smooth})
richness_pred  = model.predict(area_smooth_df)

# -- Plot ----------------------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.scatter(eco_data["habitat_area_ha"], y, color="seagreen", s=60,
            alpha=0.8, edgecolors="black", linewidths=0.5, label="Data")
plt.plot(area_smooth, richness_pred, color="purple", linewidth=2,
         label="Neural network fit")
plt.xlabel("Habitat Area (ha)", fontsize=12)
plt.ylabel("Species Richness", fontsize=12)
plt.title("Species-Area Relationship: Neural Network Fit", fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig("neural_network_fit.png", dpi=150)
print("\nSaved: neural_network_fit.png")
plt.show()
