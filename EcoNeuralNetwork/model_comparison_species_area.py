"""
model_comparison_species_area.py
--------------------------------
Trains both models on the full 15-point dataset, plots them together,
and prints a side-by-side metrics table.

Output: model_comparison.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -- Data ----------------------------------------------------------------------

eco_data = pd.DataFrame({
    "habitat_area_ha": [0.5, 1, 2, 4, 6, 8, 12, 16, 22,
                        30, 42, 58, 75, 95, 120],
    "species_richness": [4, 6, 9, 13, 16, 19, 23, 26, 30,
                         34, 38, 41, 44, 46, 48],
})

X = eco_data[["habitat_area_ha"]]
y = eco_data["species_richness"]

# -- Train both models on all 15 points ---------------------------------------

lin = LinearRegression().fit(X, y)
nn  = MLPRegressor(hidden_layer_sizes=(10, 10), activation="relu",
                   max_iter=5000, early_stopping=False, random_state=42).fit(X, y)

# -- Smooth prediction range --------------------------------------------------

area_smooth    = np.linspace(0, 130, 200)
area_smooth_df = pd.DataFrame({"habitat_area_ha": area_smooth})
lin_curve = lin.predict(area_smooth_df)
nn_curve  = nn.predict(area_smooth_df)

# -- Plot ----------------------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.scatter(eco_data["habitat_area_ha"], y, color="seagreen", s=60,
            alpha=0.8, edgecolors="black", linewidths=0.5, label="Data")
plt.plot(area_smooth, lin_curve, color="tomato", linewidth=2, label="Linear regression")
plt.plot(area_smooth, nn_curve, color="purple", linewidth=2, label="Neural network")
plt.xlabel("Habitat Area (ha)", fontsize=12)
plt.ylabel("Species Richness", fontsize=12)
plt.title("Model Comparison: Linear Regression vs Neural Network", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
print("Saved: model_comparison.png\n")

# -- Metrics table (in-sample, all 15 points) ---------------------------------

def metrics(model):
    p = model.predict(X)
    return mean_absolute_error(y, p), np.mean((y - p) ** 2), r2_score(y, p)

lin_mae, lin_mse, lin_r2 = metrics(lin)
nn_mae,  nn_mse,  nn_r2  = metrics(nn)

print("Model performance summary (all 15 data points):")
print("=" * 56)
print(f"  {'Metric':<22}{'Linear':>14}{'Neural Net':>16}")
print("  " + "-" * 52)
print(f"  {'MAE (species)':<22}{lin_mae:>14.3f}{nn_mae:>16.3f}")
print(f"  {'MSE':<22}{lin_mse:>14.3f}{nn_mse:>16.3f}")
print(f"  {'R-squared':<22}{lin_r2:>14.3f}{nn_r2:>16.3f}")
print("=" * 56)
