"""
neural_network_metrics_species_area.py
--------------------------------------
Computes MAE, MSE, and R-squared for the neural network on the
original 15 species-area data points, so they can be compared
directly against the linear model baseline.

  Linear baseline:  MAE 5.728,  MSE 42.336,  R^2 0.799

  MAE : mean absolute error (sklearn)
  MSE : mean squared error  (numpy)
  R^2 : coefficient of determination (sklearn)
"""

import numpy as np
import pandas as pd
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

# -- Train the same network (random_state=42 for reproducibility) -------------

model = MLPRegressor(
    hidden_layer_sizes=(10, 10),
    activation="relu",
    max_iter=5000,
    early_stopping=False,
    random_state=42,
)
model.fit(X, y)
y_pred = model.predict(X)

# -- Metrics on the original 15 points ----------------------------------------

mae = mean_absolute_error(y, y_pred)       # sklearn
mse = np.mean((y - y_pred) ** 2)           # numpy
r2  = r2_score(y, y_pred)                   # sklearn

print("Neural network performance (original 15 data points):")
print("-" * 55)
print(f"  Mean Absolute Error (MAE) : {mae:.3f} species")
print(f"  Mean Squared Error  (MSE) : {mse:.3f}")
print(f"  R-squared           (R^2) : {r2:.3f}")

print("\nFor comparison - linear baseline:")
print(f"  MAE 5.728,  MSE 42.336,  R^2 0.799")
