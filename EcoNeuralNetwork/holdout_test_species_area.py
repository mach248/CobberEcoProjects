"""
holdout_test_species_area.py
----------------------------
A leave-one-out generalization test. We remove the (22 ha, 30 species)
data point, retrain BOTH models on the remaining 14 points, then ask
each to predict richness at 22 ha -- a value it has never seen.

Comparing each prediction to the true value (30) shows which model
generalizes, rather than just memorizing its training data.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

# -- Full data -----------------------------------------------------------------

eco_data = pd.DataFrame({
    "habitat_area_ha": [0.5, 1, 2, 4, 6, 8, 12, 16, 22,
                        30, 42, 58, 75, 95, 120],
    "species_richness": [4, 6, 9, 13, 16, 19, 23, 26, 30,
                         34, 38, 41, 44, 46, 48],
})

# -- Remove the row where area == 22 and richness == 30 -----------------------

eco_train = eco_data[~((eco_data["habitat_area_ha"] == 22) &
                       (eco_data["species_richness"] == 30))].copy()
print(f"Training rows after removing (22, 30): {len(eco_train)}")

X_train = eco_train[["habitat_area_ha"]]
y_train = eco_train["species_richness"]

# The held-out point we will try to predict
X_test = pd.DataFrame({"habitat_area_ha": [22]})
true_value = 30

# -- Retrain the linear model -------------------------------------------------

lin = LinearRegression()
lin.fit(X_train, y_train)
lin_pred = lin.predict(X_test)[0]

# -- Retrain the neural network (same settings as before) ---------------------

nn = MLPRegressor(
    hidden_layer_sizes=(10, 10),
    activation="relu",
    max_iter=5000,
    early_stopping=False,
    random_state=42,
)
nn.fit(X_train, y_train)
nn_pred = nn.predict(X_test)[0]

# -- Absolute prediction errors at 22 ha --------------------------------------

lin_error = abs(lin_pred - true_value)
nn_error  = abs(nn_pred - true_value)

print(f"\nPredicting species richness at 22 ha (true value = {true_value}):")
print("-" * 58)
print(f"  Linear model    : predicted {lin_pred:6.2f}   "
      f"absolute error = {lin_error:.2f} species")
print(f"  Neural network  : predicted {nn_pred:6.2f}   "
      f"absolute error = {nn_error:.2f} species")
