"""
linear_residuals_species_area.py
--------------------------------
Computes and plots the residuals of the linear regression baseline
on the original 15 species-area data points.

Residuals are defined here as:  predicted - actual
  > 0  : model OVER-predicted (guessed too many species)
  < 0  : model UNDER-predicted (guessed too few species)

A linear model forced onto a curved relationship leaves a systematic
(patterned) residual shape rather than random scatter around zero.

Output: linear_residuals.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -- Data ----------------------------------------------------------------------

eco_data = pd.DataFrame({
    "habitat_area_ha": [0.5, 1, 2, 4, 6, 8, 12, 16, 22,
                        30, 42, 58, 75, 95, 120],
    "species_richness": [4, 6, 9, 13, 16, 19, 23, 26, 30,
                         34, 38, 41, 44, 46, 48],
})

X = eco_data[["habitat_area_ha"]]
y = eco_data["species_richness"]

# -- Fit and predict -----------------------------------------------------------

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# -- Residuals: predicted - actual --------------------------------------------

residuals = y_pred - y.values

# -- Residual plot -------------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.scatter(eco_data["habitat_area_ha"], residuals, color="steelblue",
            s=60, alpha=0.8, edgecolors="black", linewidths=0.5)
plt.axhline(0, color="black", linewidth=1.2, linestyle="--",
            label="Zero (perfect prediction)")
plt.xlabel("Habitat Area (ha)", fontsize=12)
plt.ylabel("Residual (Predicted - Actual) (species)", fontsize=12)
plt.title("Linear Model Residuals", fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig("linear_residuals.png", dpi=150)
print("Saved: linear_residuals.png")
plt.show()
