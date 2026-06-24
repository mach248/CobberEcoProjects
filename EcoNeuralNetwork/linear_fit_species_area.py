"""
linear_fit_species_area.py
--------------------------
Fits a straight-line (linear regression) model to the species-area
data as a baseline before trying a neural network.

A linear model can only draw a straight line, so it cannot follow the
decelerating species-area curve. Seeing where it misses (overshooting
at the extremes, undershooting in the middle) motivates a non-linear
model.

Output: linear_fit.png
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

# X must be 2D for sklearn: shape (n_samples, 1)
X = eco_data[["habitat_area_ha"]]      # double brackets -> 2D DataFrame
y = eco_data["species_richness"]

# -- Fit the linear model ------------------------------------------------------

model = LinearRegression()
model.fit(X, y)

print(f"Fitted line: richness = {model.coef_[0]:.3f} * area + {model.intercept_:.3f}")
print(f"R-squared on training data: {model.score(X, y):.3f}")

# -- Predict across a smooth range so the line looks continuous ---------------

# Use a DataFrame with the same column name as X so sklearn is happy
area_smooth   = np.linspace(0, 130, 200)
area_smooth_df = pd.DataFrame({"habitat_area_ha": area_smooth})
richness_pred = model.predict(area_smooth_df)

# -- Plot ----------------------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.scatter(eco_data["habitat_area_ha"], y, color="seagreen", s=60,
            alpha=0.8, edgecolors="black", linewidths=0.5, label="Data")
plt.plot(area_smooth, richness_pred, color="tomato", linewidth=2,
         label="Linear fit")
plt.xlabel("Habitat Area (ha)", fontsize=12)
plt.ylabel("Species Richness", fontsize=12)
plt.title("Species-Area Relationship: Linear Fit", fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig("linear_fit.png", dpi=150)
print("\nSaved: linear_fit.png")
plt.show()
