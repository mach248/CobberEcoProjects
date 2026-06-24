"""
width_exploration_species_area.py
---------------------------------
Explores how the WIDTH of a neural network (neurons in a single hidden
layer) affects the fit to the species-area data. Trains four networks
with 2, 5, 10, and 50 neurons and plots each in its own panel.

Lesson: too few neurons cannot bend enough to fit the curve (underfit);
more neurons add capacity. But the fit you get also depends on training
actually converging within the iteration budget -- so the reported R^2
reflects both the architecture AND the training run, not capacity alone.

Output: width_exploration.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

# -- Data ----------------------------------------------------------------------

eco_data = pd.DataFrame({
    "habitat_area_ha": [0.5, 1, 2, 4, 6, 8, 12, 16, 22,
                        30, 42, 58, 75, 95, 120],
    "species_richness": [4, 6, 9, 13, 16, 19, 23, 26, 30,
                         34, 38, 41, 44, 46, 48],
})

X = eco_data[["habitat_area_ha"]]
y = eco_data["species_richness"]

area_smooth    = np.linspace(0, 130, 200)
area_smooth_df = pd.DataFrame({"habitat_area_ha": area_smooth})

# -- Four widths to compare ----------------------------------------------------

widths = [2, 5, 10, 50]

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

for ax, n in zip(axes.ravel(), widths):
    model = MLPRegressor(
        hidden_layer_sizes=(n,),       # single hidden layer of n neurons
        activation="relu",
        max_iter=5000,
        early_stopping=False,
        random_state=42,
    )
    model.fit(X, y)

    r2       = r2_score(y, model.predict(X))
    curve    = model.predict(area_smooth_df)
    maxed    = model.n_iter_ >= 5000   # did it hit the iteration cap?

    ax.scatter(eco_data["habitat_area_ha"], y, color="seagreen", s=45,
               alpha=0.8, edgecolors="black", linewidths=0.5, label="Data")
    ax.plot(area_smooth, curve, color="purple", linewidth=2, label="NN fit")

    note = "  (hit iter cap)" if maxed else f"  (converged @ {model.n_iter_})"
    ax.set_title(f"{n} neurons   R\u00b2 = {r2:.3f}{note}", fontsize=12)
    ax.set_xlabel("Habitat Area (ha)", fontsize=10)
    ax.set_ylabel("Species Richness", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

fig.suptitle("Effect of Network Width on Fit (single hidden layer)", fontsize=14)
plt.tight_layout()
plt.savefig("width_exploration.png", dpi=150)
print("Saved: width_exploration.png")
plt.show()
