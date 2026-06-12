"""
plot_residuals.py
-----------------
Loads the regression results CSV and produces a residual plot:
  - X-axis : actual mean annual temperature
  - Y-axis : residuals (actual - predicted)
  - Points above zero (model under-predicted) in tomato red
  - Points below zero (model over-predicted) in steelblue
  - Horizontal line at zero marks perfect prediction

A good residual plot should show points scattered randomly around
zero with no clear pattern. Systematic curves or fans indicate
the model assumptions may not be fully met.

Input  : regression_results_lat_temp.csv
Output : plot_residuals.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# -- Configuration -------------------------------------------------------------

INPUT_CSV  = "regression_results_lat_temp.csv"
OUTPUT_PNG = "plot_residuals.png"

TITLE   = "Residual Plot: Latitude → Mean Annual Temperature\n(Vulpes vulpes, GBIF 2000–2025)"
X_LABEL = "Actual Mean Annual Temperature (°C)"
Y_LABEL = "Residual (Actual − Predicted) (°C)"

# -- Load ----------------------------------------------------------------------

df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df):,} test set records.")

# -- Split into above/below zero -----------------------------------------------

above = df[df["residual"] >= 0]   # model under-predicted
below = df[df["residual"] <  0]   # model over-predicted

print(f"  Points above zero (under-predicted) : {len(above):,}")
print(f"  Points below zero (over-predicted)  : {len(below):,}")

# -- Plot ----------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    above["actual_temp_c"],
    above["residual"],
    alpha=0.5,
    s=20,
    color="tomato",
    edgecolors="none",
    label="Above zero (under-predicted)",
)

ax.scatter(
    below["actual_temp_c"],
    below["residual"],
    alpha=0.5,
    s=20,
    color="steelblue",
    edgecolors="none",
    label="Below zero (over-predicted)",
)

# Zero reference line
ax.axhline(0, color="black", linewidth=1.2, linestyle="--", label="Zero (perfect prediction)")

# Labels, title, legend
ax.set_xlabel(X_LABEL, fontsize=12)
ax.set_ylabel(Y_LABEL, fontsize=12)
ax.set_title(TITLE, fontsize=13)
ax.legend(fontsize=9)

# Grid
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
ax.grid(True, which="minor", linestyle=":",  linewidth=0.3, alpha=0.4)

# Record count annotation
ax.annotate(
    f"n = {len(df):,}",
    xy=(0.97, 0.97),
    xycoords="axes fraction",
    ha="right", va="top",
    fontsize=10,
    color="gray",
)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150)
print(f"Plot saved: {OUTPUT_PNG}")
plt.show()
