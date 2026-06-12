"""
plot_predicted_vs_actual.py
----------------------------
Loads the regression results CSV and produces a predicted vs. actual
plot for the linear regression model (latitude -> mean annual temperature).

A perfect model would have all points on the 1:1 diagonal line.
Scatter around that line shows where the model over- or under-predicts.

Input  : regression_results_lat_temp.csv
Output : plot_predicted_vs_actual.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# -- Configuration -------------------------------------------------------------

INPUT_CSV  = "regression_results_lat_temp.csv"
OUTPUT_PNG = "plot_predicted_vs_actual.png"

TITLE      = "Predicted vs. Actual Temperature\n(Vulpes vulpes, Linear Regression: Latitude → Temperature)"
X_LABEL    = "Actual Mean Annual Temperature (°C)"
Y_LABEL    = "Predicted Mean Annual Temperature (°C)"

# -- Load ----------------------------------------------------------------------

df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df):,} test set records.")

# -- Plot ----------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(
    df["actual_temp_c"],
    df["predicted_temp_c"],
    alpha=0.4,
    s=20,
    color="tomato",
    edgecolors="none",
    label="Test set predictions",
)

# 1:1 perfect prediction line
min_val = min(df["actual_temp_c"].min(), df["predicted_temp_c"].min())
max_val = max(df["actual_temp_c"].max(), df["predicted_temp_c"].max())
ax.plot(
    [min_val, max_val],
    [min_val, max_val],
    color="steelblue",
    linewidth=1.2,
    linestyle="--",
    label="Perfect prediction (1:1 line)",
)

# Labels, title, legend
ax.set_xlabel(X_LABEL, fontsize=12)
ax.set_ylabel(Y_LABEL, fontsize=12)
ax.set_title(TITLE, fontsize=13)
ax.legend(fontsize=9)

# Equal axes so the 1:1 line sits at 45°
ax.set_aspect("equal", adjustable="box")

# Grid
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
ax.grid(True, which="minor", linestyle=":",  linewidth=0.3, alpha=0.4)

# Record count annotation
ax.annotate(
    f"n = {len(df):,}",
    xy=(0.97, 0.03),
    xycoords="axes fraction",
    ha="right", va="bottom",
    fontsize=10,
    color="gray",
)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150)
print(f"Plot saved: {OUTPUT_PNG}")
plt.show()
