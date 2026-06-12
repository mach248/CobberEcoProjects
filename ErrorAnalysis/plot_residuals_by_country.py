"""
plot_residuals_by_country.py
-----------------------------
Merges the place_or_country grouping variable from the original
climate CSV into the regression results dataframe, then creates
a grouped box plot showing the distribution of residuals for
each country/place.

A horizontal line at zero marks perfect prediction. Countries
where the box sits above zero are systematically under-predicted;
countries where it sits below zero are systematically over-predicted.

Inputs : regression_results_lat_temp.csv
         vulpes_vulpes_red_fox_GBIF_with_climate.csv
Output : plot_residuals_by_country.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# -- Configuration -------------------------------------------------------------

RESULTS_CSV = "regression_results_lat_temp.csv"
CLIMATE_CSV = "vulpes_vulpes_red_fox_GBIF_with_climate.csv"
OUTPUT_PNG  = "plot_residuals_by_country.png"

TITLE   = "Residuals by Country/Place\n(Vulpes vulpes, Latitude → Temperature Model)"
X_LABEL = "Country / Place"
Y_LABEL = "Residual (Actual − Predicted) (°C)"

# Minimum number of records a group needs to appear in the plot
# (groups with very few records make unreliable boxes)
MIN_GROUP_SIZE = 3

# -- Load and merge ------------------------------------------------------------

results = pd.read_csv(RESULTS_CSV)
climate = pd.read_csv(CLIMATE_CSV)

print(f"Results rows  : {len(results):,}")
print(f"Climate rows  : {len(climate):,}")

# The regression script used an 80/20 split with random_state=42.
# sklearn preserves the original DataFrame index in the test set,
# so we can use that index to pull the matching rows from the climate CSV.
climate_reset = climate.reset_index(drop=True)

# Attach the index back onto results so we can merge on it
results.index.name = "original_index"
results = results.reset_index()

# Pull place_or_country from the climate CSV using the saved index
results["place_or_country"] = climate_reset.loc[
    results["original_index"], "place_or_country"
].values

print(f"Unique countries/places : {results['place_or_country'].nunique()}")

# -- Filter to groups with enough records -------------------------------------

group_counts = results["place_or_country"].value_counts()
valid_groups = group_counts[group_counts >= MIN_GROUP_SIZE].index
filtered     = results[results["place_or_country"].isin(valid_groups)].copy()

print(f"Groups with >= {MIN_GROUP_SIZE} records : {len(valid_groups)}")
print(f"Records after filtering : {len(filtered):,}")

# -- Order groups by median residual (worst bias to best) ---------------------

group_order = (
    filtered.groupby("place_or_country")["residual"]
    .median()
    .sort_values()
    .index.tolist()
)

# -- Build data for boxplot ----------------------------------------------------

grouped_data = [
    filtered.loc[filtered["place_or_country"] == g, "residual"].values
    for g in group_order
]

# -- Plot ----------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(max(8, len(group_order) * 0.9), 6))

bp = ax.boxplot(
    grouped_data,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=1.5),
    flierprops=dict(marker="o", markersize=3, alpha=0.4),
)

# Color boxes: above-zero median = tomato, below-zero median = steelblue
for patch, group in zip(bp["boxes"], group_order):
    median = filtered.loc[
        filtered["place_or_country"] == group, "residual"
    ].median()
    patch.set_facecolor("tomato" if median >= 0 else "steelblue")
    patch.set_alpha(0.7)

# Zero reference line
ax.axhline(0, color="black", linewidth=1.2, linestyle="--", label="Zero (perfect prediction)")

# X-axis labels
ax.set_xticks(range(1, len(group_order) + 1))
ax.set_xticklabels(group_order, rotation=45, ha="right", fontsize=9)

# Labels, title, legend
ax.set_xlabel(X_LABEL, fontsize=12)
ax.set_ylabel(Y_LABEL, fontsize=12)
ax.set_title(TITLE, fontsize=13)
ax.legend(fontsize=9)

# Grid
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
ax.grid(True, which="minor", axis="y", linestyle=":",  linewidth=0.3, alpha=0.4)

# Record count annotation
ax.annotate(
    f"n = {len(filtered):,}",
    xy=(0.99, 0.97),
    xycoords="axes fraction",
    ha="right", va="top",
    fontsize=10,
    color="gray",
)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"\nPlot saved: {OUTPUT_PNG}")
plt.show()
