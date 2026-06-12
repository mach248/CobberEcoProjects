"""
scatter_lat_temp.py
--------------------
Scatter plot of latitude vs. mean annual temperature for red fox
(Vulpes vulpes) sightings, using WorldClim BIO1 joined to GBIF records.

Latitude vs. temperature has a strong, well-known negative linear
relationship — a good candidate for linear regression.

Input  : vulpes_vulpes_red_fox_GBIF_with_climate.csv
Output : scatter_lat_vs_temp.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# -- Configuration -------------------------------------------------------------

INPUT_CSV  = "vulpes_vulpes_red_fox_GBIF_with_climate.csv"
OUTPUT_PNG = "scatter_lat_vs_temp.png"

X_COL   = "latitude"
Y_COL   = "mean_annual_temp_c"
X_LABEL = "Latitude (°N)"
Y_LABEL = "Mean Annual Temperature (°C)"
TITLE   = "Red Fox Sightings: Latitude vs. Mean Annual Temperature\n(Vulpes vulpes, GBIF 2000–2025, WorldClim BIO1)"

# -- Load and clean ------------------------------------------------------------

df = pd.read_csv(INPUT_CSV)

before = len(df)
df = df.dropna(subset=[X_COL, Y_COL])
dropped = before - len(df)
if dropped:
    print(f"Dropped {dropped} rows with missing values.")
print(f"Plotting {len(df):,} records.")

# -- Plot ----------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    df[X_COL],
    df[Y_COL],
    alpha=0.4,
    s=20,
    color="tomato",
    edgecolors="none",
)

# Labels and title
ax.set_xlabel(X_LABEL, fontsize=12)
ax.set_ylabel(Y_LABEL, fontsize=12)
ax.set_title(TITLE, fontsize=13)

# Reference line at 0°C — useful ecological threshold
ax.axhline(0, color="steelblue", linewidth=0.8, linestyle="--", alpha=0.6,
           label="0°C")
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
