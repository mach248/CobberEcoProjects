"""
join_worldclim_precipitation.py
--------------------------------
Downloads the WorldClim v2.1 BIO12 (annual precipitation) raster at
5 arcmin resolution, extracts the precipitation value at each fox
sighting's lat/lon, and saves an enriched CSV ready for analysis.

Steps:
  1. Download wc2.1_5m_bio_12.tif from WorldClim (one-time, ~400 MB)
  2. Open the GeoTIFF with rasterio
  3. Sample the raster at each (longitude, latitude) point
  4. Add a 'annual_precip_mm' column to the DataFrame
  5. Save the enriched CSV

Inputs : vulpes_vulpes_red_fox_GBIF_occurrences.csv
Outputs: vulpes_vulpes_red_fox_GBIF_with_precip.csv
         wc2.1_5m_bio_12.tif  (cached raster — kept for reuse)

WorldClim citation:
  Fick & Hijmans (2017) WorldClim 2. Int. J. Climatology 37:4302-4315.
  https://www.worldclim.org/data/worldclim21.html
"""

import os
import zipfile
import requests
import numpy as np
import pandas as pd
import rasterio

# -- Configuration -------------------------------------------------------------

INPUT_CSV   = "vulpes_vulpes_red_fox_GBIF_occurrences.csv"
OUTPUT_CSV  = "vulpes_vulpes_red_fox_GBIF_with_precip.csv"
RASTER_FILE = "wc2.1_5m_bio_12.tif"
ZIP_FILE    = "wc2.1_5m_bio_12.zip"

# WorldClim v2.1, 5 arcmin, all bioclim variables (BIO1-BIO19)
# BIO12 (annual precipitation) is one of the TIFs inside this zip
WORLDCLIM_URL = (
    "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/"
    "wc2.1_5m_bio.zip"
)
ZIP_FILE    = "wc2.1_5m_bio.zip"
RASTER_FILE = "wc2.1_5m_bio_12.tif"   # BIO12 = annual precipitation (mm)

# -- Step 1: Download raster (skip if already cached) -------------------------

if os.path.exists(RASTER_FILE):
    print(f"Raster already present: {RASTER_FILE} — skipping download.")
else:
    print(f"Downloading WorldClim BIO12 (5 arcmin) — this may take a few minutes...")
    with requests.get(WORLDCLIM_URL, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(ZIP_FILE, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"  {pct:5.1f}%  ({downloaded/1e6:.0f} / {total/1e6:.0f} MB)", end="\r")
    print(f"\nDownload complete. Extracting {ZIP_FILE} ...")
    with zipfile.ZipFile(ZIP_FILE, "r") as z:
        z.extractall(".")
    print(f"Extracted: {RASTER_FILE}")

# -- Step 2: Load occurrence CSV -----------------------------------------------

print(f"\nLoading {INPUT_CSV} ...")
df = pd.read_csv(INPUT_CSV)
print(f"  {len(df):,} records loaded.")

# Drop rows without coordinates (shouldn't be any given our GBIF filter,
# but defensive check in case the CSV was edited manually)
before = len(df)
df = df.dropna(subset=["latitude", "longitude"])
if len(df) < before:
    print(f"  Dropped {before - len(df)} rows with missing coordinates.")

# -- Step 3: Sample raster at each point --------------------------------------

print(f"\nSampling precipitation raster at {len(df):,} locations ...")

with rasterio.open(RASTER_FILE) as src:
    nodata = src.nodata   # value used by WorldClim to mark ocean/missing cells

    # rasterio.sample expects an iterable of (x, y) = (longitude, latitude)
    coords = list(zip(df["longitude"], df["latitude"]))
    sampled = list(src.sample(coords))   # returns list of 1-element arrays

# Flatten to a plain list of floats; replace nodata with NaN
precip_values = []
for val in sampled:
    v = float(val[0])
    if nodata is not None and v == nodata:
        precip_values.append(np.nan)
    else:
        precip_values.append(v)

df["annual_precip_mm"] = precip_values

# -- Step 4: Report -----------------------------------------------------------

n_missing = df["annual_precip_mm"].isna().sum()
print(f"  Precipitation extracted for {len(df) - n_missing:,} records.")
if n_missing > 0:
    print(f"  {n_missing} records fell on ocean/nodata cells and have NaN precipitation.")

print(f"\nPrecipitation summary (mm/year):")
print(df["annual_precip_mm"].describe().round(1).to_string())

# -- Step 5: Save enriched CSV ------------------------------------------------

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nEnriched CSV saved: {OUTPUT_CSV}")
print(f"  Columns: {', '.join(df.columns.tolist())}")

# -- Step 6: Preview ----------------------------------------------------------

print(f"\nFirst 5 rows of enriched data:")
print("-" * 60)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
print(df[["scientific_name", "event_date", "latitude", "longitude",
          "place_or_country", "annual_precip_mm"]].head().to_string(index=False))

"""
scatter_lat_precip.py
----------------------
Loads the enriched GBIF occurrence CSV and produces a scatter plot
of latitude vs. annual precipitation (WorldClim BIO12) for
red fox (Vulpes vulpes) sightings.

This is an exploratory plot to visually assess whether a linear
relationship between the two variables looks plausible before
fitting a model.

Input : vulpes_vulpes_red_fox_GBIF_with_precip.csv
Output: scatter_lat_vs_precip.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# -- Configuration -------------------------------------------------------------

INPUT_CSV  = "vulpes_vulpes_red_fox_GBIF_with_precip.csv"
OUTPUT_PNG = "scatter_lat_vs_precip.png"

X_COL      = "latitude"
Y_COL      = "annual_precip_mm"
X_LABEL    = "Latitude (°N)"
Y_LABEL    = "Annual Precipitation (mm)"
TITLE      = "Red Fox Sightings: Latitude vs. Annual Precipitation\n(Vulpes vulpes, GBIF 2000–2025)"

# -- Load and clean ------------------------------------------------------------

df = pd.read_csv(INPUT_CSV)

# Drop rows where either variable is missing
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
    alpha=0.4,          # transparency so overlapping points stay visible
    s=20,               # point size
    color="steelblue",
    edgecolors="none",
)

# Labels and title
ax.set_xlabel(X_LABEL, fontsize=12)
ax.set_ylabel(Y_LABEL, fontsize=12)
ax.set_title(TITLE, fontsize=13)

# Light grid for readability
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
ax.grid(True, which="minor", linestyle=":",  linewidth=0.3, alpha=0.4)

# Annotation: record count in corner
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
