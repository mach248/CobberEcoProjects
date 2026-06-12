"""
add_temperature.py
------------------
Extracts WorldClim BIO1 (mean annual temperature) from the already-
downloaded wc2.1_5m_bio.zip and joins it to the enriched occurrence
CSV that already has annual precipitation (BIO12).

WorldClim stores BIO1 in units of °C * 10 (e.g. 143 = 14.3°C).
This script converts to °C automatically.

Input  : vulpes_vulpes_red_fox_GBIF_with_precip.csv
         wc2.1_5m_bio.zip  (already on disk from previous step)
Output : vulpes_vulpes_red_fox_GBIF_with_climate.csv
"""

import os
import zipfile
import numpy as np
import pandas as pd
import rasterio

# -- Configuration -------------------------------------------------------------

INPUT_CSV   = "vulpes_vulpes_red_fox_GBIF_with_precip.csv"
OUTPUT_CSV  = "vulpes_vulpes_red_fox_GBIF_with_climate.csv"
ZIP_FILE    = "wc2.1_5m_bio.zip"
RASTER_FILE = "wc2.1_5m_bio_1.tif"    # BIO1 = mean annual temperature

# -- Extract BIO1 TIF from zip if not already done ----------------------------

if os.path.exists(RASTER_FILE):
    print(f"Raster already present: {RASTER_FILE} — skipping extraction.")
else:
    print(f"Extracting {RASTER_FILE} from {ZIP_FILE} ...")
    with zipfile.ZipFile(ZIP_FILE, "r") as z:
        # List contents so we can find the exact BIO1 filename
        all_files = z.namelist()
        bio1_name = next((f for f in all_files if "bio_1.tif" in f), None)
        if bio1_name is None:
            raise FileNotFoundError(
                f"Could not find a BIO1 TIF inside {ZIP_FILE}.\n"
                f"Files found: {all_files}"
            )
        # Extract just this one file, then rename to our expected name
        z.extract(bio1_name, ".")
        if bio1_name != RASTER_FILE:
            os.rename(bio1_name, RASTER_FILE)
    print(f"Extracted: {RASTER_FILE}")

# -- Load CSV ------------------------------------------------------------------

print(f"\nLoading {INPUT_CSV} ...")
df = pd.read_csv(INPUT_CSV)
print(f"  {len(df):,} records loaded.")

before = len(df)
df = df.dropna(subset=["latitude", "longitude"])
if len(df) < before:
    print(f"  Dropped {before - len(df)} rows with missing coordinates.")

# -- Sample raster at each point ----------------------------------------------

print(f"Sampling temperature raster at {len(df):,} locations ...")

with rasterio.open(RASTER_FILE) as src:
    nodata = src.nodata
    coords = list(zip(df["longitude"], df["latitude"]))  # (x, y) order
    sampled = list(src.sample(coords))

temp_values = []
for val in sampled:
    v = float(val[0])
    if nodata is not None and v == nodata:
        temp_values.append(np.nan)
    else:
        # WorldClim BIO1 is stored as °C × 10 — convert to °C
        temp_values.append(round(v / 10.0, 2))

df["mean_annual_temp_c"] = temp_values

# -- Report -------------------------------------------------------------------

n_missing = df["mean_annual_temp_c"].isna().sum()
print(f"  Temperature extracted for {len(df) - n_missing:,} records.")
if n_missing:
    print(f"  {n_missing} records fell on nodata cells and have NaN temperature.")

print(f"\nTemperature summary (°C):")
print(df["mean_annual_temp_c"].describe().round(2).to_string())

# -- Save ---------------------------------------------------------------------

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nEnriched CSV saved: {OUTPUT_CSV}")
print(f"  Columns: {', '.join(df.columns.tolist())}")
