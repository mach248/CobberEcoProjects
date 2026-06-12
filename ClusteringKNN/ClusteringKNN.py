"""
species_summary.py
------------------
Loads the Portal ecology dataset (surveys, species, plots), merges
them, filters to rodents only, and builds a per-species summary table.

Summary columns (one row per species):
  - mean_weight          : mean body weight (g)
  - mean_hindfoot_length : mean hindfoot length (mm)
  - n_records            : number of observation records
  - n_plots              : number of distinct plots the species was seen in
  - n_years              : number of distinct years the species was observed

Species with missing mean weight OR mean hindfoot length are dropped.

Inputs : surveys.csv, species.csv, plots.csv
Output : rodent_species_summary.csv
"""

import pandas as pd

# -- Configuration -------------------------------------------------------------

SURVEYS_CSV = "surveys.csv"
SPECIES_CSV = "species.csv"
PLOTS_CSV   = "plots.csv"
OUTPUT_CSV  = "rodent_species_summary.csv"

# -- Load ----------------------------------------------------------------------

print("Loading CSV files ...")
surveys = pd.read_csv(SURVEYS_CSV)
species = pd.read_csv(SPECIES_CSV)
plots   = pd.read_csv(PLOTS_CSV)

print(f"  surveys : {len(surveys):,} rows")
print(f"  species : {len(species):,} rows")
print(f"  plots   : {len(plots):,} rows")

# -- Merge ---------------------------------------------------------------------

# surveys + species on species_id, then + plots on plot_id
df = surveys.merge(species, on="species_id", how="left")
df = df.merge(plots, on="plot_id", how="left")
print(f"\nMerged dataset: {len(df):,} rows, {len(df.columns)} columns")

# -- Filter to rodents only ----------------------------------------------------

rodents = df[df["taxa"] == "Rodent"].copy()
print(f"Rodent records: {len(rodents):,}")

# -- Build per-species summary -------------------------------------------------

# Create a readable species name by combining genus + species
rodents["species_name"] = rodents["genus"] + " " + rodents["species"]

summary = rodents.groupby("species_name").agg(
    species_id           = ("species_id",      "first"),  # short code, e.g. DM
    mean_weight          = ("weight",          "mean"),
    mean_hindfoot_length = ("hindfoot_length", "mean"),
    n_records            = ("record_id",        "count"),
    n_plots              = ("plot_id",          "nunique"),
    n_years              = ("year",             "nunique"),
).reset_index()

# Round the means for readability
summary["mean_weight"]          = summary["mean_weight"].round(2)
summary["mean_hindfoot_length"] = summary["mean_hindfoot_length"].round(2)

print(f"\nSpecies before dropping missing values : {len(summary)}")

# -- Drop species missing either mean -----------------------------------------

summary = summary.dropna(subset=["mean_weight", "mean_hindfoot_length"])
print(f"Species after dropping missing values  : {len(summary)}")

# -- Sort by record count (most observed first) -------------------------------

summary = summary.sort_values("n_records", ascending=False).reset_index(drop=True)

# -- Display and save ----------------------------------------------------------

print("\nSpecies summary table:")
print("=" * 80)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
print(summary.to_string(index=False))

summary.to_csv(OUTPUT_CSV, index=False)
print(f"\nSummary saved: {OUTPUT_CSV}")
