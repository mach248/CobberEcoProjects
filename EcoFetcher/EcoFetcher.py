"""
gbif_red_fox.py
---------------
Fetches up to 300 occurrence records for red fox (Vulpes vulpes)
from the GBIF Occurrence Search API.

Filters applied:
  - Records with coordinates only (hasCoordinate=true)
  - Observation year range: 2000-2025

Output:
  - Saves all fetched records to a CSV file
  - Prints all field names from the first record for exploration
  - Prints a preview of the first 10 records in the terminal

GBIF API docs: https://www.gbif.org/developer/occurrence
No API key or authentication required for read-only searches.
"""

import csv
import requests

# -- Configuration -------------------------------------------------------------

GBIF_URL      = "https://api.gbif.org/v1/occurrence/search"
SPECIES       = "Vulpes vulpes"
MAX_RESULTS   = 300
YEAR_FROM     = 2000
YEAR_TO       = 2025
PREVIEW_ROWS  = 10
OUTPUT_CSV    = "vulpes_vulpes_red_fox_GBIF_occurrences.csv"

# Column names for the CSV (and terminal preview), mapped to GBIF field names
COLUMNS = {
    "scientific_name" : "scientificName",
    "common_name"     : "vernacularName",
    "event_date"      : "eventDate",
    "latitude"        : "decimalLatitude",
    "longitude"       : "decimalLongitude",
    "place_or_country": None,               # derived from multiple fields; see below
    "gbif_url"        : None,               # derived from key; see below
}

# -- Fetch data ----------------------------------------------------------------

params = {
    "scientificName": SPECIES,           # filter to this species
    "limit": MAX_RESULTS,                # return at most 300 records
    "hasCoordinate": "true",             # only records with lat/lon
    "year": f"{YEAR_FROM},{YEAR_TO}",    # GBIF range filter: "2000,2025"
}

response = requests.get(GBIF_URL, params=params)
response.raise_for_status()   # raise an error for bad HTTP status codes

data    = response.json()
results = data.get("results", [])

if not results:
    print("No records found.")
else:
    # -- Print all field names from the first record --------------------------
    first_record = results[0]
    field_names  = sorted(first_record.keys())

    print(f"Fields available in the first record ({len(field_names)} total):")
    print("-" * 60)
    for name in field_names:
        print(f"  {name}")
    print("=" * 60)

    # -- Helper: extract the selected fields from one record ------------------
    def extract(record):
        """Return an ordered dict of the fields we care about."""
        gbif_key = record.get("key")
        return {
            "scientific_name" : record.get("scientificName", ""),
            "common_name"     : record.get("vernacularName",  ""),
            "event_date"      : record.get("eventDate",       ""),
            "latitude"        : record.get("decimalLatitude",  ""),
            "longitude"       : record.get("decimalLongitude", ""),
            "place_or_country": (record.get("locality")
                                 or record.get("stateProvince")
                                 or record.get("country")
                                 or ""),
            "gbif_url"        : (f"https://www.gbif.org/occurrence/{gbif_key}"
                                 if gbif_key else ""),
        }

    rows = [extract(r) for r in results]

    # -- Save all records to CSV ----------------------------------------------
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} records to: {OUTPUT_CSV}")
    print("=" * 60)

    # -- Print first 10 records as a terminal preview -------------------------
    print(f"\nPreview — first {PREVIEW_ROWS} of {len(rows)} records for '{SPECIES}':\n")
    print("=" * 60)

    for i, row in enumerate(rows[:PREVIEW_ROWS], start=1):
        print(f"Record #{i}")
        print(f"  Scientific name  : {row['scientific_name']  or 'N/A'}")
        print(f"  Common name      : {row['common_name']      or 'N/A'}")
        print(f"  Date             : {row['event_date']       or 'N/A'}")
        print(f"  Latitude         : {row['latitude']         or 'N/A'}")
        print(f"  Longitude        : {row['longitude']        or 'N/A'}")
        print(f"  Place/Country    : {row['place_or_country'] or 'N/A'}")
        print(f"  GBIF URL         : {row['gbif_url']         or 'N/A'}")
        print("-" * 60)