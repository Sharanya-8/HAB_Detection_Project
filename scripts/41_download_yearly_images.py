"""
Step 41: Download one representative Sentinel-2 image per year.

Uses the dates selected by Step 40.

Output:
data/raw/sentinel2/historical/

Each image contains the 14 channels:
B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12,
NDWI, MNDWI, NDCI, FAI
"""

from pathlib import Path
import sys
import pandas as pd
import ee


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.gee_utils import (
    initialize_gee,
    get_sentinel2_collection,
    get_cloud_probability_collection,
    join_cloud_probability,
    mask_sentinel2_clouds,
    get_image_for_date,
    create_14_channel_image,
    download_14_channel_image,
)


# =========================================================
# CONFIGURATION
# =========================================================

WATERBODY_NAME = "Hussain Sagar"

DATES_FILE = (
    PROJECT_ROOT
    / "results"
    / "historical"
    / "yearly_representative_dates.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sentinel2"
    / "historical"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Hussain Sagar boundary
BOUNDARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "hab_masks"
    / "Hussain_Sagar_boundary.geojson"
)


# =========================================================
# START
# =========================================================

print("=" * 75)
print("STEP 41 - DOWNLOAD YEARLY SENTINEL-2 IMAGES")
print("=" * 75)

print(f"\nWaterbody: {WATERBODY_NAME}")
print(f"Dates file: {DATES_FILE}")
print(f"Output folder: {OUTPUT_DIR}")


# =========================================================
# CHECK FILES
# =========================================================

if not DATES_FILE.exists():

    raise FileNotFoundError(
        f"\nCould not find:\n{DATES_FILE}\n"
        "Run Step 40 first."
    )


if not BOUNDARY_FILE.exists():

    raise FileNotFoundError(
        f"\nCould not find:\n{BOUNDARY_FILE}"
    )


# =========================================================
# INITIALIZE GEE
# =========================================================

print("\nInitializing Google Earth Engine...")

initialize_gee()

print("GEE initialized successfully.")


# =========================================================
# LOAD YEARLY DATES
# =========================================================

dates_df = pd.read_csv(DATES_FILE)

dates_df = dates_df.dropna(
    subset=["selected_date"]
)

print(
    f"\nFound {len(dates_df)} yearly dates to download."
)


# =========================================================
# LOAD WATERBODY BOUNDARY
# =========================================================

import json

with open(
    BOUNDARY_FILE,
    "r",
    encoding="utf-8"
) as file:

    boundary_geojson = json.load(file)


# Use the first feature geometry
geometry = boundary_geojson["features"][0]["geometry"]

# Convert GeoJSON geometry to Earth Engine geometry
if geometry["type"] == "Polygon":

    aoi = ee.Geometry.Polygon(
        geometry["coordinates"]
    )

elif geometry["type"] == "MultiPolygon":

    aoi = ee.Geometry.MultiPolygon(
        geometry["coordinates"]
    )

else:

    raise ValueError(
        f"Unsupported geometry type: {geometry['type']}"
    )


print("Waterbody boundary loaded successfully.")


# =========================================================
# DOWNLOAD EACH YEAR
# =========================================================

successful = []
failed = []


for _, row in dates_df.iterrows():

    year = int(row["year"])
    selected_date = str(row["selected_date"])

    print("\n" + "-" * 75)
    print(f"YEAR: {year}")
    print(f"DATE: {selected_date}")
    print("-" * 75)

    output_file = (
        OUTPUT_DIR
        / f"Hussain_Sagar_{year}_{selected_date}_14Channel.tif"
    )

    # -----------------------------------------------------
    # Skip if already downloaded
    # -----------------------------------------------------

    if output_file.exists():

        print(
            f"Already exists: {output_file.name}"
        )

        successful.append(
            (year, selected_date, "already_exists")
        )

        continue

    try:

        # -------------------------------------------------
        # Get Sentinel-2 collection
        # -------------------------------------------------

        print("Getting Sentinel-2 collection...")

        start_date = selected_date
        end_date = (
            ee.Date(selected_date)
            .advance(1, "day")
            .format("YYYY-MM-dd")
            .getInfo()
        )

        collection = (
            get_sentinel2_collection(
                aoi,
                start_date,
                end_date
            )
        )

        # -------------------------------------------------
        # Cloud probability collection
        # -------------------------------------------------

        print("Getting cloud probability data...")

        cloud_collection = (
            get_cloud_probability_collection(
                aoi,
                start_date,
                end_date
            )
        )

        # -------------------------------------------------
        # Join cloud probability
        # -------------------------------------------------

        print("Joining cloud probability...")

        joined_collection = join_cloud_probability(
            collection,
            cloud_collection
        )

        # -------------------------------------------------
        # Apply cloud masking
        # -------------------------------------------------

        print("Applying cloud mask...")

        masked_collection = (
            joined_collection.map(
                lambda image:
                mask_sentinel2_clouds(
                    image,
                    threshold=40
                )
            )
        )

        # -------------------------------------------------
        # Get image for selected date
        # -------------------------------------------------

        print("Creating same-day image...")

        image = get_image_for_date(
            masked_collection,
            selected_date
        )

        # -------------------------------------------------
        # Create 14 channels
        # -------------------------------------------------

        print("Creating 14-channel image...")

        image_14 = create_14_channel_image(
            image
        )

        # -------------------------------------------------
        # Download
        # -------------------------------------------------

        print("Downloading image...")

        download_14_channel_image(
            image_14,
            aoi,
            str(output_file)
        )

        print(
            f"SUCCESS: {output_file.name}"
        )

        successful.append(
            (year, selected_date, "downloaded")
        )

    except Exception as error:

        print(
            f"FAILED for {year}: {error}"
        )

        failed.append(
            (year, selected_date, str(error))
        )


# =========================================================
# SUMMARY
# =========================================================

print("\n")
print("=" * 75)
print("STEP 41 SUMMARY")
print("=" * 75)

print(
    f"\nSuccessful: {len(successful)}"
)

print(
    f"Failed:     {len(failed)}"
)


if successful:

    print("\nSuccessful downloads:")

    for year, date, status in successful:

        print(
            f"  {year} -> {date} -> {status}"
        )


if failed:

    print("\nFailed downloads:")

    for year, date, error in failed:

        print(
            f"  {year} -> {date}"
        )

        print(
            f"     Error: {error}"
        )


print("\nOutput folder:")
print(OUTPUT_DIR)

print("\nStep 41 completed.")