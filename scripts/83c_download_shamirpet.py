import os
import sys
import pandas as pd
import geopandas as gpd
import ee

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_ROOT)

from utils.gee_utils import (
    initialize_gee,
    get_masked_sentinel2_collection,
    get_image_for_date,
    create_14_channel_image,
    download_14_channel_image
)

# ============================================================
# PATHS
# ============================================================

BOUNDARY_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "labels",
    "hab_masks",
    "training_waterbodies",
    "Shamirpet_Lake_OSM_boundary.geojson"
)

DATES_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "shamirpet_test_dates.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "sentinel2",
    "shamirpet_test"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

START_DATE = "2016-01-01"
END_DATE = "2026-01-01"
CLOUD_THRESHOLD = 40
SCALE = 10

# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 83C - CORRECTED SHAMIRPET 14-BAND DOWNLOAD")
print("=" * 70)

initialize_gee()

# ============================================================
# LOAD BOUNDARY
# ============================================================

print("\nLoading Shamirpet boundary...")

gdf = gpd.read_file(BOUNDARY_FILE)

if gdf.empty:
    raise ValueError("Shamirpet boundary is empty.")

geometry = gdf.geometry.unary_union

if geometry is None or geometry.is_empty:
    raise ValueError("Shamirpet geometry is empty.")

aoi = ee.Geometry(geometry.__geo_interface__)

print("Boundary loaded successfully.")

# ============================================================
# LOAD SELECTED DATES
# ============================================================

dates_df = pd.read_csv(DATES_FILE)

dates = dates_df["date"].astype(str).tolist()

print("\nSelected test dates:", len(dates))

# ============================================================
# CREATE CLOUD-MASKED COLLECTION
# ============================================================

print("\nCreating Sentinel-2 collection...")

collection = get_masked_sentinel2_collection(
    aoi,
    START_DATE,
    END_DATE,
    cloud_threshold=CLOUD_THRESHOLD
)

print("Collection ready.")

# ============================================================
# DOWNLOAD
# ============================================================

successful = 0
failed = 0

for index, date in enumerate(dates, start=1):

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{date}.tif"
    )

    print("\n" + "=" * 70)
    print(f"[{index}/{len(dates)}] {date}")
    print("=" * 70)

    try:

        # ----------------------------------------------------
        # FIND SENTINEL-2 IMAGE
        # ----------------------------------------------------

        print("Finding Sentinel-2 image...")

        image = get_image_for_date(
            collection,
            date
        )

        if image is None:
            print("No image found.")
            failed += 1
            continue

        print("Sentinel-2 image found.")

        # ----------------------------------------------------
        # CREATE EXACT 14 CHANNELS
        # ----------------------------------------------------

        print("Creating 14-channel image...")

        image_14 = create_14_channel_image(
            image
        )

        print("14-channel image created.")

        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        print("Downloading 14-band GeoTIFF...")

        download_14_channel_image(
            image_14,
            aoi,
            output_path,
            scale=SCALE
        )

        if not os.path.exists(output_path):

            print("ERROR: Output file not found.")
            failed += 1
            continue

        # ----------------------------------------------------
        # VERIFY IMMEDIATELY
        # ----------------------------------------------------

        import rasterio

        with rasterio.open(output_path) as src:

            band_count = src.count

        if band_count != 14:

            print(
                f"ERROR: Downloaded file has "
                f"{band_count} bands instead of 14."
            )

            # Remove incorrect file
            os.remove(output_path)

            failed += 1
            continue

        print("SUCCESS: 14-band image saved.")

        successful += 1

    except Exception as e:

        print("DOWNLOAD FAILED:")
        print(e)

        failed += 1

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 83C CORRECTED SUMMARY")
print("=" * 70)

print("Selected dates :", len(dates))
print("Downloaded     :", successful)
print("Failed         :", failed)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\n" + "=" * 70)

if successful == len(dates) and failed == 0:

    print("ALL SHAMIRPET IMAGES SUCCESSFULLY DOWNLOADED AS 14-BAND TIFFs")

else:

    print("SOME FILES FAILED - DO NOT PROCEED TO LABEL CREATION YET")

print("=" * 70)