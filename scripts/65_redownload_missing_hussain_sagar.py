import sys
from pathlib import Path

import geopandas as gpd
import ee


# ============================================================
# STEP 65 - RE-DOWNLOAD 6 PROBLEMATIC HUSSAIN SAGAR IMAGES
# ============================================================

print("=" * 70)
print("STEP 65 - RE-DOWNLOAD MISSING/INVALID HUSSAIN SAGAR DATA")
print("=" * 70)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ============================================================
# IMPORT EXISTING GEE UTILITIES
# ============================================================

from utils.gee_utils import (
    initialize_gee,
    get_masked_sentinel2_collection,
    get_image_for_date,
    create_14_channel_image,
    download_14_channel_image,
)


# ============================================================
# PATHS
# ============================================================

BOUNDARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "hab_masks"
    / "training_waterbodies"
    / "Hussain_Sagar_boundary.geojson"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sentinel2"
    / "multi_waterbody"
    / "Hussain_Sagar"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SIX PROBLEMATIC DATES
# ============================================================

DATES = [
    "2017-03-25",
    "2017-04-14",
    "2017-05-04",
    "2017-05-24",
    "2017-09-01",
    "2017-11-20",
]


# ============================================================
# INITIALIZE GEE
# ============================================================

print()
print("Initializing Google Earth Engine...")

initialize_gee()

print(
    "Google Earth Engine initialized successfully."
)


# ============================================================
# LOAD HUSSAIN SAGAR BOUNDARY
# ============================================================

print()
print("Loading Hussain Sagar boundary...")


if not BOUNDARY_FILE.exists():

    raise FileNotFoundError(
        f"Boundary not found:\n{BOUNDARY_FILE}"
    )


gdf = gpd.read_file(
    BOUNDARY_FILE
)


if gdf.empty:

    raise ValueError(
        "Hussain Sagar boundary is empty."
    )


geometry = gdf.geometry.union_all()

ee_geometry = ee.Geometry(
    geometry.__geo_interface__
)


print(
    "Hussain Sagar boundary loaded."
)


# ============================================================
# CREATE GEE COLLECTION
# ============================================================

print()
print("Creating Sentinel-2 collection...")


collection = get_masked_sentinel2_collection(
    ee_geometry,
    "2016-01-01",
    "2026-01-01",
    cloud_threshold=40
)


print(
    "Collection ready."
)


# ============================================================
# PROCESS SIX DATES
# ============================================================

successful = 0
failed = 0


for index, date in enumerate(
    DATES,
    start=1
):

    print()
    print("=" * 70)

    print(
        f"[{index}/{len(DATES)}] {date}"
    )

    print("=" * 70)


    output_file = (
        OUTPUT_DIR
        / f"{date}.tif"
    )


    try:

        # ----------------------------------------------------
        # Get image
        # ----------------------------------------------------

        print(
            "Finding Sentinel-2 image..."
        )

        image = get_image_for_date(
            collection,
            date
        )

        print(
            "Sentinel-2 image found."
        )


        # ----------------------------------------------------
        # Create 14 channels
        # ----------------------------------------------------

        print(
            "Creating 14-channel image..."
        )

        image_14 = create_14_channel_image(
            image
        )

        print(
            "14-channel image created."
        )


        # ----------------------------------------------------
        # Download
        # ----------------------------------------------------

        print(
            "Downloading image..."
        )

        download_14_channel_image(
            image_14,
            ee_geometry,
            str(output_file),
            scale=10
        )


        if not output_file.exists():

            raise RuntimeError(
                "Download finished but output file "
                "was not created."
            )


        print(
            f"SUCCESS: {output_file}"
        )


        successful += 1


    except Exception as e:

        print()
        print(
            "FAILED:"
        )

        print(e)

        failed += 1


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("STEP 65 COMPLETE")
print("=" * 70)

print()

print(
    f"Attempted : {len(DATES)}"
)

print(
    f"Successful: {successful}"
)

print(
    f"Failed    : {failed}"
)

print()

print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)

print()
print("=" * 70)