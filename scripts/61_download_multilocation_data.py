import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd
import ee


# ============================================================
# STEP 61 - DOWNLOAD MULTI-WATERBODY SENTINEL-2 DATA
# ============================================================

print("=" * 70)
print("STEP 61 - DOWNLOAD MULTI-WATERBODY SENTINEL-2 DATA")
print("=" * 70)


# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ------------------------------------------------------------
# Import existing GEE utilities
# ------------------------------------------------------------

from utils.gee_utils import (
    initialize_gee,
    get_masked_sentinel2_collection,
    get_image_for_date,
    create_14_channel_image,
    download_14_channel_image,
)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

DATE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "selected_training_dates.csv"
)

BOUNDARY_DIR = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "hab_masks"
    / "training_waterbodies"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sentinel2"
    / "multi_waterbody"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Boundary files
# ------------------------------------------------------------

BOUNDARY_FILES = {

    "Hussain Sagar":
        "Hussain_Sagar_boundary.geojson",

    "Saroor Nagar":
        "Saroor_Nagar_OSM_boundary.geojson",

    "Osman Sagar":
        "Osman_Sagar_boundary.geojson",

    "Himayat Sagar":
        "Himayat_Sagar_relation_boundary.geojson",
}


# ------------------------------------------------------------
# Read selected dates
# ------------------------------------------------------------

if not DATE_FILE.exists():

    raise FileNotFoundError(
        f"Date selection file not found:\n{DATE_FILE}"
    )


df = pd.read_csv(
    DATE_FILE
)


required_columns = {
    "waterbody",
    "year",
    "date",
}

missing_columns = (
    required_columns
    - set(df.columns)
)

if missing_columns:

    raise ValueError(
        f"Missing columns in CSV: {missing_columns}"
    )


print()
print(
    f"Selected records: {len(df)}"
)


# ------------------------------------------------------------
# Initialize GEE
# ------------------------------------------------------------

print()
print(
    "Initializing Google Earth Engine..."
)

initialize_gee()

print(
    "Google Earth Engine initialized successfully."
)


# ============================================================
# CACHE BOUNDARIES
# ============================================================

print()
print("Loading waterbody boundaries...")


boundaries = {}


for waterbody, filename in BOUNDARY_FILES.items():

    boundary_file = (
        BOUNDARY_DIR
        / filename
    )

    if not boundary_file.exists():

        raise FileNotFoundError(
            f"Boundary file not found:\n{boundary_file}"
        )

    gdf = gpd.read_file(
        boundary_file
    )

    if gdf.empty:

        raise ValueError(
            f"Empty boundary:\n{boundary_file}"
        )

    geometry = gdf.geometry.union_all()

    ee_geometry = ee.Geometry(
        geometry.__geo_interface__
    )

    boundaries[waterbody] = ee_geometry

    print(
        f"Loaded boundary: {waterbody}"
    )


print(
    "All boundaries loaded successfully."
)


# ============================================================
# CACHE GEE COLLECTIONS
# ============================================================

print()
print("Creating Sentinel-2 collections...")


collections = {}


# We need imagery from 2016 through 2025.
# The selected dates stop at 2025.
START_DATE = "2016-01-01"
END_DATE = "2026-01-01"


for waterbody in BOUNDARY_FILES:

    print()
    print(
        f"Creating collection for {waterbody}..."
    )

    try:

        collections[waterbody] = (
            get_masked_sentinel2_collection(
                boundaries[waterbody],
                START_DATE,
                END_DATE,
                cloud_threshold=40
            )
        )

        print(
            f"Collection ready: {waterbody}"
        )

    except Exception as e:

        print(
            f"ERROR creating collection for "
            f"{waterbody}:"
        )

        print(e)

        raise


print()
print(
    "All Sentinel-2 collections ready."
)


# ============================================================
# DOWNLOAD
# ============================================================

successful = []
failed = []
skipped = []


for index, row in df.iterrows():

    waterbody = row["waterbody"]

    date = str(
        row["date"]
    )

    year = int(
        row["year"]
    )


    print()
    print("=" * 70)

    print(
        f"[{index + 1}/{len(df)}] "
        f"{waterbody} | {date}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Check waterbody
    # --------------------------------------------------------

    if waterbody not in collections:

        print(
            f"ERROR: Collection not available "
            f"for {waterbody}"
        )

        failed.append({
            "waterbody": waterbody,
            "year": year,
            "date": date,
            "reason": "collection_not_available"
        })

        continue


    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    waterbody_dir = (
        OUTPUT_DIR
        / waterbody.replace(
            " ",
            "_"
        )
    )

    waterbody_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Output file
    # --------------------------------------------------------

    output_file = (
        waterbody_dir
        / f"{date}.tif"
    )


    # --------------------------------------------------------
    # Skip existing
    # --------------------------------------------------------

    if output_file.exists():

        print(
            "Already exists - skipping."
        )

        skipped.append({
            "waterbody": waterbody,
            "year": year,
            "date": date
        })

        continue


    # --------------------------------------------------------
    # Get image for date
    # --------------------------------------------------------

    try:

        print(
            "Finding Sentinel-2 image..."
        )

        image = get_image_for_date(
            collections[waterbody],
            date
        )

        print(
            "Sentinel-2 image found."
        )

    except Exception as e:

        print(
            "ERROR finding image:"
        )

        print(e)

        failed.append({
            "waterbody": waterbody,
            "year": year,
            "date": date,
            "reason": str(e)
        })

        continue


    # --------------------------------------------------------
    # Create 14 channels
    # --------------------------------------------------------

    try:

        print(
            "Creating 14-channel image..."
        )

        image_14 = create_14_channel_image(
            image
        )

        print(
            "14-channel image created."
        )

    except Exception as e:

        print(
            "ERROR creating 14-channel image:"
        )

        print(e)

        failed.append({
            "waterbody": waterbody,
            "year": year,
            "date": date,
            "reason": str(e)
        })

        continue


    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    try:

        print(
            "Downloading 14-band GeoTIFF..."
        )

        download_14_channel_image(
            image_14,
            boundaries[waterbody],
            str(output_file),
            scale=10
        )


        if output_file.exists():

            print(
                "SUCCESS:",
                output_file
            )

            successful.append({
                "waterbody": waterbody,
                "year": year,
                "date": date,
                "file": str(output_file)
            })

        else:

            print(
                "WARNING: Output file was not found."
            )

            failed.append({
                "waterbody": waterbody,
                "year": year,
                "date": date,
                "reason": "output_file_missing"
            })


    except Exception as e:

        print(
            "DOWNLOAD FAILED:"
        )

        print(e)

        failed.append({
            "waterbody": waterbody,
            "year": year,
            "date": date,
            "reason": str(e)
        })


# ============================================================
# SAVE REPORTS
# ============================================================

pd.DataFrame(
    successful
).to_csv(
    OUTPUT_DIR
    / "download_success.csv",
    index=False
)

pd.DataFrame(
    failed
).to_csv(
    OUTPUT_DIR
    / "download_failed.csv",
    index=False
)

pd.DataFrame(
    skipped
).to_csv(
    OUTPUT_DIR
    / "download_skipped.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("STEP 61 COMPLETE")
print("=" * 70)

print()

print(
    "Selected:",
    len(df)
)

print(
    "Downloaded successfully:",
    len(successful)
)

print(
    "Skipped:",
    len(skipped)
)

print(
    "Failed:",
    len(failed)
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