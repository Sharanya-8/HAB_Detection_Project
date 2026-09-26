"""
Step 43: Create yearly HAB pseudo-labels for historical images.

For each yearly 14-channel Sentinel-2 image:
    1. Read the 14 bands.
    2. Apply the Hussain Sagar waterbody boundary.
    3. Use NDCI + FAI to create conservative HAB pseudo-labels.
    4. Calculate HAB / non-HAB pixels.
    5. Calculate HAB percentage.
    6. Calculate HAB area.
    7. Save a yearly HAB mask.
    8. Save a CSV containing all yearly statistics.

IMPORTANT:
These are spectral pseudo-labels, NOT field-verified ground truth.
"""


from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sentinel2"
    / "historical"
)

BOUNDARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "hab_masks"
    / "Hussain_Sagar_boundary.geojson"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "hab_masks"
    / "historical"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "historical"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# BAND INFORMATION
# =========================================================

BAND_NAMES = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
    "NDWI",
    "MNDWI",
    "NDCI",
    "FAI",
]


# Band positions in the 14-channel TIFF
NDCI_BAND = 13
FAI_BAND = 14


# =========================================================
# HAB THRESHOLDS
# =========================================================
#
# These are the conservative spectral thresholds used
# previously in the project.
#
# HAB candidate:
#       NDCI > 0.30
#       AND
#       FAI > 0.05
#
# These are pseudo-label rules, not ground truth.
# =========================================================

NDCI_THRESHOLD = 0.30
FAI_THRESHOLD = 0.05


# =========================================================
# AREA CALCULATION
# =========================================================
#
# The downloaded images are in EPSG:4326.
# Therefore, pixel size is expressed in degrees.
#
# Instead of assuming a fixed 10m x 10m pixel size,
# calculate pixel area from the raster's geographic
# transform at the Hussain Sagar latitude.
#
# This gives an approximate area in square metres.
# =========================================================


def calculate_pixel_area_m2(transform, latitude):

    # Size of one pixel in degrees
    pixel_width_deg = abs(transform.a)
    pixel_height_deg = abs(transform.e)

    # Approximate metres per degree at this latitude
    lat_m = 111320.0

    lon_m = (
        111320.0
        * np.cos(
            np.deg2rad(latitude)
        )
    )

    pixel_width_m = (
        pixel_width_deg * lon_m
    )

    pixel_height_m = (
        pixel_height_deg * lat_m
    )

    pixel_area_m2 = (
        pixel_width_m
        * pixel_height_m
    )

    return pixel_area_m2


# =========================================================
# LOAD WATERBODY BOUNDARY
# =========================================================

print("=" * 80)
print("STEP 43 - CREATE HISTORICAL HAB PSEUDO-LABELS")
print("=" * 80)

print("\nInput directory:")
print(INPUT_DIR)

print("\nBoundary:")
print(BOUNDARY_FILE)


if not BOUNDARY_FILE.exists():

    raise FileNotFoundError(
        f"\nWaterbody boundary not found:\n"
        f"{BOUNDARY_FILE}"
    )


with open(
    BOUNDARY_FILE,
    "r",
    encoding="utf-8"
) as file:

    boundary_geojson = json.load(file)


# Extract all geometries
boundary_geometries = []

for feature in boundary_geojson["features"]:

    geometry = feature["geometry"]

    boundary_geometries.append(
        geometry
    )


print(
    f"\nLoaded {len(boundary_geometries)} "
    "waterbody boundary geometry/ies."
)


# =========================================================
# FIND HISTORICAL TIFF FILES
# =========================================================

files = sorted(
    INPUT_DIR.glob(
        "Hussain_Sagar_*_14Channel.tif"
    )
)


print(
    f"\nFound {len(files)} historical images."
)


if len(files) == 0:

    raise FileNotFoundError(
        "\nNo historical 14-channel TIFF files found."
    )


# =========================================================
# RESULTS
# =========================================================

results = []


# =========================================================
# PROCESS EACH YEAR
# =========================================================

for file_path in files:

    print("\n")
    print("=" * 80)
    print(f"PROCESSING: {file_path.name}")
    print("=" * 80)

    # -----------------------------------------------------
    # Extract year
    # -----------------------------------------------------

    year_match = re.search(
        r"Hussain_Sagar_(\d{4})_",
        file_path.name
    )

    if year_match is None:

        print(
            "Could not determine year. Skipping."
        )

        continue

    year = int(
        year_match.group(1)
    )

    # -----------------------------------------------------
    # Open raster
    # -----------------------------------------------------

    with rasterio.open(file_path) as src:

        if src.count != 14:

            print(
                f"ERROR: Expected 14 bands, "
                f"found {src.count}. Skipping."
            )

            continue

        data = src.read()

        transform = src.transform
        width = src.width
        height = src.height
        crs = src.crs

        # -------------------------------------------------
        # Approximate centre latitude
        # -------------------------------------------------

        centre_y = (
            transform.f
            + (
                height / 2
                * transform.e
            )
        )

        # -------------------------------------------------
        # Create waterbody mask
        # -------------------------------------------------
        #
        # geometry_mask returns:
        # True  = outside geometry
        # False = inside geometry
        #
        # We invert it so:
        # True = waterbody
        # False = outside
        # -------------------------------------------------

        water_mask = geometry_mask(
            boundary_geometries,
            transform=transform,
            invert=True,
            out_shape=(height, width),
            all_touched=True
        )

        # -------------------------------------------------
        # Get NDCI and FAI
        # -------------------------------------------------

        ndci = data[NDCI_BAND - 1].astype(
            np.float32
        )

        fai = data[FAI_BAND - 1].astype(
            np.float32
        )

        # -------------------------------------------------
        # Valid pixels
        # -------------------------------------------------

        valid_pixels = (
            np.isfinite(ndci)
            &
            np.isfinite(fai)
        )

        # -------------------------------------------------
        # HAB pseudo-label
        # -------------------------------------------------
        #
        # 0 = outside waterbody
        # 1 = non-HAB
        # 2 = HAB
        # -------------------------------------------------

        hab_condition = (
            (ndci > NDCI_THRESHOLD)
            &
            (fai > FAI_THRESHOLD)
            &
            valid_pixels
            &
            water_mask
        )

        # -------------------------------------------------
        # Create classification mask
        # -------------------------------------------------

        hab_mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        # Inside waterbody + valid = non-HAB initially
        hab_mask[
            valid_pixels & water_mask
        ] = 1

        # HAB
        hab_mask[
            hab_condition
        ] = 2

        # -------------------------------------------------
        # Pixel statistics
        # -------------------------------------------------

        water_pixels = int(
            np.sum(
                water_mask
                &
                valid_pixels
            )
        )

        hab_pixels = int(
            np.sum(
                hab_condition
            )
        )

        non_hab_pixels = (
            water_pixels
            - hab_pixels
        )

        if water_pixels > 0:

            hab_percentage = (
                hab_pixels
                / water_pixels
                * 100.0
            )

        else:

            hab_percentage = 0.0

        # -------------------------------------------------
        # Pixel area
        # -------------------------------------------------

        pixel_area_m2 = (
            calculate_pixel_area_m2(
                transform,
                centre_y
            )
        )

        hab_area_m2 = (
            hab_pixels
            * pixel_area_m2
        )

        hab_area_hectares = (
            hab_area_m2
            / 10000.0
        )

        water_area_hectares = (
            water_pixels
            * pixel_area_m2
            / 10000.0
        )

        # -------------------------------------------------
        # Save HAB mask
        # -------------------------------------------------

        output_file = (
            OUTPUT_DIR
            / f"Hussain_Sagar_{year}_HAB_mask.tif"
        )

        profile = src.profile.copy()

        profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress="lzw",
            nodata=0
        )

        with rasterio.open(
            output_file,
            "w",
            **profile
        ) as dst:

            dst.write(
                hab_mask,
                1
            )

        # -------------------------------------------------
        # Print results
        # -------------------------------------------------

        print(
            f"\nYear:                 {year}"
        )

        print(
            f"Image size:           "
            f"{width} x {height}"
        )

        print(
            f"Waterbody pixels:     "
            f"{water_pixels:,}"
        )

        print(
            f"Non-HAB pixels:       "
            f"{non_hab_pixels:,}"
        )

        print(
            f"HAB pixels:           "
            f"{hab_pixels:,}"
        )

        print(
            f"HAB percentage:       "
            f"{hab_percentage:.2f}%"
        )

        print(
            f"Estimated water area: "
            f"{water_area_hectares:.4f} ha"
        )

        print(
            f"Estimated HAB area:   "
            f"{hab_area_hectares:.4f} ha"
        )

        print(
            f"Saved mask:           "
            f"{output_file.name}"
        )

        # -------------------------------------------------
        # Store result
        # -------------------------------------------------

        results.append(
            {
                "year": year,
                "image": file_path.name,
                "water_pixels": water_pixels,
                "non_hab_pixels": non_hab_pixels,
                "hab_pixels": hab_pixels,
                "hab_percentage": hab_percentage,
                "water_area_hectares": water_area_hectares,
                "hab_area_hectares": hab_area_hectares,
                "ndci_threshold": NDCI_THRESHOLD,
                "fai_threshold": FAI_THRESHOLD,
            }
        )


# =========================================================
# SAVE CSV
# =========================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "year"
)


csv_file = (
    RESULTS_DIR
    / "historical_hab_statistics.csv"
)


results_df.to_csv(
    csv_file,
    index=False
)


# =========================================================
# FINAL TABLE
# =========================================================

print("\n\n")
print("=" * 100)
print("HISTORICAL HAB RESULTS")
print("=" * 100)

print(
    f"{'Year':<8}"
    f"{'Water Pixels':<16}"
    f"{'HAB Pixels':<14}"
    f"{'HAB %':<12}"
    f"{'HAB Area (ha)':<16}"
)

print("-" * 100)


for _, row in results_df.iterrows():

    print(
        f"{int(row['year']):<8}"
        f"{int(row['water_pixels']):<16}"
        f"{int(row['hab_pixels']):<14}"
        f"{row['hab_percentage']:<12.2f}"
        f"{row['hab_area_hectares']:<16.4f}"
    )


# =========================================================
# FINAL
# =========================================================

print("\n")
print("=" * 80)
print("STEP 43 COMPLETED")
print("=" * 80)

print(
    f"\nHAB masks saved to:"
)
print(OUTPUT_DIR)

print(
    f"\nStatistics saved to:"
)
print(csv_file)

print(
    "\nIMPORTANT:"
)
print(
    "These HAB labels are spectral pseudo-labels "
    "based on NDCI > 0.30 AND FAI > 0.05."
)
print(
    "They are NOT field-verified ground truth."
)