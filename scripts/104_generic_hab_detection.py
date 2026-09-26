"""
STEP 104 — GENERIC HAB DETECTION

Runs the trained multi-waterbody Swin Transformer on a
14-channel Sentinel-2 image and restricts the prediction
to the selected waterbody.

Input:
    Sentinel-2 14-channel image
    Waterbody mask

Output:
    HAB prediction GeoTIFF
    HAB area
    HAB coverage
    Summary CSV
"""

import os
import importlib.util

import numpy as np
import pandas as pd
import rasterio


# ============================================================
# INPUTS
# ============================================================

IMAGE_PATH = os.path.join(
    "data",
    "raw",
    "sentinel2",
    "historical",
    "Hussain_Sagar_2025_2025-06-21_14Channel.tif"
)

WATER_MASK_PATH = os.path.join(
    "results",
    "water_masks",
    "generic",
    "Hussain_Sagar_Hussain_Sagar_2025_2025-06-21_14Channel_water_mask.tif"
)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "swin",
    "multilocation_corrected_best_swin_hab_model.pth"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIRECTORY = os.path.join(
    "results",
    "swin",
    "generic_detection"
)

os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True
)

PREDICTION_PATH = os.path.join(
    OUTPUT_DIRECTORY,
    "Hussain_Sagar_2025_HAB_prediction.tif"
)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIRECTORY,
    "Hussain_Sagar_2025_HAB_summary.csv"
)


# ============================================================
# CHECK INPUTS
# ============================================================

print("=" * 70)
print("STEP 104 — GENERIC HAB DETECTION")
print("=" * 70)

print()

if not os.path.exists(IMAGE_PATH):

    print("ERROR: Sentinel-2 image not found:")
    print(os.path.abspath(IMAGE_PATH))

    raise SystemExit


if not os.path.exists(WATER_MASK_PATH):

    print("ERROR: Water mask not found:")
    print(os.path.abspath(WATER_MASK_PATH))

    raise SystemExit


if not os.path.exists(MODEL_PATH):

    print("ERROR: Swin model not found:")
    print(os.path.abspath(MODEL_PATH))

    raise SystemExit


# ============================================================
# LOAD GENERIC INFERENCE ENGINE
# ============================================================

print("Loading generic inference engine...")

spec = importlib.util.spec_from_file_location(
    "generic_inference",
    os.path.join(
        "scripts",
        "89_generic_waterbody_inference.py"
    )
)

generic_inference = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    generic_inference
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading trained Swin Transformer...")
print()

model = generic_inference.load_model()


# ============================================================
# RUN INFERENCE
# ============================================================

print("=" * 70)
print("RUNNING HAB DETECTION")
print("=" * 70)

print()

print("Image:")
print(os.path.abspath(IMAGE_PATH))

print()

print("Water mask:")
print(os.path.abspath(WATER_MASK_PATH))

print()


result = generic_inference.run_waterbody_inference(
    image_path=IMAGE_PATH,
    water_mask_path=WATER_MASK_PATH,
    output_prediction_path=PREDICTION_PATH
)


# ============================================================
# READ RESULT
# ============================================================

prediction, _ = result


# ============================================================
# READ WATER MASK DIRECTLY
# ============================================================

print("Reading water mask for final statistics...")

with rasterio.open(WATER_MASK_PATH) as mask_src:

    water_mask = mask_src.read(1)

    mask_transform = mask_src.transform
    mask_crs = mask_src.crs

    mask_height = mask_src.height
    mask_width = mask_src.width


print()

print("Water mask information:")
print(
    f"  Size: {mask_width} x {mask_height}"
)

print(
    f"  CRS:  {mask_crs}"
)

print(
    f"  Values: {np.unique(water_mask)}"
)


# ============================================================
# CHECK DIMENSIONS
# ============================================================

if prediction.shape != water_mask.shape:

    raise ValueError(
        "Prediction and water mask dimensions do not match.\n"
        f"Prediction shape: {prediction.shape}\n"
        f"Water mask shape: {water_mask.shape}"
    )


# ============================================================
# STATISTICS
# ============================================================

# Water mask uses:
# 0 = non-water
# 1 = water

water_pixels = int(
    np.count_nonzero(
        water_mask == 1
    )
)

# Prediction uses:
# 0 = Non-HAB
# 1 = HAB

hab_pixels = int(
    np.count_nonzero(
        prediction == 1
    )
)

hab_coverage = (
    hab_pixels
    / water_pixels
    * 100
    if water_pixels > 0
    else 0
)


# ============================================================
# AREA
# ============================================================

# Read transform and CRS from the Sentinel-2 image.
# This keeps the area calculation aligned with the
# actual image used for inference.

with rasterio.open(IMAGE_PATH) as src:

    image_transform = src.transform
    image_crs = src.crs


pixel_area_m2 = (
    generic_inference.calculate_pixel_area_m2(
        image_transform,
        image_crs
    )
)


hab_area_m2 = (
    hab_pixels
    * pixel_area_m2
)

hab_area_ha = (
    hab_area_m2
    / 10000
)


water_area_ha = (
    water_pixels
    * pixel_area_m2
    / 10000
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("HAB DETECTION RESULTS")
print("=" * 70)

print()

print(
    f"Water pixels       : {water_pixels:,}"
)

print(
    f"HAB pixels         : {hab_pixels:,}"
)

print(
    f"Water area         : {water_area_ha:.4f} ha"
)

print(
    f"HAB area           : {hab_area_ha:.4f} ha"
)

print(
    f"HAB coverage       : {hab_coverage:.2f}%"
)

print()


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        {
            "waterbody": "Hussain Sagar",
            "date": "2025-06-21",
            "water_pixels": water_pixels,
            "hab_pixels": hab_pixels,
            "water_area_ha": water_area_ha,
            "hab_area_ha": hab_area_ha,
            "hab_coverage_percent": hab_coverage
        }
    ]
)


summary.to_csv(
    SUMMARY_PATH,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("=" * 70)
print("STEP 104 COMPLETE")
print("=" * 70)

print()

print("Prediction saved:")
print(
    os.path.abspath(
        PREDICTION_PATH
    )
)

print()

print("Summary saved:")
print(
    os.path.abspath(
        SUMMARY_PATH
    )
)

print()