import os
import numpy as np
import pandas as pd
import rasterio


print("=" * 70)
print("STEP 112 — ANALYZE GENERIC HAB DETECTION")
print("=" * 70)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

IMAGE_PATH = (
    "data/raw/sentinel2/generic_selected/"
    "2025-12-18_14Channel.tif"
)

WATER_MASK_PATH = (
    "results/water_masks/generic/"
    "Musi_River_water_mask_2025-12-18.tif"
)

PREDICTION_PATH = (
    "results/swin/generic_detection/"
    "Musi_River_2025-12-18_HAB_prediction.tif"
)

OUTPUT_DIR = "results/swin/generic_detection/analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "Musi_River_2025-12-18_analysis.csv"
)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "Musi_River_2025-12-18_report.txt"
)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\nLoading prediction and water mask...")

with rasterio.open(WATER_MASK_PATH) as src:
    water_mask = src.read(1)
    transform = src.transform
    crs = src.crs

with rasterio.open(PREDICTION_PATH) as src:
    prediction = src.read(1)

print(f"Water mask shape : {water_mask.shape}")
print(f"Prediction shape : {prediction.shape}")
print(f"CRS              : {crs}")


# ------------------------------------------------------------
# BASIC COUNTS
# ------------------------------------------------------------

water = water_mask == 1

water_pixels = int(water.sum())

hab = (
    (prediction == 1) &
    water
)

non_hab = (
    (prediction == 0) &
    water
)

hab_pixels = int(hab.sum())
non_hab_pixels = int(non_hab.sum())


# ------------------------------------------------------------
# CHECK FOR INVALID PREDICTIONS
# ------------------------------------------------------------

invalid_prediction = (
    water &
    ~np.isin(prediction, [0, 1])
)

invalid_pixels = int(invalid_prediction.sum())


# ------------------------------------------------------------
# COVERAGE
# ------------------------------------------------------------

if water_pixels > 0:

    hab_percentage = (
        hab_pixels / water_pixels
    ) * 100

    non_hab_percentage = (
        non_hab_pixels / water_pixels
    ) * 100

else:

    hab_percentage = 0.0
    non_hab_percentage = 0.0


# ------------------------------------------------------------
# PIXEL AREA
# ------------------------------------------------------------

if crs is not None and crs.is_projected:

    pixel_width = abs(transform.a)
    pixel_height = abs(transform.e)

    pixel_area_m2 = pixel_width * pixel_height

else:

    pixel_area_m2 = 0.0


water_area_ha = (
    water_pixels * pixel_area_m2 / 10000
)

hab_area_ha = (
    hab_pixels * pixel_area_m2 / 10000
)

non_hab_area_ha = (
    non_hab_pixels * pixel_area_m2 / 10000
)


# ------------------------------------------------------------
# PREDICTION DISTRIBUTION
# ------------------------------------------------------------

hab_outside_water = int(
    ((prediction == 1) & (~water)).sum()
)

non_hab_outside_water = int(
    ((prediction == 0) & (~water)).sum()
)


# ------------------------------------------------------------
# SUMMARY DATAFRAME
# ------------------------------------------------------------

summary = pd.DataFrame([{

    "waterbody": "Musi River",

    "date": "2025-12-18",

    "water_pixels": water_pixels,

    "hab_pixels": hab_pixels,

    "non_hab_pixels": non_hab_pixels,

    "invalid_prediction_pixels": invalid_pixels,

    "hab_percentage": hab_percentage,

    "non_hab_percentage": non_hab_percentage,

    "water_area_ha": water_area_ha,

    "hab_area_ha": hab_area_ha,

    "non_hab_area_ha": non_hab_area_ha,

    "hab_pixels_outside_water": hab_outside_water,

    "non_hab_pixels_outside_water": non_hab_outside_water,

    "pixel_area_m2": pixel_area_m2

}])


# ------------------------------------------------------------
# SAVE CSV
# ------------------------------------------------------------

summary.to_csv(
    SUMMARY_PATH,
    index=False
)


# ------------------------------------------------------------
# TEXT REPORT
# ------------------------------------------------------------

report = f"""
GENERIC HAB DETECTION ANALYSIS
==============================

Waterbody
---------
Musi River

Date
----
2025-12-18

Input
-----
Sentinel-2 14-channel image

Model
-----
Multi-waterbody Swin Transformer

Spatial Information
-------------------
Image CRS: {crs}
Pixel area: {pixel_area_m2:.4f} m²

Waterbody Statistics
--------------------
Water pixels       : {water_pixels:,}
Water area         : {water_area_ha:.4f} ha

HAB Prediction
--------------
HAB pixels         : {hab_pixels:,}
HAB area           : {hab_area_ha:.4f} ha
HAB coverage       : {hab_percentage:.2f}%

Non-HAB Prediction
------------------
Non-HAB pixels     : {non_hab_pixels:,}
Non-HAB area       : {non_hab_area_ha:.4f} ha
Non-HAB coverage   : {non_hab_percentage:.2f}%

Validation Checks
-----------------
Invalid prediction pixels : {invalid_pixels:,}
HAB pixels outside water  : {hab_outside_water:,}
Non-HAB pixels outside water : {non_hab_outside_water:,}

Coverage Check
--------------
HAB + Non-HAB coverage:
{hab_percentage + non_hab_percentage:.2f}%

Interpretation
--------------
The model prediction is restricted to the selected waterbody
using the water mask.

The HAB percentage represents the model's predicted HAB coverage
for this input image. It is not a field-validated measurement of
actual HAB concentration.
"""


with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(report)


# ------------------------------------------------------------
# PRINT RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 112 COMPLETE")
print("=" * 70)

print("\nWATERBODY")
print("---------")
print("Musi River")

print("\nWATER")
print("-----")
print(f"Water pixels : {water_pixels:,}")
print(f"Water area   : {water_area_ha:.4f} ha")

print("\nHAB PREDICTION")
print("--------------")
print(f"HAB pixels   : {hab_pixels:,}")
print(f"HAB area     : {hab_area_ha:.4f} ha")
print(f"HAB coverage : {hab_percentage:.2f}%")

print("\nNON-HAB")
print("------")
print(f"Non-HAB pixels   : {non_hab_pixels:,}")
print(f"Non-HAB area     : {non_hab_area_ha:.4f} ha")
print(f"Non-HAB coverage : {non_hab_percentage:.2f}%")

print("\nVALIDATION")
print("----------")
print(f"Invalid pixels        : {invalid_pixels:,}")
print(f"HAB outside water     : {hab_outside_water:,}")
print(f"Non-HAB outside water : {non_hab_outside_water:,}")

print("\nFiles saved:")
print(os.path.abspath(SUMMARY_PATH))
print(os.path.abspath(REPORT_PATH))

print("=" * 70)