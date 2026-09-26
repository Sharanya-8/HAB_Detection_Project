import os
import importlib.util
import numpy as np
import pandas as pd
import rasterio


print("=" * 70)
print("STEP 116 — OSMAN SAGAR SWIN HAB DETECTION")
print("=" * 70)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

IMAGE_PATH = (
    "data/raw/sentinel2/second_waterbody/"
    "Osman_Sagar_2025_14Channel.tif"
)

WATER_MASK_PATH = (
    "results/water_masks/second_waterbody/"
    "Osman_Sagar_water_mask_2025-01-02.tif"
)

OUTPUT_DIR = "results/swin/generic_detection"

OUTPUT_PREDICTION_PATH = os.path.join(
    OUTPUT_DIR,
    "Osman_Sagar_2025-01-02_HAB_prediction.tif"
)

OUTPUT_SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "Osman_Sagar_2025-01-02_HAB_summary.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# LOAD STEP 89 GENERIC INFERENCE ENGINE
# ------------------------------------------------------------

print("\nLoading generic Swin inference engine...")

STEP89_PATH = "scripts/89_generic_waterbody_inference.py"

spec = importlib.util.spec_from_file_location(
    "generic_inference",
    STEP89_PATH
)

generic_inference = importlib.util.module_from_spec(spec)

spec.loader.exec_module(generic_inference)


# ------------------------------------------------------------
# RUN SWIN INFERENCE
# ------------------------------------------------------------

print("\nRunning Swin Transformer inference...")

prediction, water_mask_from_model = (
    generic_inference.run_waterbody_inference(
        image_path=IMAGE_PATH,
        water_mask_path=WATER_MASK_PATH,
        output_prediction_path=OUTPUT_PREDICTION_PATH
    )
)


# ------------------------------------------------------------
# READ WATER MASK
# ------------------------------------------------------------

print("\nReading water mask...")

with rasterio.open(WATER_MASK_PATH) as src:
    water_mask = src.read(1)
    transform = src.transform
    crs = src.crs


# ------------------------------------------------------------
# CALCULATE PIXEL AREA
# ------------------------------------------------------------

pixel_area_m2 = generic_inference.calculate_pixel_area_m2(
    transform,
    crs
)

pixel_area_ha = pixel_area_m2 / 10000.0


# ------------------------------------------------------------
# CALCULATE HAB STATISTICS
# ------------------------------------------------------------

water_pixels = water_mask == 1

hab_pixels = (
    (prediction == 1) &
    water_pixels
)

non_hab_pixels = (
    (prediction == 0) &
    water_pixels
)

water_pixel_count = int(water_pixels.sum())
hab_pixel_count = int(hab_pixels.sum())
non_hab_pixel_count = int(non_hab_pixels.sum())

water_area_ha = water_pixel_count * pixel_area_ha
hab_area_ha = hab_pixel_count * pixel_area_ha
non_hab_area_ha = non_hab_pixel_count * pixel_area_ha

hab_coverage = (
    hab_pixel_count / water_pixel_count * 100
    if water_pixel_count > 0
    else 0
)

non_hab_coverage = (
    non_hab_pixel_count / water_pixel_count * 100
    if water_pixel_count > 0
    else 0
)


# ------------------------------------------------------------
# SAVE SUMMARY
# ------------------------------------------------------------

summary = pd.DataFrame([{
    "waterbody": "Osman Sagar",
    "date": "2025-01-02",
    "water_pixels": water_pixel_count,
    "hab_pixels": hab_pixel_count,
    "non_hab_pixels": non_hab_pixel_count,
    "water_area_ha": water_area_ha,
    "hab_area_ha": hab_area_ha,
    "non_hab_area_ha": non_hab_area_ha,
    "hab_coverage_percent": hab_coverage,
    "non_hab_coverage_percent": non_hab_coverage
}])

summary.to_csv(
    OUTPUT_SUMMARY_PATH,
    index=False
)


# ------------------------------------------------------------
# FINAL OUTPUT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 116 COMPLETE")
print("=" * 70)

print(f"\nWaterbody          : Osman Sagar")
print(f"Date               : 2025-01-02")
print(f"Water pixels       : {water_pixel_count:,}")
print(f"HAB pixels         : {hab_pixel_count:,}")
print(f"Non-HAB pixels     : {non_hab_pixel_count:,}")
print(f"Water area         : {water_area_ha:.4f} ha")
print(f"Predicted HAB area : {hab_area_ha:.4f} ha")
print(f"HAB coverage       : {hab_coverage:.2f}%")
print(f"Non-HAB coverage   : {non_hab_coverage:.2f}%")

print("\nPrediction saved to:")
print(os.path.abspath(OUTPUT_PREDICTION_PATH))

print("\nSummary saved to:")
print(os.path.abspath(OUTPUT_SUMMARY_PATH))

print("=" * 70)