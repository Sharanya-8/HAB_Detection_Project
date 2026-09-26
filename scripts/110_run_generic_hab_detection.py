import os
import importlib.util
import pandas as pd
import rasterio


print("=" * 70)
print("STEP 110 — GENERIC HAB DETECTION")
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

OUTPUT_DIR = "results/swin/generic_detection"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PREDICTION_PATH = os.path.join(
    OUTPUT_DIR,
    "Musi_River_2025-12-18_HAB_prediction.tif"
)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "Musi_River_2025-12-18_HAB_summary.csv"
)


# ------------------------------------------------------------
# LOAD GENERIC INFERENCE ENGINE
# ------------------------------------------------------------

print("\nLoading generic Swin inference engine...")

spec = importlib.util.spec_from_file_location(
    "generic_inference",
    "scripts/89_generic_waterbody_inference.py"
)

generic_inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generic_inference)


# ------------------------------------------------------------
# RUN MODEL
# ------------------------------------------------------------

print("\nRunning Swin HAB model...")
print("Please wait...")

prediction, model_water_mask = (
    generic_inference.run_waterbody_inference(
        image_path=IMAGE_PATH,
        water_mask_path=WATER_MASK_PATH,
        output_prediction_path=PREDICTION_PATH
    )
)


# ------------------------------------------------------------
# READ WATER MASK
# ------------------------------------------------------------

with rasterio.open(WATER_MASK_PATH) as src:
    water_mask = src.read(1)


# ------------------------------------------------------------
# CALCULATE STATISTICS
# ------------------------------------------------------------

water_pixels = int((water_mask == 1).sum())

hab_pixels = int(
    ((prediction == 1) & (water_mask == 1)).sum()
)

if water_pixels > 0:
    hab_percentage = hab_pixels / water_pixels * 100
else:
    hab_percentage = 0.0


# ------------------------------------------------------------
# AREA CALCULATION
# ------------------------------------------------------------

with rasterio.open(IMAGE_PATH) as src:
    transform = src.transform
    crs = src.crs

pixel_area_m2 = generic_inference.calculate_pixel_area_m2(
    transform,
    crs
)

water_area_ha = (
    water_pixels * pixel_area_m2 / 10000
)

hab_area_ha = (
    hab_pixels * pixel_area_m2 / 10000
)


# ------------------------------------------------------------
# SAVE SUMMARY
# ------------------------------------------------------------

summary = pd.DataFrame([{
    "waterbody": "Musi River",
    "date": "2025-12-18",
    "water_pixels": water_pixels,
    "hab_pixels": hab_pixels,
    "water_area_ha": water_area_ha,
    "hab_area_ha": hab_area_ha,
    "hab_coverage_percent": hab_percentage
}])

summary.to_csv(
    SUMMARY_PATH,
    index=False
)


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 110 COMPLETE")
print("=" * 70)

print(f"\nWaterbody        : Musi River")
print(f"Date             : 2025-12-18")
print(f"Water pixels     : {water_pixels:,}")
print(f"HAB pixels       : {hab_pixels:,}")
print(f"Water area       : {water_area_ha:.4f} ha")
print(f"HAB area         : {hab_area_ha:.4f} ha")
print(f"HAB coverage     : {hab_percentage:.2f}%")

print("\nPrediction saved to:")
print(os.path.abspath(PREDICTION_PATH))

print("\nSummary saved to:")
print(os.path.abspath(SUMMARY_PATH))

print("=" * 70)