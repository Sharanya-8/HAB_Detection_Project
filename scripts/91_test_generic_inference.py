import importlib.util
from pathlib import Path


# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sentinel2"
    / "shamirpet_test"
    / "2016-01-30.tif"
)

WATER_MASK_PATH = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "hab_masks"
    / "shamirpet_test"
    / "Shamirpet_water_mask_2016-01-30.tif"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "swin"
    / "generic_inference_test"
    / "2016-01-30_shamirpet_prediction.tif"
)


# ------------------------------------------------------------
# LOAD STEP 89 MODULE
# ------------------------------------------------------------

SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "89_generic_waterbody_inference.py"
)

spec = importlib.util.spec_from_file_location(
    "generic_inference",
    SCRIPT_PATH
)

generic_inference = importlib.util.module_from_spec(spec)

spec.loader.exec_module(generic_inference)


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------

print("=" * 70)
print("STEP 91 — TEST GENERIC WATERBODY INFERENCE")
print("=" * 70)

print("\nInput image:")
print(IMAGE_PATH)

print("\nWater mask:")
print(WATER_MASK_PATH)

print("\nOutput prediction:")
print(OUTPUT_PATH)


prediction, statistics = (
    generic_inference.run_waterbody_inference(
        image_path=IMAGE_PATH,
        water_mask_path=WATER_MASK_PATH,
        output_prediction_path=OUTPUT_PATH
    )
)


# ------------------------------------------------------------
# FINAL RESULT
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("STEP 91 RESULT")
print("=" * 70)

print(
    f"Water pixels : "
    f"{statistics['water_pixels']:,}"
)

print(
    f"HAB pixels   : "
    f"{statistics['hab_pixels']:,}"
)

print(
    f"HAB coverage : "
    f"{statistics['hab_percentage']:.2f}%"
)

print(
    f"HAB area     : "
    f"{statistics['hab_area_ha']:.4f} ha"
)

print("\nPrediction saved to:")
print(OUTPUT_PATH)

print("\nSTEP 91 COMPLETE")
print("=" * 70)