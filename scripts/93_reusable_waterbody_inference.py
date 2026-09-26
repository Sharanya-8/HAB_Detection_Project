"""
STEP 93
Reusable waterbody inference test.

The inference engine remains generic.
This script allows us to change the input
Sentinel-2 image and aligned water mask
without changing the inference engine.
"""

from pathlib import Path
import importlib.util


# ================================================================
# PROJECT ROOT
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ================================================================
# STEP 89 GENERIC INFERENCE ENGINE
# ================================================================

ENGINE_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "89_generic_waterbody_inference.py"
)

spec = importlib.util.spec_from_file_location(
    "generic_inference",
    ENGINE_PATH
)

generic_inference = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    generic_inference
)


# ================================================================
# SELECTED WATERBODY
# ================================================================

WATERBODY = "Shamirpet"

# Change ONLY this date when testing another image.
DATE = "2016-01-30"


# ================================================================
# INPUT IMAGE
# ================================================================

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sentinel2"
    / "shamirpet_test"
    / f"{DATE}.tif"
)


# ================================================================
# WATER MASK
# ================================================================

# For now, use the aligned Shamirpet mask we created
# for the first test image.

WATER_MASK_PATH = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "hab_masks"
    / "shamirpet_test"
    / "Shamirpet_water_mask_2016-01-30.tif"
)


# ================================================================
# OUTPUT
# ================================================================

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "swin"
    / "reusable_inference"
    / f"{WATERBODY}_{DATE}_prediction.tif"
)


# ================================================================
# CHECK INPUTS
# ================================================================

print("=" * 70)
print("STEP 93 — REUSABLE WATERBODY INFERENCE")
print("=" * 70)

print()
print("Waterbody :", WATERBODY)
print("Date      :", DATE)

print()
print("Input image:")
print(IMAGE_PATH)

print()
print("Water mask:")
print(WATER_MASK_PATH)

print()
print("Output:")
print(OUTPUT_PATH)


if not IMAGE_PATH.exists():

    raise FileNotFoundError(
        f"\nSentinel-2 image not found:\n{IMAGE_PATH}"
    )


if not WATER_MASK_PATH.exists():

    raise FileNotFoundError(
        f"\nWater mask not found:\n{WATER_MASK_PATH}"
    )


# ================================================================
# RUN INFERENCE
# ================================================================

prediction, statistics = (
    generic_inference.run_waterbody_inference(
        image_path=IMAGE_PATH,
        water_mask_path=WATER_MASK_PATH,
        output_prediction_path=OUTPUT_PATH
    )
)


# ================================================================
# FINAL SUMMARY
# ================================================================

print()
print("=" * 70)
print("STEP 93 RESULT")
print("=" * 70)

print(
    f"Waterbody       : {WATERBODY}"
)

print(
    f"Date            : {DATE}"
)

print(
    f"Water pixels    : "
    f"{statistics['water_pixels']:,}"
)

print(
    f"HAB pixels      : "
    f"{statistics['hab_pixels']:,}"
)

print(
    f"HAB coverage    : "
    f"{statistics['hab_percentage']:.2f}%"
)

print(
    f"HAB area        : "
    f"{statistics['hab_area_ha']:.4f} ha"
)

print()
print("Prediction saved:")
print(OUTPUT_PATH)

print()
print("=" * 70)
print("STEP 93 COMPLETE")
print("=" * 70)