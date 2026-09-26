"""
STEP 95
End-to-end waterbody HAB detection test.

Pipeline:

Waterbody boundary
        ↓
Sentinel-2 image
        ↓
Automatically create aligned water mask
        ↓
14-channel image
        ↓
Swin Transformer
        ↓
HAB prediction
        ↓
HAB coverage + HAB area
"""

from pathlib import Path
import importlib.util


# ================================================================
# PROJECT ROOT
# ================================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)


# ================================================================
# TEST SETTINGS
# ================================================================

WATERBODY = "Shamirpet"

DATE = "2016-01-30"


# ================================================================
# WATERBODY BOUNDARY
# ================================================================

BOUNDARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "hab_masks"
    / "training_waterbodies"
    / "Shamirpet_Lake_OSM_boundary.geojson"
)


# ================================================================
# SENTINEL-2 IMAGE
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
# OUTPUT WATER MASK
# ================================================================

WATER_MASK_PATH = (
    PROJECT_ROOT
    / "results"
    / "water_masks"
    / WATERBODY
    / f"{DATE}_water_mask.tif"
)


# ================================================================
# OUTPUT PREDICTION
# ================================================================

PREDICTION_PATH = (
    PROJECT_ROOT
    / "results"
    / "swin"
    / "end_to_end"
    / f"{WATERBODY}_{DATE}_prediction.tif"
)


# ================================================================
# LOAD STEP 94
# ================================================================

MASK_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "94_create_aligned_water_mask.py"
)


mask_spec = (
    importlib.util.spec_from_file_location(
        "water_mask_generator",
        MASK_SCRIPT
    )
)

water_mask_module = (
    importlib.util.module_from_spec(
        mask_spec
    )
)

mask_spec.loader.exec_module(
    water_mask_module
)


# ================================================================
# LOAD STEP 89
# ================================================================

INFERENCE_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "89_generic_waterbody_inference.py"
)


inference_spec = (
    importlib.util.spec_from_file_location(
        "generic_inference",
        INFERENCE_SCRIPT
    )
)

inference_module = (
    importlib.util.module_from_spec(
        inference_spec
    )
)

inference_spec.loader.exec_module(
    inference_module
)


# ================================================================
# START
# ================================================================

print()
print("=" * 70)
print("STEP 95 — END-TO-END WATERBODY TEST")
print("=" * 70)

print()
print("Waterbody :", WATERBODY)
print("Date      :", DATE)

print()
print("Boundary:")
print(BOUNDARY_PATH)

print()
print("Sentinel-2 image:")
print(IMAGE_PATH)

print()
print("Water mask:")
print(WATER_MASK_PATH)

print()
print("Prediction:")
print(PREDICTION_PATH)


# ================================================================
# CHECK INPUTS
# ================================================================

if not BOUNDARY_PATH.exists():

    raise FileNotFoundError(
        f"\nBoundary not found:\n"
        f"{BOUNDARY_PATH}"
    )


if not IMAGE_PATH.exists():

    raise FileNotFoundError(
        f"\nSentinel-2 image not found:\n"
        f"{IMAGE_PATH}"
    )


# ================================================================
# STEP 1 — CREATE ALIGNED WATER MASK
# ================================================================

print()
print("=" * 70)
print("STEP 1 — CREATE ALIGNED WATER MASK")
print("=" * 70)

water_mask_module.create_aligned_water_mask(
    boundary_path=BOUNDARY_PATH,
    image_path=IMAGE_PATH,
    output_mask_path=WATER_MASK_PATH
)


# ================================================================
# STEP 2 — RUN SWIN INFERENCE
# ================================================================

print()
print("=" * 70)
print("STEP 2 — RUN SWIN HAB INFERENCE")
print("=" * 70)

prediction, statistics = (
    inference_module.run_waterbody_inference(
        image_path=IMAGE_PATH,
        water_mask_path=WATER_MASK_PATH,
        output_prediction_path=PREDICTION_PATH
    )
)


# ================================================================
# FINAL RESULT
# ================================================================

print()
print("=" * 70)
print("STEP 95 FINAL RESULT")
print("=" * 70)

print()
print("Waterbody:")
print(WATERBODY)

print()
print("Date:")
print(DATE)

print()
print(
    "Water pixels:",
    f"{statistics['water_pixels']:,}"
)

print(
    "HAB pixels:",
    f"{statistics['hab_pixels']:,}"
)

print(
    "HAB coverage:",
    f"{statistics['hab_percentage']:.2f}%"
)

print(
    "HAB area:",
    f"{statistics['hab_area_ha']:.4f} ha"
)

print(
    "HAB area:",
    f"{statistics['hab_area_m2']:.2f} m²"
)

print()
print("Prediction saved:")
print(PREDICTION_PATH)

print()
print("=" * 70)
print("STEP 95 COMPLETE")
print("=" * 70)