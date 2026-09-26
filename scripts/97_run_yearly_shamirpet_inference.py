"""
STEP 97
Run the complete HAB detection pipeline for
one representative Sentinel-2 image per year.

Period:
2016-2025

For every year:

Sentinel-2 image
        ↓
Aligned water mask
        ↓
Swin inference
        ↓
HAB coverage
        ↓
HAB area
"""

from pathlib import Path
import importlib.util
import pandas as pd


# ================================================================
# PROJECT ROOT
# ================================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)


# ================================================================
# INPUT CSV
# ================================================================

DATES_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "shamirpet_yearly_representative_dates.csv"
)


# ================================================================
# SHAMIRPET BOUNDARY
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
# OUTPUT DIRECTORIES
# ================================================================

MASK_DIR = (
    PROJECT_ROOT
    / "results"
    / "water_masks"
    / "Shamirpet_yearly"
)

PREDICTION_DIR = (
    PROJECT_ROOT
    / "results"
    / "swin"
    / "shamirpet_yearly"
)

RESULTS_CSV = (
    PROJECT_ROOT
    / "results"
    / "swin"
    / "shamirpet_yearly_results.csv"
)


# ================================================================
# LOAD WATER MASK GENERATOR
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

mask_module = (
    importlib.util.module_from_spec(
        mask_spec
    )
)

mask_spec.loader.exec_module(
    mask_module
)


# ================================================================
# LOAD GENERIC INFERENCE ENGINE
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
# CHECK INPUTS
# ================================================================

if not DATES_CSV.exists():

    raise FileNotFoundError(
        f"Representative dates CSV not found:\n"
        f"{DATES_CSV}"
    )


if not BOUNDARY_PATH.exists():

    raise FileNotFoundError(
        f"Shamirpet boundary not found:\n"
        f"{BOUNDARY_PATH}"
    )


# ================================================================
# LOAD DATES
# ================================================================

dates_df = pd.read_csv(
    DATES_CSV
)

print("=" * 70)
print("STEP 97 — SHAMIRPET YEARLY HAB INFERENCE")
print("=" * 70)

print()
print(
    "Years to process:",
    len(dates_df)
)

print()


# ================================================================
# CREATE OUTPUT DIRECTORIES
# ================================================================

MASK_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# RESULTS
# ================================================================

results = []


# ================================================================
# PROCESS EACH YEAR
# ================================================================

for index, row in dates_df.iterrows():

    year = int(
        row["year"]
    )

    date = str(
        row["selected_date"]
    )

    image_path = Path(
        row["image_path"]
    )

    # ------------------------------------------------------------
    # Make relative paths absolute if necessary
    # ------------------------------------------------------------

    if not image_path.is_absolute():

        image_path = (
            PROJECT_ROOT /
            image_path
        )

    water_mask_path = (
        MASK_DIR /
        f"{date}_water_mask.tif"
    )

    prediction_path = (
        PREDICTION_DIR /
        f"Shamirpet_{date}_prediction.tif"
    )

    print()
    print()
    print("#" * 70)
    print(
        f"YEAR {year} "
        f"({index + 1}/{len(dates_df)})"
    )
    print(
        f"DATE: {date}"
    )
    print("#" * 70)

    # ------------------------------------------------------------
    # Check image
    # ------------------------------------------------------------

    if not image_path.exists():

        print(
            f"WARNING: Image not found:\n"
            f"{image_path}"
        )

        results.append(
            {
                "year": year,
                "date": date,
                "status": "image_missing"
            }
        )

        continue

    # ------------------------------------------------------------
    # STEP 1 — WATER MASK
    # ------------------------------------------------------------

    print()
    print(
        "Creating aligned water mask..."
    )

    mask_module.create_aligned_water_mask(
        boundary_path=BOUNDARY_PATH,
        image_path=image_path,
        output_mask_path=water_mask_path
    )

    # ------------------------------------------------------------
    # STEP 2 — SWIN INFERENCE
    # ------------------------------------------------------------

    print()
    print(
        "Running Swin inference..."
    )

    prediction, statistics = (
        inference_module.run_waterbody_inference(
            image_path=image_path,
            water_mask_path=water_mask_path,
            output_prediction_path=prediction_path
        )
    )

    # ------------------------------------------------------------
    # SAVE RESULT
    # ------------------------------------------------------------

    result = {
        "year": year,
        "date": date,
        "status": "success",
        "water_pixels": (
            statistics["water_pixels"]
        ),
        "hab_pixels": (
            statistics["hab_pixels"]
        ),
        "hab_percentage": (
            statistics["hab_percentage"]
        ),
        "hab_area_ha": (
            statistics["hab_area_ha"]
        ),
        "hab_area_m2": (
            statistics["hab_area_m2"]
        ),
        "image_path": str(
            image_path
        ),
        "water_mask_path": str(
            water_mask_path
        ),
        "prediction_path": str(
            prediction_path
        )
    }

    results.append(
        result
    )

    # ------------------------------------------------------------
    # YEAR RESULT
    # ------------------------------------------------------------

    print()
    print(
        f"YEAR {year} RESULT"
    )

    print(
        f"HAB coverage: "
        f"{statistics['hab_percentage']:.2f}%"
    )

    print(
        f"HAB area: "
        f"{statistics['hab_area_ha']:.4f} ha"
    )


# ================================================================
# SAVE FINAL CSV
# ================================================================

results_df = pd.DataFrame(
    results
)

RESULTS_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    RESULTS_CSV,
    index=False
)


# ================================================================
# DISPLAY SUMMARY
# ================================================================

print()
print()
print("=" * 70)
print("STEP 97 — YEARLY SUMMARY")
print("=" * 70)

successful = results_df[
    results_df["status"] == "success"
]

print()

for _, row in successful.iterrows():

    print(
        f"{int(row['year'])} | "
        f"{row['date']} | "
        f"HAB: "
        f"{row['hab_percentage']:.2f}% | "
        f"Area: "
        f"{row['hab_area_ha']:.4f} ha"
    )


print()
print(
    "Successful years:",
    len(successful)
)

print(
    "Total years:",
    len(results_df)
)

print()
print(
    "Results CSV:"
)

print(
    RESULTS_CSV
)

print()
print("=" * 70)
print("STEP 97 COMPLETE")
print("=" * 70)