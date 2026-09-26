import os
import rasterio
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


print("=" * 70)
print("STEP 117 — OSMAN SAGAR HAB ANALYSIS + VISUALIZATION")
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

PREDICTION_PATH = (
    "results/swin/generic_detection/"
    "Osman_Sagar_2025-01-02_HAB_prediction.tif"
)

OUTPUT_DIR = (
    "results/swin/generic_detection/"
    "analysis"
)

VISUALIZATION_DIR = (
    "results/swin/generic_detection/"
    "visualization"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)


REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "Osman_Sagar_2025-01-02_analysis.csv"
)

TEXT_REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "Osman_Sagar_2025-01-02_report.txt"
)

VISUALIZATION_PATH = os.path.join(
    VISUALIZATION_DIR,
    "Osman_Sagar_2025-01-02_HAB_visualization.png"
)


# ------------------------------------------------------------
# READ DATA
# ------------------------------------------------------------

print("\nReading water mask...")

with rasterio.open(WATER_MASK_PATH) as src:
    water_mask = src.read(1)
    transform = src.transform
    crs = src.crs


print("Reading Swin prediction...")

with rasterio.open(PREDICTION_PATH) as src:
    prediction = src.read(1)


print("Reading Sentinel-2 image...")

with rasterio.open(IMAGE_PATH) as src:
    image = src.read()


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

if water_mask.shape != prediction.shape:
    raise ValueError(
        f"Water mask shape {water_mask.shape} "
        f"does not match prediction shape {prediction.shape}"
    )


# ------------------------------------------------------------
# MASK DEFINITIONS
# ------------------------------------------------------------

water = water_mask == 1

hab = (prediction == 1) & water
non_hab = (prediction == 0) & water

hab_outside_water = (prediction == 1) & (~water)
non_hab_outside_water = (prediction == 0) & (~water)


# ------------------------------------------------------------
# PIXEL COUNTS
# ------------------------------------------------------------

water_pixels = int(water.sum())
hab_pixels = int(hab.sum())
non_hab_pixels = int(non_hab.sum())

hab_outside_water_pixels = int(
    hab_outside_water.sum()
)

non_hab_outside_water_pixels = int(
    non_hab_outside_water.sum()
)


# ------------------------------------------------------------
# AREA
# EPSG:32644 = UTM, so 10 m pixels = 100 m²
# ------------------------------------------------------------

pixel_area_m2 = abs(
    transform.a * transform.e
)

pixel_area_ha = pixel_area_m2 / 10000.0

water_area_ha = (
    water_pixels * pixel_area_ha
)

hab_area_ha = (
    hab_pixels * pixel_area_ha
)

non_hab_area_ha = (
    non_hab_pixels * pixel_area_ha
)


# ------------------------------------------------------------
# PERCENTAGES
# ------------------------------------------------------------

hab_percentage = (
    hab_pixels / water_pixels * 100
    if water_pixels > 0
    else 0
)

non_hab_percentage = (
    non_hab_pixels / water_pixels * 100
    if water_pixels > 0
    else 0
)


# ------------------------------------------------------------
# PRINT RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("OSMAN SAGAR HAB ANALYSIS")
print("=" * 70)

print(f"Water pixels              : {water_pixels:,}")
print(f"HAB pixels                : {hab_pixels:,}")
print(f"Non-HAB pixels            : {non_hab_pixels:,}")

print(f"\nWater area                : {water_area_ha:.4f} ha")
print(f"Predicted HAB area        : {hab_area_ha:.4f} ha")
print(f"Predicted Non-HAB area    : {non_hab_area_ha:.4f} ha")

print(f"\nPredicted HAB coverage    : {hab_percentage:.2f}%")
print(f"Predicted Non-HAB         : {non_hab_percentage:.2f}%")

print(
    f"\nHAB outside water         : "
    f"{hab_outside_water_pixels:,}"
)

print(
    f"Non-HAB outside water     : "
    f"{non_hab_outside_water_pixels:,}"
)


# ------------------------------------------------------------
# SAVE CSV
# ------------------------------------------------------------

analysis = pd.DataFrame([{
    "waterbody": "Osman Sagar",
    "date": "2025-01-02",
    "water_pixels": water_pixels,
    "hab_pixels": hab_pixels,
    "non_hab_pixels": non_hab_pixels,
    "water_area_ha": water_area_ha,
    "hab_area_ha": hab_area_ha,
    "non_hab_area_ha": non_hab_area_ha,
    "hab_coverage_percent": hab_percentage,
    "non_hab_coverage_percent": non_hab_percentage,
    "hab_outside_water_pixels": hab_outside_water_pixels,
    "non_hab_outside_water_pixels": non_hab_outside_water_pixels
}])

analysis.to_csv(
    REPORT_PATH,
    index=False
)


# ------------------------------------------------------------
# TEXT REPORT
# ------------------------------------------------------------

with open(TEXT_REPORT_PATH, "w", encoding="utf-8") as f:

    f.write("OSMAN SAGAR HAB DETECTION ANALYSIS\n")
    f.write("=" * 60 + "\n\n")

    f.write("Date: 2025-01-02\n")
    f.write("Model: Multi-waterbody Swin Transformer\n\n")

    f.write(f"Water pixels: {water_pixels:,}\n")
    f.write(f"HAB pixels: {hab_pixels:,}\n")
    f.write(f"Non-HAB pixels: {non_hab_pixels:,}\n\n")

    f.write(f"Water area: {water_area_ha:.4f} ha\n")
    f.write(
        f"Predicted HAB area: "
        f"{hab_area_ha:.4f} ha\n"
    )

    f.write(
        f"Predicted HAB coverage: "
        f"{hab_percentage:.2f}%\n"
    )

    f.write(
        "\nNote: HAB values are model predictions "
        "and are not field-validated ground truth.\n"
    )


# ------------------------------------------------------------
# VISUALIZATION
# ------------------------------------------------------------

print("\nCreating visualization...")

# RGB
rgb = np.stack([
    image[2],  # B4
    image[1],  # B3
    image[0]   # B2
], axis=0)

rgb = np.nan_to_num(rgb)

# Normalize RGB for display
rgb_min = np.percentile(rgb, 2)
rgb_max = np.percentile(rgb, 98)

rgb_display = (
    (rgb - rgb_min) /
    (rgb_max - rgb_min + 1e-8)
)

rgb_display = np.clip(
    rgb_display,
    0,
    1
)

rgb_display = np.transpose(
    rgb_display,
    (1, 2, 0)
)


# HAB overlay
hab_overlay = np.ma.masked_where(
    ~hab,
    hab
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6)
)


# RGB
axes[0].imshow(rgb_display)
axes[0].set_title(
    "Osman Sagar — Sentinel-2 RGB"
)
axes[0].axis("off")


# HAB
axes[1].imshow(rgb_display)
axes[1].imshow(
    hab_overlay,
    alpha=0.65
)
axes[1].set_title(
    f"Predicted HAB — {hab_percentage:.2f}%"
)
axes[1].axis("off")


plt.tight_layout()

plt.savefig(
    VISUALIZATION_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 117 COMPLETE")
print("=" * 70)

print("\nAnalysis CSV:")
print(os.path.abspath(REPORT_PATH))

print("\nText report:")
print(os.path.abspath(TEXT_REPORT_PATH))

print("\nVisualization:")
print(os.path.abspath(VISUALIZATION_PATH))

print("=" * 70)