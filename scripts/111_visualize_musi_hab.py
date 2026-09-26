import matplotlib
matplotlib.use("Agg")
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt


print("=" * 70)
print("STEP 111 — VISUALIZE MUSI RIVER HAB PREDICTION")
print("=" * 70)


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

OUTPUT_DIR = "results/swin/generic_detection/visualization"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "Musi_River_2025-12-18_HAB_visualization.png"
)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\nLoading data...")

with rasterio.open(IMAGE_PATH) as src:
    image = src.read()
    transform = src.transform

with rasterio.open(WATER_MASK_PATH) as src:
    water_mask = src.read(1)

with rasterio.open(PREDICTION_PATH) as src:
    prediction = src.read(1)


# ------------------------------------------------------------
# CREATE RGB IMAGE
# ------------------------------------------------------------

# Sentinel-2:
# B2 = band 1
# B3 = band 2
# B4 = band 3

red = image[2]
green = image[1]
blue = image[0]

rgb = np.stack([red, green, blue], axis=-1)

# Robust stretching for visualization
valid = np.isfinite(rgb)

if valid.any():
    low = np.nanpercentile(rgb[valid], 2)
    high = np.nanpercentile(rgb[valid], 98)

    if high > low:
        rgb = (rgb - low) / (high - low)

rgb = np.clip(rgb, 0, 1)


# ------------------------------------------------------------
# MASK OUT NON-WATER
# ------------------------------------------------------------

display_rgb = rgb.copy()

display_rgb[water_mask == 0] = 0


# ------------------------------------------------------------
# CREATE HAB OVERLAY
# ------------------------------------------------------------

hab = (
    (prediction == 1) &
    (water_mask == 1)
)


# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------

plt.figure(figsize=(10, 8))

plt.imshow(display_rgb)

# HAB pixels shown with transparent overlay
overlay = np.zeros(
    (hab.shape[0], hab.shape[1], 4),
    dtype=float
)

overlay[hab, 0] = 1.0
overlay[hab, 3] = 0.55

plt.imshow(overlay)

plt.title(
    "Musi River — Sentinel-2 + Predicted HAB\n"
    "2025-12-18"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# RESULT
# ------------------------------------------------------------

hab_pixels = int(hab.sum())
water_pixels = int((water_mask == 1).sum())

coverage = (
    hab_pixels / water_pixels * 100
    if water_pixels > 0
    else 0
)

print("\n" + "=" * 70)
print("STEP 111 COMPLETE")
print("=" * 70)

print(f"\nWater pixels : {water_pixels:,}")
print(f"HAB pixels   : {hab_pixels:,}")
print(f"HAB coverage : {coverage:.2f}%")

print("\nVisualization saved to:")
print(os.path.abspath(OUTPUT_PATH))

print("=" * 70)