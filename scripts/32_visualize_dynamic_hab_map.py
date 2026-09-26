from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import rasterio


# ============================================================
# CONFIGURATION
# ============================================================

MAP_FILE = Path(
    "results/maps/"
    "Dynamic_Hussain_Sagar_HAB_Swin_2025_01_02.tif"
)

IMAGE_FILE = Path(
    "data/processed/sentinel2/"
    "Dynamic_Hussain_Sagar_14Channel_Corrected.tif"
)

OUTPUT_FILE = Path(
    "results/maps/"
    "Dynamic_Hussain_Sagar_HAB_Visualization.png"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("DYNAMIC HAB MAP VISUALIZATION")
print("=" * 70)


# ============================================================
# READ HAB MAP
# ============================================================

print("\nReading HAB prediction map...")

with rasterio.open(MAP_FILE) as src:

    hab_map = src.read(1)

    transform = src.transform


# ============================================================
# READ RGB BANDS
# ============================================================

print("Reading Sentinel-2 RGB bands...")

with rasterio.open(IMAGE_FILE) as src:

    red = src.read(3).astype(np.float32)
    green = src.read(2).astype(np.float32)
    blue = src.read(1).astype(np.float32)


# ============================================================
# CREATE RGB IMAGE
# ============================================================

print("Creating RGB background...")


def stretch(band):

    valid = band[
        np.isfinite(band) &
        (band > 0)
    ]

    if len(valid) == 0:
        return np.zeros_like(band)

    low = np.percentile(
        valid,
        2
    )

    high = np.percentile(
        valid,
        98
    )

    result = (
        band - low
    ) / (
        high - low + 1e-8
    )

    return np.clip(
        result,
        0,
        1
    )


rgb = np.stack(
    [
        stretch(red),
        stretch(green),
        stretch(blue)
    ],
    axis=-1
)


# ============================================================
# CREATE HAB MASK
# ============================================================

hab_mask = (
    hab_map == 1
)

water_mask = (
    hab_map != 255
)


# ============================================================
# PLOT
# ============================================================

print("\nCreating visualization...")


fig, ax = plt.subplots(
    figsize=(10, 10)
)


ax.imshow(
    rgb,
    extent=[
        transform.c,
        transform.c + transform.a * rgb.shape[1],
        transform.f + transform.e * rgb.shape[0],
        transform.f
    ]
)


# HAB overlay

overlay = np.ma.masked_where(
    ~hab_mask,
    hab_mask
)


ax.imshow(
    overlay,
    extent=[
        transform.c,
        transform.c + transform.a * rgb.shape[1],
        transform.f + transform.e * rgb.shape[0],
        transform.f
    ],
    alpha=0.65
)


# ============================================================
# TITLE
# ============================================================

hab_pixels = int(
    np.sum(hab_mask)
)

water_pixels = int(
    np.sum(water_mask)
)

hab_percentage = (
    hab_pixels /
    water_pixels *
    100
)


ax.set_title(
    "Dynamic Swin HAB Prediction\n"
    f"Hussain Sagar — 2025-01-02\n"
    f"Predicted HAB area: {hab_percentage:.2f}%",
    fontsize=14
)


ax.set_xlabel(
    "UTM Easting (m)"
)

ax.set_ylabel(
    "UTM Northing (m)"
)


# ============================================================
# LEGEND
# ============================================================

from matplotlib.patches import Patch

legend_elements = [
    Patch(
        label="Predicted HAB"
    ),
    Patch(
        label="Waterbody"
    )
]


ax.legend(
    handles=legend_elements,
    loc="upper right"
)


ax.grid(
    alpha=0.2
)


plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    OUTPUT_FILE,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print(
    f"\nSaved visualization:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print(
    "Dynamic HAB prediction map visualization created."
)

print("=" * 70)