import os
import glob
import numpy as np
import rasterio
import matplotlib

# Use non-interactive backend to avoid Tk/Tcl errors
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "sentinel2",
    "historical"
)

MASK_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "labels",
    "hab_masks",
    "historical"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "historical",
    "heatmaps"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FIND YEARLY IMAGES
# ============================================================

image_files = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "Hussain_Sagar_*_14Channel.tif"))
)

if not image_files:
    print("ERROR: No historical Sentinel-2 images found.")
    print(f"Checked: {IMAGE_DIR}")
    raise SystemExit(1)


print("=" * 70)
print("STEP 44 - VISUALIZE HISTORICAL HAB MAPS")
print("=" * 70)

print(f"Found {len(image_files)} yearly images.")
print()


# ============================================================
# PROCESS EACH YEAR
# ============================================================

processed = 0

for image_path in image_files:

    filename = os.path.basename(image_path)

    # Example:
    # Hussain_Sagar_2016_2016-05-09_14Channel.tif

    parts = filename.split("_")

    year = parts[2]

    mask_filename = f"Hussain_Sagar_{year}_HAB_mask.tif"
    mask_path = os.path.join(MASK_DIR, mask_filename)

    if not os.path.exists(mask_path):

        print(f"[SKIP] {year}: HAB mask not found")
        print(f"       Expected: {mask_path}")
        continue

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    with rasterio.open(image_path) as src:

        # Read RGB bands
        # 14-channel order:
        # 1 B2
        # 2 B3
        # 3 B4
        # ...
        # 13 NDCI
        # 14 FAI

        blue = src.read(1).astype(np.float32)
        green = src.read(2).astype(np.float32)
        red = src.read(3).astype(np.float32)

    # --------------------------------------------------------
    # READ HAB MASK
    # --------------------------------------------------------

    with rasterio.open(mask_path) as src:

        hab_mask = src.read(1)

    # --------------------------------------------------------
    # CALCULATE HAB STATISTICS
    # --------------------------------------------------------

    water_pixels = np.sum(hab_mask > 0)
    hab_pixels = np.sum(hab_mask == 2)

    if water_pixels > 0:
        hab_percentage = (hab_pixels / water_pixels) * 100
    else:
        hab_percentage = 0.0

    # --------------------------------------------------------
    # CREATE RGB IMAGE
    # --------------------------------------------------------

    rgb = np.stack(
        [
            red,
            green,
            blue
        ],
        axis=-1
    )

    # Robust percentile stretch
    valid = np.isfinite(rgb)

    if np.any(valid):

        low = np.nanpercentile(rgb[valid], 2)
        high = np.nanpercentile(rgb[valid], 98)

        if high > low:
            rgb = (rgb - low) / (high - low)

        rgb = np.clip(rgb, 0, 1)

    # --------------------------------------------------------
    # CREATE HAB OVERLAY
    # --------------------------------------------------------

    hab_overlay = np.full(
        hab_mask.shape,
        np.nan,
        dtype=np.float32
    )

    # Only show HAB pixels
    hab_overlay[hab_mask == 2] = 1

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 8))

    ax.imshow(rgb)

    # HAB shown as transparent overlay
    ax.imshow(
        hab_overlay,
        cmap="Reds",
        alpha=0.65,
        vmin=0,
        vmax=1
    )

    ax.set_title(
        f"Hussain Sagar - {year}\n"
        f"Historical HAB Pseudo-Label Heatmap | "
        f"HAB: {hab_percentage:.2f}%",
        fontsize=14
    )

    ax.set_xlabel("Pixel")
    ax.set_ylabel("Pixel")

    # --------------------------------------------------------
    # LEGEND
    # --------------------------------------------------------

    legend_elements = [
        Patch(
            facecolor="red",
            alpha=0.65,
            label="HAB"
        )
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper right"
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        f"Hussain_Sagar_{year}_HAB_heatmap.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    processed += 1

    print(
        f"[OK] {year} | "
        f"Water pixels: {water_pixels} | "
        f"HAB pixels: {hab_pixels} | "
        f"HAB: {hab_percentage:.2f}%"
    )

    print(f"     Saved: {output_path}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("STEP 44 COMPLETE")
print("=" * 70)

print(f"Yearly maps created: {processed}")
print(f"Output folder:")
print(OUTPUT_DIR)
print()
print("These maps are based on spectral HAB pseudo-labels.")
print("They are NOT field-validated ground truth.")