# ============================================================
# VISUALIZE MULTI-DATE HUSSAIN SAGAR HAB LABELS
# ============================================================

from pathlib import Path

import numpy as np
import rasterio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent

LABEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "hab_masks"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "maps"
)


DATES = [
    "2023_01_08",
    "2023_02_12",
    "2023_03_24",
    "2023_05_28",
    "2023_10_05",
    "2024_01_28",
    "2024_03_08",
    "2024_04_22",
    "2024_06_26",
    "2025_01_22",
    "2025_02_26",
    "2025_04_04",
    "2025_06_01",
    "2025_11_13",
    "2025_12_28",
]


def visualize_labels():

    print("=" * 80)
    print("HUSSAIN SAGAR MULTI-DATE HAB LABEL VISUALIZATION")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for date in DATES:

        print()
        print("-" * 80)
        print(f"VISUALIZING: {date}")
        print("-" * 80)

        label_file = (
            LABEL_DIR
            / f"Hussain_Sagar_HAB_{date}.tif"
        )

        if not label_file.exists():

            print(
                f"WARNING: File not found:\n"
                f"{label_file}"
            )

            continue

        with rasterio.open(label_file) as src:

            label = src.read(1)

        # ----------------------------------------------------
        # Mask outside/invalid pixels
        # ----------------------------------------------------

        display = np.ma.masked_where(
            label == 255,
            label
        )

        # ----------------------------------------------------
        # Create figure
        # ----------------------------------------------------

        plt.figure(
            figsize=(10, 8)
        )

        plt.imshow(
            display,
            vmin=0,
            vmax=1
        )

        plt.title(
            f"Hussain Sagar HAB Pseudo-Labels - {date}"
        )

        plt.xlabel("Pixel")
        plt.ylabel("Pixel")

        colorbar = plt.colorbar()

        colorbar.set_label(
            "Label (0 = Non-HAB, 1 = HAB)"
        )

        plt.tight_layout()

        output_file = (
            OUTPUT_DIR
            / f"Hussain_Sagar_HAB_{date}_verification.png"
        )

        plt.savefig(
            output_file,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        # ----------------------------------------------------
        # Print statistics
        # ----------------------------------------------------

        valid_pixels = np.count_nonzero(
            label != 255
        )

        hab_pixels = np.count_nonzero(
            label == 1
        )

        if valid_pixels > 0:

            hab_percentage = (
                hab_pixels
                / valid_pixels
                * 100
            )

        else:

            hab_percentage = 0

        print(
            f"Valid pixels : {valid_pixels:,}"
        )

        print(
            f"HAB pixels   : {hab_pixels:,}"
        )

        print(
            f"HAB percentage: {hab_percentage:.2f}%"
        )

        print(
            f"Saved: {output_file.name}"
        )

    print()
    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    visualize_labels()