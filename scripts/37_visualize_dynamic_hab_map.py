from pathlib import Path
import sys

# ============================================================
# FORCE MATPLOTLIB TO USE NON-GUI BACKEND
# ============================================================

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# INPUT / OUTPUT
# ============================================================

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "maps"
    / "Dynamic_Hussain_Sagar_HAB_Swin_2025-01-02.tif"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "maps"
    / "Dynamic_Hussain_Sagar_HAB_Swin_2025-01-02.png"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DYNAMIC HAB MAP VISUALIZATION")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input HAB map not found:\n{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # READ MAP
    # --------------------------------------------------------

    print()
    print("Reading final HAB map...")

    with rasterio.open(INPUT_FILE) as src:

        hab_map = src.read(1)
        bounds = src.bounds

    # --------------------------------------------------------
    # CREATE MASKS
    # --------------------------------------------------------

    water_mask = hab_map > 0

    non_hab_mask = hab_map == 1

    hab_mask = hab_map == 2

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    water_pixels = int(
        np.sum(water_mask)
    )

    non_hab_pixels = int(
        np.sum(non_hab_mask)
    )

    hab_pixels = int(
        np.sum(hab_mask)
    )

    if water_pixels > 0:

        hab_percentage = (
            hab_pixels
            / water_pixels
            * 100
        )

    else:

        hab_percentage = 0.0

    print(
        f"  Waterbody pixels: {water_pixels}"
    )

    print(
        f"  Non-HAB pixels:   {non_hab_pixels}"
    )

    print(
        f"  HAB pixels:       {hab_pixels}"
    )

    print(
        f"  HAB percentage:   {hab_percentage:.2f}%"
    )

    # --------------------------------------------------------
    # CREATE FIGURE
    # --------------------------------------------------------

    print()
    print("Creating visualization...")

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    # --------------------------------------------------------
    # NON-HAB
    # --------------------------------------------------------

    non_hab_display = np.full(
        hab_map.shape,
        np.nan,
        dtype=float
    )

    non_hab_display[
        non_hab_mask
    ] = 0

    ax.imshow(
        non_hab_display,
        extent=[
            bounds.left,
            bounds.right,
            bounds.bottom,
            bounds.top
        ],
        origin="upper"
    )

    # --------------------------------------------------------
    # HAB
    # --------------------------------------------------------

    hab_display = np.full(
        hab_map.shape,
        np.nan,
        dtype=float
    )

    hab_display[
        hab_mask
    ] = 1

    ax.imshow(
        hab_display,
        extent=[
            bounds.left,
            bounds.right,
            bounds.bottom,
            bounds.top
        ],
        origin="upper",
        alpha=0.85
    )

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    ax.set_title(
        "Hussain Sagar Harmful Algal Bloom Detection\n"
        "Swin Transformer - 2025-01-02",
        fontsize=15
    )

    ax.set_xlabel(
        "Longitude"
    )

    ax.set_ylabel(
        "Latitude"
    )

    # --------------------------------------------------------
    # RESULT INFORMATION
    # --------------------------------------------------------

    result_text = (
        f"Waterbody pixels: {water_pixels:,}\n"
        f"Non-HAB pixels: {non_hab_pixels:,}\n"
        f"HAB pixels: {hab_pixels:,}\n"
        f"Estimated HAB: {hab_percentage:.2f}%"
    )

    ax.text(
        0.02,
        0.02,
        result_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.85
        )
    )

    # --------------------------------------------------------
    # GRID
    # --------------------------------------------------------

    ax.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    print()
    print("Saving visualization...")

    plt.savefig(
        OUTPUT_FILE,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SUCCESS")
    print("=" * 70)

    print()
    print(
        "Visualization saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        f"Estimated HAB percentage: "
        f"{hab_percentage:.2f}%"
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()