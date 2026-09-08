# ============================================================
# STEP 4: CREATE MULTI-DATE HAB LABELS
# HUSSAIN SAGAR LARGE-AOI DATASET
# ============================================================

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sentinel2"
)

WATER_MASK_PATH = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "hab_masks"
    / "Hussain_Sagar_water_mask.tif"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "hab_masks"
)


# ============================================================
# 15 SELECTED OBSERVATION DATES
# ============================================================

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


# ============================================================
# CREATE HAB LABELS
# ============================================================

def create_hab_labels():

    print("=" * 80)
    print("HUSSAIN SAGAR LARGE-AOI MULTI-DATE HAB LABEL CREATION")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # CHECK WATER MASK
    # ========================================================

    if not WATER_MASK_PATH.exists():

        print()
        print("ERROR: Hussain Sagar water mask not found:")
        print(WATER_MASK_PATH)

        return False

    # ========================================================
    # LOAD ORIGINAL WATER MASK
    # ========================================================

    print()
    print("Loading Hussain Sagar water mask...")

    with rasterio.open(WATER_MASK_PATH) as water_src:

        original_water = water_src.read(1)

        water_transform = water_src.transform
        water_crs = water_src.crs

        water_height = water_src.height
        water_width = water_src.width

    print("Water mask loaded.")
    print(
        f"Water mask dimensions: "
        f"{water_height} x {water_width}"
    )
    print(
        f"Water mask CRS: "
        f"{water_crs}"
    )

    # ========================================================
    # PROCESS EACH DATE
    # ========================================================

    successful_dates = 0

    for date in DATES:

        print()
        print("-" * 80)
        print(f"PROCESSING DATE: {date}")
        print("-" * 80)

        # ----------------------------------------------------
        # LARGE-AOI INPUT FILE
        # ----------------------------------------------------

        input_file = (
            PROCESSED_DIR
            / f"Hussain_Sagar_Large_{date}.tif"
        )

        # ----------------------------------------------------
        # OUTPUT LABEL FILE
        # ----------------------------------------------------

        output_file = (
            OUTPUT_DIR
            / f"Hussain_Sagar_HAB_{date}.tif"
        )

        if not input_file.exists():

            print()
            print("WARNING: Input file not found:")
            print(input_file)

            continue

        # ====================================================
        # READ SENTINEL-2 IMAGE
        # ====================================================

        with rasterio.open(input_file) as src:

            print(
                f"Image dimensions: "
                f"{src.height} x {src.width}"
            )

            print(
                f"Image CRS: "
                f"{src.crs}"
            )

            # ------------------------------------------------
            # Band 13 = NDCI
            # Band 14 = FAI
            # ------------------------------------------------

            ndci = src.read(13)
            fai = src.read(14)

            profile = src.profile.copy()

            # =================================================
            # ALIGN WATER MASK TO LARGE-AOI IMAGE
            # =================================================

            aligned_water = np.zeros(
                (src.height, src.width),
                dtype=np.uint8
            )

            reproject(
                source=original_water,
                destination=aligned_water,
                src_transform=water_transform,
                src_crs=water_crs,
                dst_transform=src.transform,
                dst_crs=src.crs,
                resampling=Resampling.nearest
            )

        # ====================================================
        # WATER PIXELS
        # ====================================================

        water_pixels = aligned_water == 1

        water_count = np.count_nonzero(
            water_pixels
        )

        print(
            f"Water pixels: "
            f"{water_count:,}"
        )

        if water_count == 0:

            print(
                "WARNING: No Hussain Sagar water pixels "
                "found for this date."
            )

            continue

        # ====================================================
        # VALID PIXELS
        # ====================================================

        valid = (
            water_pixels
            & np.isfinite(ndci)
            & np.isfinite(fai)
            & (ndci != 0)
            & (fai != 0)
        )

        valid_count = np.count_nonzero(
            valid
        )

        print(
            f"Valid water pixels: "
            f"{valid_count:,}"
        )

        if valid_count == 0:

            print(
                "WARNING: No valid water pixels "
                "available for this date."
            )

            continue

        # ====================================================
        # EXTRACT VALID NDCI AND FAI VALUES
        # ====================================================

        ndci_values = ndci[valid]
        fai_values = fai[valid]

        # ====================================================
        # CALCULATE DATE-SPECIFIC 95TH PERCENTILES
        # ====================================================

        ndci_threshold = np.percentile(
            ndci_values,
            95
        )

        fai_threshold = np.percentile(
            fai_values,
            95
        )

        print()
        print(
            f"NDCI 95th percentile: "
            f"{ndci_threshold:.6f}"
        )

        print(
            f"FAI 95th percentile : "
            f"{fai_threshold:.6f}"
        )

        # ====================================================
        # CREATE HAB CANDIDATES
        #
        # HAB = HIGH NDCI OR HIGH FAI
        # ====================================================

        hab = (
            valid
            & (
                (ndci > ndci_threshold)
                | (fai > fai_threshold)
            )
        )

        # ====================================================
        # CREATE FINAL LABEL MASK
        #
        # 0   = Non-HAB water
        # 1   = HAB candidate
        # 255 = Outside water / invalid
        # ====================================================

        output_mask = np.full(
            ndci.shape,
            255,
            dtype=np.uint8
        )

        # Non-HAB valid water
        output_mask[
            water_pixels & valid
        ] = 0

        # HAB candidates
        output_mask[
            hab
        ] = 1

        # ====================================================
        # STATISTICS
        # ====================================================

        hab_count = np.count_nonzero(
            hab
        )

        non_hab_count = np.count_nonzero(
            output_mask == 0
        )

        outside_count = np.count_nonzero(
            output_mask == 255
        )

        hab_percentage = (
            hab_count
            / valid_count
            * 100
        )

        print()
        print("LABEL STATISTICS")
        print("-" * 40)

        print(
            f"Valid water pixels : "
            f"{valid_count:,}"
        )

        print(
            f"Non-HAB pixels     : "
            f"{non_hab_count:,}"
        )

        print(
            f"HAB candidates     : "
            f"{hab_count:,}"
        )

        print(
            f"Outside / invalid  : "
            f"{outside_count:,}"
        )

        print(
            f"HAB percentage     : "
            f"{hab_percentage:.2f}%"
        )

        # ====================================================
        # SAVE LABEL
        # ====================================================

        profile.update(
            dtype="uint8",
            count=1,
            nodata=255,
            compress="lzw"
        )

        with rasterio.open(
            output_file,
            "w",
            **profile
        ) as dst:

            dst.write(
                output_mask,
                1
            )

        print()
        print(
            f"Saved: "
            f"{output_file.name}"
        )

        # ====================================================
        # VERIFY SAVED LABEL
        # ====================================================

        with rasterio.open(
            output_file
        ) as check:

            check_mask = check.read(1)

            unique_values = np.unique(
                check_mask
            )

            print(
                "Mask values:",
                unique_values
            )

            # Check expected values
            expected_values = {
                0,
                1,
                255
            }

            actual_values = set(
                unique_values.tolist()
            )

            if not actual_values.issubset(
                expected_values
            ):

                print(
                    "WARNING: Unexpected mask values found."
                )

                continue

        print(
            "STATUS: PASSED"
        )

        successful_dates += 1

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("MULTI-DATE HAB LABEL CREATION COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Expected dates : {len(DATES)}"
    )

    print(
        f"Successful     : {successful_dates}"
    )

    print(
        f"Failed/missing : "
        f"{len(DATES) - successful_dates}"
    )

    print()

    if successful_dates == len(DATES):

        print(
            "STATUS: SUCCESS"
        )

        return True

    else:

        print(
            "STATUS: WARNING - NOT ALL DATES PROCESSED"
        )

        return False


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    create_hab_labels()