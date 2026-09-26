from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


# ============================================================
# STEP 63 - CREATE MULTI-WATERBODY HAB PSEUDO-LABELS
# ============================================================

print("=" * 70)
print("STEP 63 - CREATE MULTI-WATERBODY HAB PSEUDO-LABELS")
print("=" * 70)


# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------
# INPUT / OUTPUT
# ------------------------------------------------------------

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sentinel2"
    / "multi_waterbody"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "hab_masks"
    / "multi_waterbody"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multilocation_label_statistics.csv"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FIND ALL IMAGES
# ============================================================

tif_files = sorted(
    INPUT_DIR.rglob("*.tif")
)

print()
print(
    f"Input TIFF files found: {len(tif_files)}"
)


if len(tif_files) != 397:

    print()
    print(
        "WARNING: Expected 397 images."
    )

    print(
        f"Found: {len(tif_files)}"
    )


# ============================================================
# PROCESS IMAGES
# ============================================================

statistics = []

successful = 0
failed = 0


for index, input_file in enumerate(
    tif_files,
    start=1
):

    print()
    print("=" * 70)

    print(
        f"[{index}/{len(tif_files)}] "
        f"{input_file.name}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Determine waterbody
    # --------------------------------------------------------

    relative_path = input_file.relative_to(
        INPUT_DIR
    )

    waterbody = relative_path.parts[0]


    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    waterbody_output_dir = (
        OUTPUT_DIR
        / waterbody
    )

    waterbody_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Output file
    # --------------------------------------------------------

    output_file = (
        waterbody_output_dir
        / input_file.name.replace(
            ".tif",
            "_HAB.tif"
        )
    )


    # --------------------------------------------------------
    # Skip if already created
    # --------------------------------------------------------

    if output_file.exists():

        print(
            "Label already exists - skipping."
        )

        continue


    try:

        # ====================================================
        # READ IMAGE
        # ====================================================

        with rasterio.open(
            input_file
        ) as src:

            if src.count != 14:

                raise ValueError(
                    f"Expected 14 bands, "
                    f"found {src.count}"
                )


            # ------------------------------------------------
            # Band 13 = NDCI
            # Band 14 = FAI
            # ------------------------------------------------

            ndci = src.read(
                13
            ).astype(
                np.float32
            )

            fai = src.read(
                14
            ).astype(
                np.float32
            )


            profile = src.profile.copy()


        # ====================================================
        # VALID PIXELS
        # ====================================================

        valid = (
            np.isfinite(ndci)
            & np.isfinite(fai)
            & (ndci != 0)
            & (fai != 0)
        )


        valid_count = int(
            np.count_nonzero(valid)
        )


        print(
            f"Valid pixels: "
            f"{valid_count:,}"
        )


        if valid_count == 0:

            raise ValueError(
                "No valid NDCI/FAI pixels."
            )


        # ====================================================
        # EXTRACT VALID VALUES
        # ====================================================

        ndci_values = ndci[
            valid
        ]

        fai_values = fai[
            valid
        ]


        # ====================================================
        # DATE-SPECIFIC 95TH PERCENTILES
        # ====================================================

        ndci_threshold = float(
            np.percentile(
                ndci_values,
                95
            )
        )

        fai_threshold = float(
            np.percentile(
                fai_values,
                95
            )
        )


        print(
            f"NDCI 95th percentile: "
            f"{ndci_threshold:.6f}"
        )

        print(
            f"FAI 95th percentile : "
            f"{fai_threshold:.6f}"
        )


        # ====================================================
        # HAB CANDIDATES
        #
        # SAME METHODOLOGY AS EXISTING SCRIPT:
        #
        # HAB = HIGH NDCI OR HIGH FAI
        # ====================================================

        hab = (
            valid
            & (
                (ndci > ndci_threshold)
                |
                (fai > fai_threshold)
            )
        )


        # ====================================================
        # CREATE OUTPUT MASK
        #
        # 0   = Non-HAB valid pixels
        # 1   = HAB candidate
        # 255 = Invalid / outside valid data
        # ====================================================

        output_mask = np.full(
            ndci.shape,
            255,
            dtype=np.uint8
        )


        output_mask[
            valid
        ] = 0


        output_mask[
            hab
        ] = 1


        # ====================================================
        # STATISTICS
        # ====================================================

        hab_count = int(
            np.count_nonzero(
                hab
            )
        )

        non_hab_count = int(
            np.count_nonzero(
                output_mask == 0
            )
        )

        invalid_count = int(
            np.count_nonzero(
                output_mask == 255
            )
        )

        hab_percentage = (
            hab_count
            / valid_count
            * 100
        )


        print()
        print(
            "LABEL STATISTICS"
        )

        print(
            f"Valid pixels     : "
            f"{valid_count:,}"
        )

        print(
            f"Non-HAB pixels   : "
            f"{non_hab_count:,}"
        )

        print(
            f"HAB candidates   : "
            f"{hab_count:,}"
        )

        print(
            f"Invalid pixels   : "
            f"{invalid_count:,}"
        )

        print(
            f"HAB percentage   : "
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


        # ====================================================
        # VERIFY OUTPUT
        # ====================================================

        with rasterio.open(
            output_file
        ) as check:

            check_mask = check.read(
                1
            )

            unique_values = np.unique(
                check_mask
            )


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

            raise ValueError(
                f"Unexpected mask values: "
                f"{unique_values}"
            )


        print()
        print(
            f"Saved: {output_file}"
        )

        print(
            f"Mask values: "
            f"{unique_values}"
        )

        print(
            "STATUS: PASSED"
        )


        # ====================================================
        # SAVE STATISTICS
        # ====================================================

        statistics.append({

            "waterbody":
                waterbody,

            "file":
                input_file.name,

            "valid_pixels":
                valid_count,

            "non_hab_pixels":
                non_hab_count,

            "hab_pixels":
                hab_count,

            "invalid_pixels":
                invalid_count,

            "hab_percentage":
                hab_percentage,

            "ndci_threshold":
                ndci_threshold,

            "fai_threshold":
                fai_threshold,

        })


        successful += 1


    except Exception as e:

        print()
        print(
            "FAILED:"
        )

        print(e)

        failed += 1


# ============================================================
# SAVE STATISTICS
# ============================================================

statistics_df = pd.DataFrame(
    statistics
)


statistics_df.to_csv(
    REPORT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("STEP 63 COMPLETE")
print("=" * 70)

print()

print(
    f"Input images : {len(tif_files)}"
)

print(
    f"Successful   : {successful}"
)

print(
    f"Failed       : {failed}"
)

print()

print(
    "HAB label directory:"
)

print(
    OUTPUT_DIR
)

print()

print(
    "Statistics report:"
)

print(
    REPORT_FILE
)

print()
print("=" * 70)