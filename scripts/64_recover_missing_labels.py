from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


# ============================================================
# STEP 64 - RECOVER 6 MISSING MULTI-WATERBODY HAB LABELS
# ============================================================

print("=" * 70)
print("STEP 64 - RECOVER MISSING HAB LABELS")
print("=" * 70)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
    / "missing_label_statistics.csv"
)


# ============================================================
# EXACT 6 MISSING FILES
# ============================================================

missing_files = [
    INPUT_DIR / "Hussain_Sagar" / "2017-03-25.tif",
    INPUT_DIR / "Hussain_Sagar" / "2017-04-14.tif",
    INPUT_DIR / "Hussain_Sagar" / "2017-05-04.tif",
    INPUT_DIR / "Hussain_Sagar" / "2017-05-24.tif",
    INPUT_DIR / "Hussain_Sagar" / "2017-09-01.tif",
    INPUT_DIR / "Hussain_Sagar" / "2017-11-20.tif",
]


statistics = []

successful = 0
failed = 0


# ============================================================
# PROCESS ONLY THE 6 FILES
# ============================================================

for index, input_file in enumerate(
    missing_files,
    start=1
):

    print()
    print("=" * 70)
    print(
        f"[{index}/{len(missing_files)}] "
        f"{input_file.name}"
    )
    print("=" * 70)

    if not input_file.exists():

        print(
            f"FAILED: Input file does not exist:"
        )

        print(input_file)

        failed += 1
        continue


    try:

        # ====================================================
        # OUTPUT
        # ====================================================

        waterbody = "Hussain_Sagar"

        output_dir = OUTPUT_DIR / waterbody

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = (
            output_dir
            / input_file.name.replace(
                ".tif",
                "_HAB.tif"
            )
        )


        # ====================================================
        # READ IMAGE
        # ====================================================

        with rasterio.open(input_file) as src:

            if src.count != 14:

                raise ValueError(
                    f"Expected 14 bands, "
                    f"found {src.count}"
                )


            # Band 13 = NDCI
            ndci = src.read(
                13
            ).astype(
                np.float32
            )


            # Band 14 = FAI
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


        if valid_count == 0:

            raise ValueError(
                "No valid NDCI/FAI pixels."
            )


        # ====================================================
        # VALID VALUES
        # ====================================================

        ndci_values = ndci[valid]

        fai_values = fai[valid]


        # ====================================================
        # 95TH PERCENTILE
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


        # ====================================================
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
        # OUTPUT MASK
        # ====================================================

        output_mask = np.full(
            ndci.shape,
            255,
            dtype=np.uint8
        )


        # Non-HAB
        output_mask[valid] = 0


        # HAB
        output_mask[hab] = 1


        # ====================================================
        # STATISTICS
        # ====================================================

        hab_count = int(
            np.count_nonzero(hab)
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
            f"Valid pixels     : {valid_count:,}"
        )

        print(
            f"NDCI threshold   : "
            f"{ndci_threshold:.6f}"
        )

        print(
            f"FAI threshold    : "
            f"{fai_threshold:.6f}"
        )

        print(
            f"Non-HAB pixels   : "
            f"{non_hab_count:,}"
        )

        print(
            f"HAB pixels       : "
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
        # VERIFY
        # ====================================================

        with rasterio.open(
            output_file
        ) as check:

            check_mask = check.read(1)

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
            f"Mask values: {unique_values}"
        )

        print(
            "STATUS: PASSED"
        )


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
        print("FAILED:")
        print(e)

        failed += 1


# ============================================================
# SAVE RECOVERY STATISTICS
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
print("STEP 64 COMPLETE")
print("=" * 70)

print()

print(
    f"Files attempted : {len(missing_files)}"
)

print(
    f"Successful      : {successful}"
)

print(
    f"Failed          : {failed}"
)

print()

print(
    "Recovery statistics:"
)

print(
    REPORT_FILE
)

print()
print("=" * 70)