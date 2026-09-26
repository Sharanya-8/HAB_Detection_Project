from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


# ============================================================
# STEP 62 - VALIDATE MULTI-WATERBODY DATA
# ============================================================

print("=" * 70)
print("STEP 62 - VALIDATE MULTI-WATERBODY DATA")
print("=" * 70)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sentinel2"
    / "multi_waterbody"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multilocation_validation.csv"
)


# ------------------------------------------------------------
# Find TIFF files
# ------------------------------------------------------------

tif_files = sorted(
    DATA_DIR.rglob("*.tif")
)

print()
print(
    f"TIFF files found: {len(tif_files)}"
)


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

results = []


for i, tif_file in enumerate(tif_files, start=1):

    print(
        f"[{i}/{len(tif_files)}] "
        f"{tif_file.name}"
    )

    relative_path = tif_file.relative_to(
        DATA_DIR
    )

    waterbody = relative_path.parts[0]

    record = {
        "waterbody": waterbody,
        "file": str(relative_path),
        "valid": False,
        "bands": None,
        "width": None,
        "height": None,
        "crs": None,
        "dtype": None,
        "nan_pixels": None,
        "inf_pixels": None,
        "reason": ""
    }

    try:

        with rasterio.open(tif_file) as src:

            record["bands"] = src.count
            record["width"] = src.width
            record["height"] = src.height
            record["crs"] = str(src.crs)
            record["dtype"] = str(src.dtypes[0])


            # ------------------------------------------------
            # Read all bands
            # ------------------------------------------------

            data = src.read()


            # ------------------------------------------------
            # Check NaN
            # ------------------------------------------------

            nan_count = int(
                np.isnan(data).sum()
            )

            record["nan_pixels"] = nan_count


            # ------------------------------------------------
            # Check infinity
            # ------------------------------------------------

            inf_count = int(
                np.isinf(data).sum()
            )

            record["inf_pixels"] = inf_count


            # ------------------------------------------------
            # Validation conditions
            # ------------------------------------------------

            problems = []


            if src.count != 14:

                problems.append(
                    f"expected 14 bands, found {src.count}"
                )


            if src.width == 0 or src.height == 0:

                problems.append(
                    "invalid image dimensions"
                )


            if nan_count > 0:

                problems.append(
                    f"{nan_count} NaN values"
                )


            if inf_count > 0:

                problems.append(
                    f"{inf_count} infinite values"
                )


            if src.crs is None:

                problems.append(
                    "missing CRS"
                )


            if problems:

                record["reason"] = "; ".join(
                    problems
                )

            else:

                record["valid"] = True
                record["reason"] = "OK"


    except Exception as e:

        record["reason"] = str(e)


    results.append(record)


# ------------------------------------------------------------
# Save report
# ------------------------------------------------------------

results_df = pd.DataFrame(
    results
)


REPORT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


results_df.to_csv(
    REPORT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

valid_count = int(
    results_df["valid"].sum()
)

invalid_count = (
    len(results_df)
    - valid_count
)


print()
print("=" * 70)
print("STEP 62 COMPLETE")
print("=" * 70)

print()

print(
    f"Total TIFF files: {len(results_df)}"
)

print(
    f"Valid files: {valid_count}"
)

print(
    f"Invalid files: {invalid_count}"
)

print()

print(
    "Waterbody-wise counts:"
)

print(
    results_df["waterbody"]
    .value_counts()
)

print()

print(
    "Validation report:"
)

print(
    REPORT_FILE
)

print()
print("=" * 70)