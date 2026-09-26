"""
Step 42: Validate historical 14-channel Sentinel-2 images.

Checks:
- File exists
- Number of bands
- Width / height
- CRS
- Data type
- No-data information
- NaN percentage
- Infinite values
- Basic statistics for all 14 bands
"""

from pathlib import Path
import sys
import numpy as np
import rasterio


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HISTORICAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sentinel2"
    / "historical"
)


# =========================================================
# EXPECTED CONFIGURATION
# =========================================================

EXPECTED_BANDS = 14

BAND_NAMES = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
    "NDWI",
    "MNDWI",
    "NDCI",
    "FAI",
]


# =========================================================
# START
# =========================================================

print("=" * 80)
print("STEP 42 - VALIDATE HISTORICAL SENTINEL-2 IMAGES")
print("=" * 80)

print(f"\nHistorical image directory:")
print(HISTORICAL_DIR)


# =========================================================
# FIND TIFF FILES
# =========================================================

files = sorted(
    HISTORICAL_DIR.glob(
        "Hussain_Sagar_*_14Channel.tif"
    )
)

print(f"\nFound {len(files)} historical TIFF files.")


if len(files) == 0:

    raise FileNotFoundError(
        "\nNo historical TIFF files found."
    )


# =========================================================
# VALIDATION
# =========================================================

all_valid = True

summary = []


for file_path in files:

    print("\n" + "-" * 80)
    print(f"FILE: {file_path.name}")
    print("-" * 80)

    try:

        with rasterio.open(file_path) as src:

            # -------------------------------------------------
            # Basic metadata
            # -------------------------------------------------

            band_count = src.count
            width = src.width
            height = src.height
            crs = src.crs
            dtype = src.dtypes[0]
            nodata = src.nodata

            print(f"Bands:       {band_count}")
            print(f"Size:        {width} x {height}")
            print(f"CRS:         {crs}")
            print(f"Data type:   {dtype}")
            print(f"NoData:      {nodata}")

            # -------------------------------------------------
            # Check band count
            # -------------------------------------------------

            if band_count != EXPECTED_BANDS:

                print(
                    f"❌ WRONG BAND COUNT: "
                    f"expected {EXPECTED_BANDS}, "
                    f"found {band_count}"
                )

                all_valid = False

            else:

                print("✅ Band count correct")

            # -------------------------------------------------
            # Read image
            # -------------------------------------------------

            data = src.read()

            # -------------------------------------------------
            # NaN / infinite checks
            # -------------------------------------------------

            total_pixels = data.size

            nan_count = np.isnan(data).sum()

            infinite_count = np.isinf(data).sum()

            nan_percentage = (
                nan_count / total_pixels * 100
            )

            infinite_percentage = (
                infinite_count / total_pixels * 100
            )

            print(
                f"NaN values:       "
                f"{nan_count:,} "
                f"({nan_percentage:.4f}%)"
            )

            print(
                f"Infinite values:  "
                f"{infinite_count:,} "
                f"({infinite_percentage:.4f}%)"
            )

            # -------------------------------------------------
            # Band statistics
            # -------------------------------------------------

            print("\nBand statistics:")

            for band_index in range(band_count):

                band = data[band_index]

                finite_values = band[
                    np.isfinite(band)
                ]

                if finite_values.size == 0:

                    print(
                        f"  Band {band_index + 1:02d} "
                        f"{BAND_NAMES[band_index]:<6} "
                        f"NO FINITE VALUES"
                    )

                    all_valid = False

                    continue

                minimum = np.min(
                    finite_values
                )

                maximum = np.max(
                    finite_values
                )

                mean = np.mean(
                    finite_values
                )

                print(
                    f"  Band {band_index + 1:02d} "
                    f"{BAND_NAMES[band_index]:<6} "
                    f"min={minimum:.6f} "
                    f"max={maximum:.6f} "
                    f"mean={mean:.6f}"
                )

            # -------------------------------------------------
            # Overall validity
            # -------------------------------------------------

            if (
                band_count == EXPECTED_BANDS
                and infinite_count == 0
            ):

                print("\n✅ IMAGE PASSED BASIC VALIDATION")

                status = "PASS"

            else:

                print("\n❌ IMAGE NEEDS ATTENTION")

                all_valid = False

                status = "CHECK"

            summary.append(
                {
                    "file": file_path.name,
                    "bands": band_count,
                    "width": width,
                    "height": height,
                    "crs": str(crs),
                    "dtype": dtype,
                    "nan_percentage": nan_percentage,
                    "infinite_percentage": infinite_percentage,
                    "status": status,
                }
            )

    except Exception as error:

        print(
            f"\n❌ ERROR READING FILE: {error}"
        )

        all_valid = False

        summary.append(
            {
                "file": file_path.name,
                "bands": "-",
                "width": "-",
                "height": "-",
                "crs": "-",
                "dtype": "-",
                "nan_percentage": "-",
                "infinite_percentage": "-",
                "status": "ERROR",
            }
        )


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n\n")
print("=" * 100)
print("STEP 42 FINAL SUMMARY")
print("=" * 100)

print(
    f"\n{'File':<50}"
    f"{'Bands':<8}"
    f"{'Size':<15}"
    f"{'NaN %':<12}"
    f"{'Status':<10}"
)

print("-" * 100)


for item in summary:

    size = (
        f"{item['width']}x{item['height']}"
        if item["width"] != "-"
        else "-"
    )

    nan_value = (
        f"{item['nan_percentage']:.4f}"
        if isinstance(
            item["nan_percentage"],
            (int, float)
        )
        else "-"
    )

    print(
        f"{item['file']:<50}"
        f"{str(item['bands']):<8}"
        f"{size:<15}"
        f"{nan_value:<12}"
        f"{item['status']:<10}"
    )


# =========================================================
# FINAL RESULT
# =========================================================

print("\n")

if all_valid:

    print("=" * 80)
    print("✅ ALL HISTORICAL IMAGES PASSED BASIC VALIDATION")
    print("=" * 80)

    print(
        "\nThe 11 yearly images are ready for the next processing step."
    )

else:

    print("=" * 80)
    print("⚠️ SOME IMAGES NEED ATTENTION")
    print("=" * 80)

    print(
        "\nDo NOT continue to HAB detection yet."
    )

print("\nStep 42 finished.")