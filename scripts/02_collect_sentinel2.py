# ============================================================
# STEP 2: SENTINEL-2 DATA COLLECTION
# ============================================================

from pathlib import Path

import rasterio


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SENTINEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sentinel2"
)

SENTINEL_FILE = (
    SENTINEL_DIR
    / "HAB_Sentinel2_2023_2025.tif"
)


def collect_sentinel2():
    """
    Verify the Sentinel-2 dataset already collected
    from Google Earth Engine.
    """

    print("=" * 60)
    print("SENTINEL-2 DATA COLLECTION")
    print("=" * 60)

    if not SENTINEL_FILE.exists():

        print("Sentinel-2 file was not found.")
        print()
        print(f"Expected location:")
        print(SENTINEL_FILE)

        return False

    print("Sentinel-2 dataset found.")
    print(f"File: {SENTINEL_FILE}")
    print()

    with rasterio.open(SENTINEL_FILE) as src:

        print("Raster information")
        print("-" * 60)
        print(f"Width       : {src.width}")
        print(f"Height      : {src.height}")
        print(f"Bands       : {src.count}")
        print(f"CRS         : {src.crs}")
        print(f"Resolution  : {src.res}")

        print()
        print("Band descriptions:")

        for index, description in enumerate(
            src.descriptions,
            start=1
        ):
            print(
                f"  Band {index:02d}: {description}"
            )

    print()
    print("Sentinel-2 collection verification: PASSED")

    return True


if __name__ == "__main__":

    success = collect_sentinel2()

    if success:
        print()
        print("Step 2 completed successfully.")
    else:
        print()
        print("Step 2 could not be completed.")