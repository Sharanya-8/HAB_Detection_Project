import sys
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import ee

from utils.gee_utils import (
    initialize_gee,
    get_masked_sentinel2_collection,
    get_available_dates,
)


# ============================================================
# HUSSAIN SAGAR
# ============================================================

LATITUDE = 17.422977
LONGITUDE = 78.474552


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("YEAR-BY-YEAR SENTINEL-2 AVAILABILITY CHECK")
    print("=" * 70)

    # --------------------------------------------------------
    # Initialize GEE
    # --------------------------------------------------------

    print("\nInitializing Google Earth Engine...")

    initialize_gee()

    print("GEE initialized successfully.")

    # --------------------------------------------------------
    # Create Hussain Sagar point
    # --------------------------------------------------------

    point = ee.Geometry.Point([
        LONGITUDE,
        LATITUDE
    ])

    # Use a smaller buffer for the availability check
    aoi = point.buffer(3000)

    # --------------------------------------------------------
    # Check each year separately
    # --------------------------------------------------------

    yearly_results = {}

    for year in range(2016, 2027):

        print("\n" + "-" * 70)
        print(f"CHECKING YEAR: {year}")
        print("-" * 70)

        start_date = f"{year}-01-01"
        end_date = f"{year + 1}-01-01"

        try:

            collection = get_masked_sentinel2_collection(
                aoi,
                start_date,
                end_date,
                cloud_threshold=40
            )

            dates = get_available_dates(collection)

            yearly_results[year] = dates

            print(
                f"Usable observation dates: {len(dates)}"
            )

            if len(dates) > 0:

                print("First available dates:")

                for date in dates[:5]:
                    print(f"  {date}")

                if len(dates) > 5:
                    print(
                        f"  ... and "
                        f"{len(dates) - 5} more"
                    )

            else:

                print("No usable observations found.")

        except Exception as error:

            print(f"ERROR for {year}:")
            print(error)

            yearly_results[year] = []

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL YEARLY SUMMARY")
    print("=" * 70)

    available_years = []

    for year in range(2016, 2027):

        dates = yearly_results.get(year, [])

        if len(dates) > 0:

            available_years.append(year)

            print(
                f"{year}: "
                f"{len(dates)} usable observations"
            )

        else:

            print(
                f"{year}: "
                f"NO usable observations"
            )

    print("\n" + "=" * 70)

    print(
        f"Years with usable Sentinel-2 data: "
        f"{len(available_years)}"
    )

    print(
        "Available years:"
    )

    print(available_years)

    print("=" * 70)

    print("\nAvailability check completed.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()