"""
Step 40: Select one representative Sentinel-2 date per year.

For each year:
1. Query available Sentinel-2 observations for Hussain Sagar.
2. Select the available date closest to June 30.
3. Print the selected date.
4. Save the results as a CSV.

This step ONLY selects dates.
It does NOT download images.
"""

import ee
from pathlib import Path
import sys


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from utils.gee_utils import initialize_gee


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ID = "remote-sensing-project-507517"

WATERBODY_NAME = "Hussain Sagar"

START_YEAR = 2016
END_YEAR = 2026

# We select the observation closest to June 30
TARGET_MONTH = 6
TARGET_DAY = 30


# =========================================================
# START
# =========================================================

print("=" * 70)
print("STEP 40 - SELECT YEARLY REPRESENTATIVE DATES")
print("=" * 70)


# =========================================================
# INITIALIZE GOOGLE EARTH ENGINE FIRST
# =========================================================

initialize_gee()

print("\nGEE initialized successfully.")
print(f"Project: {PROJECT_ID}")
print(f"Waterbody: {WATERBODY_NAME}")
print("Target date: June 30")


# =========================================================
# HUSSAIN SAGAR AOI
# =========================================================

AOI = ee.Geometry.Rectangle([
    78.4623422,   # minimum longitude
    17.4085927,   # minimum latitude
    78.4867613,   # maximum longitude
    17.4373613    # maximum latitude
])

print("Hussain Sagar AOI created successfully.")


# =========================================================
# GET AVAILABLE DATES FOR ONE YEAR
# =========================================================

def get_year_dates(year):

    start_date = f"{year}-01-01"
    end_date = f"{year + 1}-01-01"

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(AOI)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
        .select(["B2"])
    )

    dates = (
        collection
        .aggregate_array("system:time_start")
        .map(
            lambda timestamp:
            ee.Date(timestamp).format("YYYY-MM-dd")
        )
        .distinct()
        .sort()
        .getInfo()
    )

    return dates


# =========================================================
# SELECT DATE CLOSEST TO JUNE 30
# =========================================================

def select_representative_date(year, dates):

    if not dates:
        return None, None

    target_date = ee.Date(
        f"{year}-{TARGET_MONTH:02d}-{TARGET_DAY:02d}"
    )

    best_date = None
    best_difference = None

    for date_string in dates:

        current_date = ee.Date(date_string)

        difference = abs(
            current_date
            .difference(target_date, "day")
            .getInfo()
        )

        if (
            best_difference is None
            or difference < best_difference
        ):
            best_difference = difference
            best_date = date_string

    return best_date, int(best_difference)


# =========================================================
# PROCESS ALL YEARS
# =========================================================

results = []

print("\nSearching for yearly representative dates...\n")


for year in range(START_YEAR, END_YEAR + 1):

    print(f"Checking {year}...", end=" ")

    try:

        dates = get_year_dates(year)

        if not dates:

            print("NO DATA")

            results.append({
                "year": year,
                "selected_date": None,
                "days_from_target": None,
                "available_observations": 0
            })

            continue

        selected_date, difference = (
            select_representative_date(
                year,
                dates
            )
        )

        print(
            f"{len(dates)} observations -> "
            f"{selected_date} "
            f"({difference} days from June 30)"
        )

        results.append({
            "year": year,
            "selected_date": selected_date,
            "days_from_target": difference,
            "available_observations": len(dates)
        })

    except Exception as error:

        print(f"ERROR: {error}")

        results.append({
            "year": year,
            "selected_date": None,
            "days_from_target": None,
            "available_observations": 0
        })


# =========================================================
# DISPLAY FINAL TABLE
# =========================================================

print("\n")
print("=" * 80)
print("FINAL YEARLY REPRESENTATIVE DATES")
print("=" * 80)

print(
    f"{'Year':<10}"
    f"{'Selected Date':<20}"
    f"{'Days from June 30':<20}"
    f"{'Available Images':<20}"
)

print("-" * 80)


for result in results:

    year = result["year"]
    date = result["selected_date"]
    difference = result["days_from_target"]
    count = result["available_observations"]

    if date is None:

        print(
            f"{year:<10}"
            f"{'NO DATA':<20}"
            f"{'-':<20}"
            f"{count:<20}"
        )

    else:

        print(
            f"{year:<10}"
            f"{date:<20}"
            f"{difference:<20}"
            f"{count:<20}"
        )


# =========================================================
# SAVE CSV
# =========================================================

output_directory = (
    PROJECT_ROOT /
    "results" /
    "historical"
)

output_directory.mkdir(
    parents=True,
    exist_ok=True
)


output_file = (
    output_directory /
    "yearly_representative_dates.csv"
)


with open(
    output_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "year,selected_date,"
        "days_from_target,available_observations\n"
    )

    for result in results:

        file.write(
            f"{result['year']},"
            f"{result['selected_date'] or ''},"
            f"{result['days_from_target'] or ''},"
            f"{result['available_observations']}\n"
        )


# =========================================================
# FINISHED
# =========================================================

print("\n")
print("=" * 80)
print("STEP 40 COMPLETED")
print("=" * 80)

print("\nRepresentative dates saved to:")
print(output_file)

print("\nNo Sentinel-2 images were downloaded.")
print("This step only selected the yearly dates.")