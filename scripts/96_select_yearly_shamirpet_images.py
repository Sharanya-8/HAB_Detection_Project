"""
STEP 96
Select one representative Sentinel-2 image for each year
from the existing Shamirpet test dataset.

Period:
2016-2025

Selection:
Choose the image closest to June 30 of each year.
"""

from pathlib import Path
import pandas as pd


# ================================================================
# PROJECT ROOT
# ================================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


# ================================================================
# INPUT DIRECTORY
# ================================================================

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sentinel2"
    / "shamirpet_test"
)


# ================================================================
# OUTPUT
# ================================================================

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "shamirpet_yearly_representative_dates.csv"
)


# ================================================================
# SETTINGS
# ================================================================

START_YEAR = 2016
END_YEAR = 2025

TARGET_MONTH = 6
TARGET_DAY = 30


# ================================================================
# CHECK INPUT
# ================================================================

if not INPUT_DIR.exists():

    raise FileNotFoundError(
        f"Shamirpet dataset not found:\n{INPUT_DIR}"
    )


# ================================================================
# FIND TIFF FILES
# ================================================================

image_files = sorted(
    INPUT_DIR.glob("*.tif")
)

print("=" * 70)
print("STEP 96 — SELECT YEARLY REPRESENTATIVE IMAGES")
print("=" * 70)

print()
print(
    "Input images:",
    len(image_files)
)

if len(image_files) == 0:

    raise RuntimeError(
        "No Sentinel-2 TIFF files found."
    )


# ================================================================
# PARSE DATES
# ================================================================

records = []

for image_path in image_files:

    try:

        date_text = image_path.stem

        date = pd.to_datetime(
            date_text,
            format="%Y-%m-%d"
        )

        records.append(
            {
                "date": date,
                "image_path": str(
                    image_path
                )
            }
        )

    except Exception:

        print(
            "Skipping file:",
            image_path.name
        )


dates_df = pd.DataFrame(
    records
)


# ================================================================
# SELECT ONE DATE PER YEAR
# ================================================================

selected = []

for year in range(
    START_YEAR,
    END_YEAR + 1
):

    year_df = dates_df[
        dates_df["date"].dt.year == year
    ].copy()

    if year_df.empty:

        print(
            f"{year}: No image available"
        )

        continue

    target_date = pd.Timestamp(
        year=year,
        month=TARGET_MONTH,
        day=TARGET_DAY
    )

    year_df["distance_days"] = (
        year_df["date"] -
        target_date
    ).abs().dt.days

    selected_row = (
        year_df
        .sort_values("distance_days")
        .iloc[0]
    )

    selected.append(
        {
            "year": year,
            "selected_date": (
                selected_row["date"]
                .strftime("%Y-%m-%d")
            ),
            "distance_from_june_30_days": int(
                selected_row["distance_days"]
            ),
            "image_path": selected_row[
                "image_path"
            ]
        }
    )


# ================================================================
# SAVE
# ================================================================

result_df = pd.DataFrame(
    selected
)

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)

result_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ================================================================
# DISPLAY
# ================================================================

print()
print("=" * 70)
print("SELECTED YEARLY IMAGES")
print("=" * 70)

print()

for _, row in result_df.iterrows():

    print(
        f"{int(row['year'])} -> "
        f"{row['selected_date']} "
        f"("
        f"{int(row['distance_from_june_30_days'])} days "
        f"from June 30"
        f")"
    )


print()
print(
    "Years selected:",
    len(result_df)
)

print()
print(
    "CSV saved:"
)

print(
    OUTPUT_CSV
)

print()
print("=" * 70)
print("STEP 96 COMPLETE")
print("=" * 70)