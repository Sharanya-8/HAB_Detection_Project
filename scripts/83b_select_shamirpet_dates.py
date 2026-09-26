import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "shamirpet_usable_dates.txt"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "shamirpet_test_dates.csv"
)

print("=" * 70)
print("STEP 83B - SELECT SHAMIRPET TEST DATES")
print("=" * 70)

# ---------------------------------------------------------
# LOAD DATES
# ---------------------------------------------------------

with open(INPUT_FILE, "r") as f:
    dates = [
        line.strip()
        for line in f
        if line.strip()
    ]

df = pd.DataFrame({
    "date": pd.to_datetime(dates)
})

df["year"] = df["date"].dt.year

print("\nTotal usable dates:", len(df))

# ---------------------------------------------------------
# SELECT UP TO 10 DATES PER YEAR
# ---------------------------------------------------------

selected_rows = []

for year in range(2016, 2026):

    year_df = df[
        df["year"] == year
    ].sort_values("date").reset_index(drop=True)

    if len(year_df) == 0:
        print(f"{year}: NO usable dates")
        continue

    number_to_select = min(10, len(year_df))

    # Evenly distribute selections across the year
    positions = np.linspace(
        0,
        len(year_df) - 1,
        number_to_select
    ).round().astype(int)

    selected = year_df.iloc[positions].copy()

    selected_rows.append(selected)

    print(
        f"{year}: "
        f"{len(year_df)} usable -> "
        f"{len(selected)} selected"
    )

# ---------------------------------------------------------
# COMBINE
# ---------------------------------------------------------

selected_df = pd.concat(
    selected_rows,
    ignore_index=True
)

selected_df = selected_df.sort_values(
    "date"
).reset_index(drop=True)

selected_df["waterbody"] = "Shamirpet_Lake"

selected_df["date"] = selected_df[
    "date"
].dt.strftime("%Y-%m-%d")

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

selected_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("SELECTED TEST DATA")
print("=" * 70)

print("Total selected dates:", len(selected_df))

print("\nDates by year:")

print(
    selected_df.groupby("date")
)

print("\nSelected dates:")

for date in selected_df["date"]:
    print(date)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("STEP 83B COMPLETE")
print("=" * 70)