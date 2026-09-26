from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

SATELLITE_FILE = DATA_DIR / "Hussain_Sagar_Dense_Temporal_Dataset.csv"
ENVIRONMENTAL_FILE = DATA_DIR / "Hussain_Sagar_Environmental_175_Dates.csv"

OUTPUT_FILE = DATA_DIR / "Hussain_Sagar_Complete_175_Temporal_Dataset.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("MERGING DENSE SATELLITE + ENVIRONMENTAL DATA")
print("=" * 60)

satellite = pd.read_csv(SATELLITE_FILE)
environmental = pd.read_csv(ENVIRONMENTAL_FILE)

print("\nSatellite rows:", len(satellite))
print("Environmental rows:", len(environmental))


# ============================================================
# CONVERT DATES
# ============================================================

satellite["date"] = pd.to_datetime(satellite["date"])
environmental["date"] = pd.to_datetime(environmental["date"])


# ============================================================
# REMOVE UNNECESSARY ENVIRONMENTAL COLUMNS
# ============================================================

environmental = environmental.drop(
    columns=["system:index", ".geo"],
    errors="ignore"
)


# ============================================================
# MERGE USING DATE
# ============================================================

merged = pd.merge(
    satellite,
    environmental,
    on="date",
    how="inner"
)


# ============================================================
# SORT BY DATE
# ============================================================

merged = merged.sort_values("date").reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

merged.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# VERIFICATION
# ============================================================

print("\n" + "=" * 60)
print("MERGE COMPLETE")
print("=" * 60)

print("\nRows:", len(merged))

print("\nColumns:")
for column in merged.columns:
    print(" -", column)

print("\nMissing values:")
print(merged.isnull().sum())

print("\nUnique dates:", merged["date"].nunique())

print(
    "\nDate range:",
    merged["date"].min().date(),
    "to",
    merged["date"].max().date()
)

print("\nFirst 5 rows:")
print(merged.head().to_string(index=False))

print("\nLast 5 rows:")
print(merged.tail().to_string(index=False))

print("\nSaved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 60)