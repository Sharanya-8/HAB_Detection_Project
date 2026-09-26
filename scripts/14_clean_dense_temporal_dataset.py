from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = (
    DATA_DIR /
    "Hussain_Sagar_Complete_175_Temporal_Dataset.csv"
)

OUTPUT_FILE = (
    DATA_DIR /
    "Hussain_Sagar_Clean_Temporal_Dataset.csv"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])


print("=" * 60)
print("CLEANING TEMPORAL DATASET")
print("=" * 60)

print("\nOriginal observations:", len(df))


# ============================================================
# IDENTIFY SATELLITE FEATURE COLUMNS
# ============================================================

satellite_columns = [
    "fai_max",
    "fai_mean",
    "fai_median",
    "mndwi_mean",
    "mndwi_median",
    "ndci_max",
    "ndci_mean",
    "ndci_median",
    "ndwi_mean",
    "ndwi_median"
]


# ============================================================
# FIND INVALID SATELLITE OBSERVATIONS
# ============================================================

invalid_rows = df[
    df[satellite_columns].isnull().any(axis=1)
]

print("\nInvalid satellite observations:")
print(
    invalid_rows[
        ["date"] + satellite_columns
    ].to_string(index=False)
)


# ============================================================
# REMOVE INVALID OBSERVATIONS
# ============================================================

df_clean = df.dropna(
    subset=satellite_columns
).copy()


# ============================================================
# REMOVE GEE EXPORT COLUMNS
# ============================================================

df_clean = df_clean.drop(
    columns=["system:index", ".geo"],
    errors="ignore"
)


# ============================================================
# SORT BY DATE
# ============================================================

df_clean = (
    df_clean
    .sort_values("date")
    .reset_index(drop=True)
)


# ============================================================
# SAVE
# ============================================================

df_clean.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# VERIFICATION
# ============================================================

print("\n" + "=" * 60)
print("CLEAN DATASET")
print("=" * 60)

print("\nClean observations:", len(df_clean))

print(
    "Unique dates:",
    df_clean["date"].nunique()
)

print(
    "\nDate range:",
    df_clean["date"].min().date(),
    "to",
    df_clean["date"].max().date()
)

print("\nMissing values:")
print(df_clean.isnull().sum())

print("\nColumns:")
for column in df_clean.columns:
    print(" -", column)

print("\nFirst 5 observations:")
print(
    df_clean.head()
    .to_string(index=False)
)

print("\nLast 5 observations:")
print(
    df_clean.tail()
    .to_string(index=False)
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 60)