from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

SATELLITE_FILE = (
    PROCESSED_DIR / "Hussain_Sagar_temporal_features.csv"
)

ENVIRONMENTAL_FILE = (
    PROCESSED_DIR / "Hussain_Sagar_Environmental_Data.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR / "Hussain_Sagar_complete_temporal_dataset.csv"
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("=" * 60)
print("OBJECTIVE 3 - MERGING TEMPORAL DATA")
print("=" * 60)

satellite_df = pd.read_csv(SATELLITE_FILE)
environmental_df = pd.read_csv(ENVIRONMENTAL_FILE)

print(f"Satellite observations: {len(satellite_df)}")
print(f"Environmental observations: {len(environmental_df)}")


# ---------------------------------------------------------
# CLEAN ENVIRONMENTAL DATA
# ---------------------------------------------------------

environmental_df = environmental_df[
    [
        "date",
        "temperature_c",
        "rainfall_mm",
        "wind_speed_ms"
    ]
].copy()


# ---------------------------------------------------------
# CONVERT DATES
# ---------------------------------------------------------

satellite_df["date"] = pd.to_datetime(
    satellite_df["date"]
)

environmental_df["date"] = pd.to_datetime(
    environmental_df["date"]
)


# ---------------------------------------------------------
# MERGE USING DATE
# ---------------------------------------------------------

merged_df = pd.merge(
    satellite_df,
    environmental_df,
    on="date",
    how="inner"
)


# ---------------------------------------------------------
# SORT CHRONOLOGICALLY
# ---------------------------------------------------------

merged_df = merged_df.sort_values(
    "date"
).reset_index(drop=True)


# ---------------------------------------------------------
# CHECK FOR MISSING VALUES
# ---------------------------------------------------------

print("\nMissing values:")
print(merged_df.isnull().sum())


# ---------------------------------------------------------
# CHECK NUMBER OF OBSERVATIONS
# ---------------------------------------------------------

print("\nMerged observations:")
print(len(merged_df))


if len(merged_df) != len(satellite_df):
    raise ValueError(
        "Some satellite dates did not match environmental dates."
    )


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

merged_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("COMPLETE TEMPORAL DATASET CREATED")
print("=" * 60)

print(f"Output file:")
print(OUTPUT_FILE)

print("\nColumns:")
print(merged_df.columns.tolist())

print("\nComplete dataset:")
print(merged_df.to_string(index=False))

print("\nSaved successfully.")