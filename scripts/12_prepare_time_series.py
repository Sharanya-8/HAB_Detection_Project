from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "Hussain_Sagar_complete_temporal_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "Hussain_Sagar_model_features.csv"
)


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

# We use previous observations to predict the next observation.
SEQUENCE_LENGTH = 3

# Features available before the target HAB percentage.
FEATURE_COLUMNS = [
    "ndwi_mean",
    "ndwi_median",
    "mndwi_mean",
    "mndwi_median",
    "ndci_mean",
    "ndci_median",
    "ndci_max",
    "fai_mean",
    "fai_median",
    "fai_max",
    "temperature_c",
    "rainfall_mm",
    "wind_speed_ms",
]

TARGET_COLUMN = "hab_percentage"


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("=" * 60)
print("OBJECTIVE 3 - TIME SERIES PREPARATION")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

print(f"Total observations: {len(df)}")


# ---------------------------------------------------------
# CHECK DATES
# ---------------------------------------------------------

print("\nObservation dates and gaps:")

for i in range(1, len(df)):
    gap = (
        df.loc[i, "date"]
        - df.loc[i - 1, "date"]
    ).days

    print(
        f"{df.loc[i - 1, 'date'].date()} -> "
        f"{df.loc[i, 'date'].date()} : "
        f"{gap} days"
    )


# ---------------------------------------------------------
# SELECT FEATURES
# ---------------------------------------------------------

X = df[FEATURE_COLUMNS].copy()

y = df[TARGET_COLUMN].copy()


# ---------------------------------------------------------
# CHECK MISSING VALUES
# ---------------------------------------------------------

if X.isnull().any().any():
    raise ValueError(
        "Missing values found in feature columns."
    )

if y.isnull().any():
    raise ValueError(
        "Missing values found in target column."
    )


# ---------------------------------------------------------
# SCALE FEATURES
# ---------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = X_scaled.astype(np.float32)


# ---------------------------------------------------------
# CREATE SEQUENCES
# ---------------------------------------------------------

sequences = []
targets = []
target_dates = []

for i in range(
    SEQUENCE_LENGTH,
    len(df)
):

    sequence = X_scaled[
        i - SEQUENCE_LENGTH:i
    ]

    target = y.iloc[i]

    sequences.append(sequence)
    targets.append(float(target))
    target_dates.append(
        df.loc[i, "date"]
    )


# ---------------------------------------------------------
# CONVERT TO NUMPY
# ---------------------------------------------------------

X_sequences = np.array(
    sequences,
    dtype=np.float32
)

y_targets = np.array(
    targets,
    dtype=np.float32
)


# ---------------------------------------------------------
# SAVE FEATURE DATASET
# ---------------------------------------------------------

model_df = df[
    ["date"] + FEATURE_COLUMNS + [TARGET_COLUMN]
].copy()

model_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# DISPLAY INFORMATION
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("SEQUENCE DATASET CREATED")
print("=" * 60)

print(
    f"Sequence length: "
    f"{SEQUENCE_LENGTH} observations"
)

print(
    f"Number of sequences: "
    f"{len(X_sequences)}"
)

print(
    f"Input shape: "
    f"{X_sequences.shape}"
)

print(
    f"Target shape: "
    f"{y_targets.shape}"
)

print(
    f"Feature count: "
    f"{len(FEATURE_COLUMNS)}"
)

print(
    f"\nFeature file saved to:\n"
    f"{OUTPUT_FILE}"
)


# ---------------------------------------------------------
# SHOW EXAMPLE
# ---------------------------------------------------------

print("\nExample sequence:")

print(
    "Input dates:"
)

for date in df["date"].iloc[
    :SEQUENCE_LENGTH
]:
    print(
        f"  {date.date()}"
    )

print(
    f"\nFirst target date: "
    f"{target_dates[0].date()}"
)

print(
    f"First target HAB percentage: "
    f"{y_targets[0]:.2f}%"
)

print("\nSaved successfully.")