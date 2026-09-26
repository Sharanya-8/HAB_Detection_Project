from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = (
    DATA_DIR /
    "Hussain_Sagar_Clean_Temporal_Dataset.csv"
)

OUTPUT_DIR = DATA_DIR / "forecasting"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 3

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "ndci_mean",
    "ndci_median",
    "ndci_max",
    "ndwi_mean",
    "ndwi_median",
    "mndwi_mean",
    "mndwi_median",
    "fai_mean",
    "fai_median",
    "fai_max",
    "temperature_c",
    "rainfall_mm",
    "wind_speed_ms"
]

TARGET_COLUMN = "ndci_mean"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("PREPARING LSTM / GRU FORECASTING SEQUENCES")
print("=" * 65)

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)


print("\nTotal observations:", len(df))


# ============================================================
# CHECK DATA
# ============================================================

missing = df[
    FEATURE_COLUMNS
].isnull().sum()

print("\nMissing values:")
print(missing)


if missing.sum() > 0:
    raise ValueError(
        "Missing values detected in forecasting features."
    )


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

n = len(df)

train_end = int(
    n * TRAIN_RATIO
)

val_end = int(
    n * (TRAIN_RATIO + VAL_RATIO)
)

train_df = df.iloc[
    :train_end
].copy()

val_df = df.iloc[
    train_end:val_end
].copy()

test_df = df.iloc[
    val_end:
].copy()


print("\nChronological split:")
print(
    "Train:",
    len(train_df),
    "|",
    train_df["date"].min().date(),
    "to",
    train_df["date"].max().date()
)

print(
    "Validation:",
    len(val_df),
    "|",
    val_df["date"].min().date(),
    "to",
    val_df["date"].max().date()
)

print(
    "Test:",
    len(test_df),
    "|",
    test_df["date"].min().date(),
    "to",
    test_df["date"].max().date()
)


# ============================================================
# NORMALIZATION
# ============================================================

feature_scaler = StandardScaler()

target_scaler = StandardScaler()


# Fit ONLY on training observations

feature_scaler.fit(
    train_df[FEATURE_COLUMNS]
)

target_scaler.fit(
    train_df[[TARGET_COLUMN]]
)


# Transform all splits

train_features = feature_scaler.transform(
    train_df[FEATURE_COLUMNS]
)

val_features = feature_scaler.transform(
    val_df[FEATURE_COLUMNS]
)

test_features = feature_scaler.transform(
    test_df[FEATURE_COLUMNS]
)


train_target = target_scaler.transform(
    train_df[[TARGET_COLUMN]]
).flatten()

val_target = target_scaler.transform(
    val_df[[TARGET_COLUMN]]
).flatten()

test_target = target_scaler.transform(
    test_df[[TARGET_COLUMN]]
).flatten()


# ============================================================
# SEQUENCE CREATION
# ============================================================

def create_sequences(
    features,
    targets,
    dates,
    sequence_length
):

    X = []
    y = []
    target_dates = []

    for i in range(
        sequence_length,
        len(features)
    ):

        X.append(
            features[
                i-sequence_length:i
            ]
        )

        y.append(
            targets[i]
        )

        target_dates.append(
            dates.iloc[i]
        )

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        target_dates
    )


# ============================================================
# CREATE SEQUENCES
# ============================================================

X_train, y_train, dates_train = create_sequences(
    train_features,
    train_target,
    train_df["date"],
    SEQUENCE_LENGTH
)

X_val, y_val, dates_val = create_sequences(
    val_features,
    val_target,
    val_df["date"],
    SEQUENCE_LENGTH
)

X_test, y_test, dates_test = create_sequences(
    test_features,
    test_target,
    test_df["date"],
    SEQUENCE_LENGTH
)


# ============================================================
# SAVE NUMPY DATA
# ============================================================

np.save(
    OUTPUT_DIR / "X_train.npy",
    X_train
)

np.save(
    OUTPUT_DIR / "y_train.npy",
    y_train
)

np.save(
    OUTPUT_DIR / "X_val.npy",
    X_val
)

np.save(
    OUTPUT_DIR / "y_val.npy",
    y_val
)

np.save(
    OUTPUT_DIR / "X_test.npy",
    X_test
)

np.save(
    OUTPUT_DIR / "y_test.npy",
    y_test
)


# ============================================================
# SAVE DATES
# ============================================================

pd.DataFrame({
    "date": dates_train
}).to_csv(
    OUTPUT_DIR / "train_dates.csv",
    index=False
)

pd.DataFrame({
    "date": dates_val
}).to_csv(
    OUTPUT_DIR / "val_dates.csv",
    index=False
)

pd.DataFrame({
    "date": dates_test
}).to_csv(
    OUTPUT_DIR / "test_dates.csv",
    index=False
)


# ============================================================
# SAVE SCALERS
# ============================================================

import joblib

joblib.dump(
    feature_scaler,
    OUTPUT_DIR / "feature_scaler.pkl"
)

joblib.dump(
    target_scaler,
    OUTPUT_DIR / "target_scaler.pkl"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("SEQUENCE PREPARATION COMPLETE")
print("=" * 65)

print("\nSequence length:", SEQUENCE_LENGTH)

print("\nNumber of features:", len(FEATURE_COLUMNS))

print("\nFeatures:")
for feature in FEATURE_COLUMNS:
    print(" -", feature)

print("\nTarget:", TARGET_COLUMN)

print("\nSequence shapes:")

print(
    "X_train:",
    X_train.shape,
    "y_train:",
    y_train.shape
)

print(
    "X_val:",
    X_val.shape,
    "y_val:",
    y_val.shape
)

print(
    "X_test:",
    X_test.shape,
    "y_test:",
    y_test.shape
)

print("\nFirst training target date:")
print(dates_train[0])

print("\nLast training target date:")
print(dates_train[-1])

print("\nFirst validation target date:")
print(dates_val[0])

print("\nFirst test target date:")
print(dates_test[0])

print("\nSaved forecasting files to:")
print(OUTPUT_DIR)

print("\n" + "=" * 65)