# ============================================================
# STEP 117
# TEST MULTI-WATERBODY LSTM + GRU ON SHAMIRPET LAKE
#
# IMPORTANT:
# Shamirpet was NOT used during training.
# It is an unseen test waterbody.
#
# Training models:
#   models/temporal/multiwaterbody_lstm_best.pth
#   models/temporal/multiwaterbody_gru_best.pth
#
# Shamirpet input:
#   Existing Shamirpet 14-band Sentinel-2 images
#
# Features must match Step 116 exactly:
#   ndci_mean
#   ndci_median
#   ndci_max
#   ndwi_mean
#   ndwi_median
#   mndwi_mean
#   mndwi_median
#   fai_mean
#   fai_median
#   fai_max
#
# Sequence:
#   Previous 3 observations -> next NDCI
#
# Metrics:
#   MSE
#   MAE
#   RMSE
#   R2
#   Accuracy within +/- 0.05 NDCI
#
# ============================================================

from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sentinel2"
    / "shamirpet_test"
)

LABEL_ROOT = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "hab_masks"
    / "shamirpet_test"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "temporal"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "temporal"
    / "shamirpet_test"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


LSTM_MODEL_FILE = (
    MODEL_DIR
    / "multiwaterbody_lstm_best.pth"
)

GRU_MODEL_FILE = (
    MODEL_DIR
    / "multiwaterbody_gru_best.pth"
)

SCALER_FILE = (
    MODEL_DIR
    / "multiwaterbody_temporal_feature_scaler.pkl"
)


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
]


SEQUENCE_LENGTH = 3

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 80)
print("STEP 117 - SHAMIRPET UNSEEN WATERBODY TEMPORAL TEST")
print("=" * 80)

print()
print(f"Device: {DEVICE}")

print()
print("IMPORTANT:")
print("Shamirpet was NOT used during LSTM/GRU training.")
print("This is an unseen-waterbody generalization test.")


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    LSTM_MODEL_FILE,
    GRU_MODEL_FILE,
    SCALER_FILE,
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )


if not IMAGE_ROOT.exists():

    raise FileNotFoundError(
        f"\nShamirpet image directory not found:\n"
        f"{IMAGE_ROOT}"
    )


# ============================================================
# DATE EXTRACTION
# ============================================================

def extract_date(filename):

    match = re.search(
        r"(20\d{2})[-_](\d{2})[-_](\d{2})",
        filename
    )

    if not match:
        return None

    return pd.Timestamp(
        year=int(match.group(1)),
        month=int(match.group(2)),
        day=int(match.group(3))
    )


# ============================================================
# FIND LABEL
# ============================================================

def find_label(date):

    candidates = [

        LABEL_ROOT
        / f"{date:%Y-%m-%d}_HAB.tif",

        LABEL_ROOT
        / f"{date:%Y_%m_%d}_HAB.tif",

        LABEL_ROOT
        / f"Shamirpet_Lake_{date:%Y-%m-%d}_HAB.tif",

        LABEL_ROOT
        / f"Shamirpet_Lake_{date:%Y_%m_%d}_HAB.tif",
    ]

    for path in candidates:

        if path.exists():
            return path


    if LABEL_ROOT.exists():

        date_text_1 = (
            date.strftime("%Y-%m-%d")
        )

        date_text_2 = (
            date.strftime("%Y_%m_%d")
        )

        for path in LABEL_ROOT.glob(
            "*.tif"
        ):

            if (
                date_text_1 in path.name
                or date_text_2 in path.name
            ):

                return path


    return None


# ============================================================
# SAFE STATISTICS
# ============================================================

def safe_stats(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:

        return (
            np.nan,
            np.nan,
            np.nan
        )

    return (
        float(values.mean()),
        float(np.median(values)),
        float(values.max())
    )


# ============================================================
# PROCESS SHAMIRPET IMAGE
# ============================================================

def process_image(
    image_path,
    label_path
):

    with rasterio.open(
        image_path
    ) as src:

        image = src.read().astype(
            np.float32
        )


    with rasterio.open(
        label_path
    ) as src:

        label = src.read(1)


    if image.shape[0] != 14:

        raise ValueError(
            f"Expected 14 bands, "
            f"found {image.shape[0]}"
        )


    if label.shape != image.shape[1:]:

        raise ValueError(
            "Image/label dimensions do not match."
        )


    # --------------------------------------------------------
    # 255 = invalid / outside water
    # --------------------------------------------------------

    valid = (
        (label != 255)
        & np.isfinite(image[10])
        & np.isfinite(image[11])
        & np.isfinite(image[12])
        & np.isfinite(image[13])
    )


    if not valid.any():

        raise ValueError(
            "No valid water pixels."
        )


    # --------------------------------------------------------
    # INDICES
    # --------------------------------------------------------

    ndwi = image[10][valid]

    mndwi = image[11][valid]

    ndci = image[12][valid]

    fai = image[13][valid]


    ndwi_mean, ndwi_median, _ = (
        safe_stats(ndwi)
    )

    mndwi_mean, mndwi_median, _ = (
        safe_stats(mndwi)
    )

    ndci_mean, ndci_median, ndci_max = (
        safe_stats(ndci)
    )

    fai_mean, fai_median, fai_max = (
        safe_stats(fai)
    )


    water_pixels = int(
        valid.sum()
    )

    hab_pixels = int(
        (label[valid] == 1).sum()
    )


    return {

        "ndci_mean":
            ndci_mean,

        "ndci_median":
            ndci_median,

        "ndci_max":
            ndci_max,

        "ndwi_mean":
            ndwi_mean,

        "ndwi_median":
            ndwi_median,

        "mndwi_mean":
            mndwi_mean,

        "mndwi_median":
            mndwi_median,

        "fai_mean":
            fai_mean,

        "fai_median":
            fai_median,

        "fai_max":
            fai_max,

        "valid_water_pixels":
            water_pixels,

        "hab_pixels":
            hab_pixels,

        "hab_percentage":
            (
                100.0
                * hab_pixels
                / water_pixels
            ),
    }


# ============================================================
# COLLECT SHAMIRPET TEMPORAL DATA
# ============================================================

print()
print("=" * 80)
print("BUILDING SHAMIRPET TEMPORAL DATASET")
print("=" * 80)


records = []

skipped = []


image_files = sorted(
    IMAGE_ROOT.glob(
        "*.tif"
    )
)


print()
print(
    f"Images found: "
    f"{len(image_files)}"
)


for image_path in image_files:

    date = extract_date(
        image_path.name
    )


    if date is None:

        skipped.append({
            "image":
                str(image_path),
            "reason":
                "Could not extract date"
        })

        continue


    label_path = find_label(
        date
    )


    if label_path is None:

        skipped.append({
            "image":
                str(image_path),
            "date":
                str(date.date()),
            "reason":
                "HAB label not found"
        })

        continue


    try:

        stats = process_image(
            image_path,
            label_path
        )


        records.append({

            "waterbody":
                "Shamirpet_Lake",

            "date":
                date,

            **stats,

            "image_path":
                str(image_path),

            "label_path":
                str(label_path),
        })


    except Exception as exc:

        skipped.append({

            "image":
                str(image_path),

            "date":
                str(date.date()),

            "reason":
                str(exc)
        })


if len(records) == 0:

    raise RuntimeError(
        "No Shamirpet temporal records were created."
    )


shamirpet_df = pd.DataFrame(
    records
)


shamirpet_df = (
    shamirpet_df
    .sort_values("date")
    .reset_index(drop=True)
)


# ============================================================
# REMOVE INVALID FEATURE ROWS
# ============================================================

for column in FEATURE_COLUMNS:

    shamirpet_df[column] = pd.to_numeric(
        shamirpet_df[column],
        errors="coerce"
    )


shamirpet_df = (
    shamirpet_df
    .dropna(
        subset=FEATURE_COLUMNS
    )
    .reset_index(drop=True)
)


print()
print(
    f"Valid Shamirpet observations: "
    f"{len(shamirpet_df)}"
)


print()
print(
    f"Date range: "
    f"{shamirpet_df['date'].min().date()} "
    f"to "
    f"{shamirpet_df['date'].max().date()}"
)


# ============================================================
# CREATE NEXT-NDCI TARGET
# ============================================================

shamirpet_df[
    "target_next_ndci"
] = (
    shamirpet_df[
        "ndci_mean"
    ].shift(-1)
)


shamirpet_df[
    "next_date"
] = (
    shamirpet_df[
        "date"
    ].shift(-1)
)


shamirpet_df[
    "days_to_next_observation"
] = (
    shamirpet_df[
        "next_date"
    ]
    - shamirpet_df[
        "date"
    ]
).dt.days


# Last observation has no future target.

shamirpet_df = (
    shamirpet_df
    .dropna(
        subset=["target_next_ndci"]
    )
    .reset_index(drop=True)
)


# ============================================================
# LOAD TRAINING SCALER
# ============================================================

print()
print("Loading training feature scaler...")


with open(
    SCALER_FILE,
    "rb"
) as file:

    scaler = pickle.load(
        file
    )


# ============================================================
# CREATE SEQUENCES
# ============================================================

X_raw = []

y = []

metadata = []


for target_position in range(
    SEQUENCE_LENGTH,
    len(shamirpet_df)
):

    sequence_df = (
        shamirpet_df.iloc[
            target_position
            - SEQUENCE_LENGTH:
            target_position
        ]
    )


    target_row = (
        shamirpet_df.iloc[
            target_position
        ]
    )


    sequence_values = (
        sequence_df[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=np.float32
        )
    )


    X_raw.append(
        sequence_values
    )


    y.append(
        float(
            target_row[
                "target_next_ndci"
            ]
        )
    )


    metadata.append({

        "waterbody":
            "Shamirpet_Lake",

        "target_date":
            target_row["date"],

        "previous_start_date":
            sequence_df[
                "date"
            ].iloc[0],

        "previous_end_date":
            sequence_df[
                "date"
            ].iloc[-1],

        "days_to_target":
            target_row[
                "days_to_next_observation"
            ],
    })


X_raw = np.asarray(
    X_raw,
    dtype=np.float32
)


y = np.asarray(
    y,
    dtype=np.float32
)


metadata = pd.DataFrame(
    metadata
)


print()
print(
    f"Shamirpet test sequences: "
    f"{len(X_raw)}"
)


print(
    f"Sequence shape: "
    f"{X_raw.shape}"
)


# ============================================================
# APPLY TRAINING SCALER
# ============================================================

original_shape = X_raw.shape


X_2d = X_raw.reshape(
    -1,
    len(FEATURE_COLUMNS)
)


X_scaled = scaler.transform(
    X_2d
)


X_scaled = X_scaled.reshape(
    original_shape
).astype(
    np.float32
)


X_tensor = torch.tensor(
    X_scaled,
    dtype=torch.float32
)


# ============================================================
# MODEL DEFINITIONS
# ============================================================

class LSTMRegressor(
    nn.Module
):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            )
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.fc = nn.Linear(
            hidden_size,
            1
        )


    def forward(self, x):

        output, _ = self.lstm(x)

        last_output = (
            output[:, -1, :]
        )

        last_output = (
            self.dropout(
                last_output
            )
        )

        return self.fc(
            last_output
        )


class GRURegressor(
    nn.Module
):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            )
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.fc = nn.Linear(
            hidden_size,
            1
        )


    def forward(self, x):

        output, _ = self.gru(x)

        last_output = (
            output[:, -1, :]
        )

        last_output = (
            self.dropout(
                last_output
            )
        )

        return self.fc(
            last_output
        )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_model(
    model_file,
    model_class
):

    checkpoint = torch.load(
        model_file,
        map_location=DEVICE
    )


    model = model_class(
        input_size=checkpoint[
            "input_size"
        ],
        hidden_size=checkpoint[
            "hidden_size"
        ],
        num_layers=checkpoint[
            "num_layers"
        ],
        dropout=checkpoint[
            "dropout"
        ]
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    model = model.to(
        DEVICE
    )

    model.eval()

    return model


# ============================================================
# LOAD BOTH MODELS
# ============================================================

print()
print("=" * 80)
print("LOADING TRAINED MODELS")
print("=" * 80)


lstm_model = load_model(
    LSTM_MODEL_FILE,
    LSTMRegressor
)


gru_model = load_model(
    GRU_MODEL_FILE,
    GRURegressor
)


# ============================================================
# PREDICTIONS
# ============================================================

def predict(
    model,
    X
):

    model.eval()

    with torch.no_grad():

        predictions = (
            model(
                X.to(DEVICE)
            )
            .cpu()
            .numpy()
            .reshape(-1)
        )

    return predictions


lstm_pred = predict(
    lstm_model,
    X_tensor
)


gru_pred = predict(
    gru_model,
    X_tensor
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual,
    predicted
):

    mse = mean_squared_error(
        actual,
        predicted
    )

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mse
    )


    try:

        r2 = r2_score(
            actual,
            predicted
        )

    except Exception:

        r2 = np.nan


    # Percentage within +/- 0.05 NDCI.

    accuracy = (
        np.mean(
            np.abs(
                actual
                - predicted
            )
            <= 0.05
        )
        * 100.0
    )


    return {

        "MSE":
            float(mse),

        "MAE":
            float(mae),

        "RMSE":
            float(rmse),

        "R2":
            float(r2),

        "accuracy_within_0.05_ndci_percent":
            float(accuracy),

        "test_samples":
            int(len(actual)),
    }


lstm_metrics = calculate_metrics(
    y,
    lstm_pred
)


gru_metrics = calculate_metrics(
    y,
    gru_pred
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 80)
print("SHAMIRPET UNSEEN-WATERBODY TEST RESULTS")
print("=" * 80)


print()
print("LSTM")

for key, value in lstm_metrics.items():

    if isinstance(
        value,
        float
    ):

        print(
            f"  {key}: "
            f"{value:.6f}"
        )

    else:

        print(
            f"  {key}: "
            f"{value}"
        )


print()
print("GRU")

for key, value in gru_metrics.items():

    if isinstance(
        value,
        float
    ):

        print(
            f"  {key}: "
            f"{value:.6f}"
        )

    else:

        print(
            f"  {key}: "
            f"{value}"
        )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

results = metadata.copy()


results[
    "actual_ndci"
] = y


results[
    "lstm_predicted_ndci"
] = lstm_pred


results[
    "gru_predicted_ndci"
] = gru_pred


results[
    "lstm_absolute_error"
] = np.abs(
    y
    - lstm_pred
)


results[
    "gru_absolute_error"
] = np.abs(
    y
    - gru_pred
)


results[
    "lstm_accuracy_within_0.05"
] = (
    results[
        "lstm_absolute_error"
    ]
    <= 0.05
)


results[
    "gru_accuracy_within_0.05"
] = (
    results[
        "gru_absolute_error"
    ]
    <= 0.05
)


results[
    "lstm_hab_risk"
] = np.where(
    results[
        "lstm_predicted_ndci"
    ] >= 0.30,
    "Higher HAB-risk indicator",
    "Lower HAB-risk indicator"
)


results[
    "gru_hab_risk"
] = np.where(
    results[
        "gru_predicted_ndci"
    ] >= 0.30,
    "Higher HAB-risk indicator",
    "Lower HAB-risk indicator"
)


PREDICTIONS_FILE = (
    RESULT_DIR
    / "shamirpet_lstm_gru_predictions.csv"
)


results.to_csv(
    PREDICTIONS_FILE,
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({

    "Model": [
        "LSTM",
        "GRU"
    ],

    "MSE": [
        lstm_metrics["MSE"],
        gru_metrics["MSE"]
    ],

    "MAE": [
        lstm_metrics["MAE"],
        gru_metrics["MAE"]
    ],

    "RMSE": [
        lstm_metrics["RMSE"],
        gru_metrics["RMSE"]
    ],

    "R2": [
        lstm_metrics["R2"],
        gru_metrics["R2"]
    ],

    "Accuracy_within_0.05_NDCI_percent": [

        lstm_metrics[
            "accuracy_within_0.05_ndci_percent"
        ],

        gru_metrics[
            "accuracy_within_0.05_ndci_percent"
        ]
    ],

    "Test_samples": [

        lstm_metrics[
            "test_samples"
        ],

        gru_metrics[
            "test_samples"
        ]
    ]
})


METRICS_FILE = (
    RESULT_DIR
    / "shamirpet_lstm_gru_metrics.csv"
)


metrics_df.to_csv(
    METRICS_FILE,
    index=False
)


# ============================================================
# ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(
    figsize=(13, 6)
)


dates = pd.to_datetime(
    results[
        "target_date"
    ]
)


plt.plot(
    dates,
    results[
        "actual_ndci"
    ],
    marker="o",
    label="Actual NDCI"
)


plt.plot(
    dates,
    results[
        "lstm_predicted_ndci"
    ],
    marker="x",
    label="LSTM Predicted NDCI"
)


plt.plot(
    dates,
    results[
        "gru_predicted_ndci"
    ],
    marker="s",
    label="GRU Predicted NDCI"
)


plt.axhline(
    y=0.30,
    linestyle="--",
    label="NDCI Risk Threshold"
)


plt.xlabel(
    "Target Date"
)

plt.ylabel(
    "NDCI"
)

plt.title(
    "Shamirpet Lake — "
    "Actual vs LSTM/GRU Predicted NDCI"
)

plt.legend()

plt.xticks(
    rotation=45
)

plt.tight_layout()


GRAPH_FILE = (
    RESULT_DIR
    / "shamirpet_lstm_gru_actual_vs_predicted.png"
)


plt.savefig(
    GRAPH_FILE,
    dpi=150
)

plt.close()


# ============================================================
# ERROR GRAPH
# ============================================================

plt.figure(
    figsize=(13, 5)
)


plt.plot(
    dates,
    results[
        "lstm_absolute_error"
    ],
    marker="x",
    label="LSTM Absolute Error"
)


plt.plot(
    dates,
    results[
        "gru_absolute_error"
    ],
    marker="s",
    label="GRU Absolute Error"
)


plt.axhline(
    y=0.05,
    linestyle="--",
    label="±0.05 Accuracy Boundary"
)


plt.xlabel(
    "Target Date"
)

plt.ylabel(
    "Absolute NDCI Error"
)

plt.title(
    "Shamirpet Lake — "
    "LSTM vs GRU Prediction Error"
)

plt.legend()

plt.xticks(
    rotation=45
)

plt.tight_layout()


ERROR_GRAPH_FILE = (
    RESULT_DIR
    / "shamirpet_lstm_gru_error.png"
)


plt.savefig(
    ERROR_GRAPH_FILE,
    dpi=150
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 80)
print("STEP 117 FINAL SUMMARY")
print("=" * 80)


print()
print(
    f"Shamirpet observations used: "
    f"{len(shamirpet_df)}"
)


print(
    f"Shamirpet test sequences: "
    f"{len(results)}"
)


print()
print(
    f"LSTM MAE: "
    f"{lstm_metrics['MAE']:.6f}"
)

print(
    f"LSTM RMSE: "
    f"{lstm_metrics['RMSE']:.6f}"
)

print(
    f"LSTM R²: "
    f"{lstm_metrics['R2']:.6f}"
)

print(
    f"LSTM ±0.05 accuracy: "
    f"{lstm_metrics['accuracy_within_0.05_ndci_percent']:.2f}%"
)


print()
print(
    f"GRU MAE: "
    f"{gru_metrics['MAE']:.6f}"
)

print(
    f"GRU RMSE: "
    f"{gru_metrics['RMSE']:.6f}"
)

print(
    f"GRU R²: "
    f"{gru_metrics['R2']:.6f}"
)

print(
    f"GRU ±0.05 accuracy: "
    f"{gru_metrics['accuracy_within_0.05_ndci_percent']:.2f}%"
)


print()
print("Saved prediction file:")

print(
    PREDICTIONS_FILE
)


print()
print("Saved metrics:")

print(
    METRICS_FILE
)


print()
print("Saved graph:")

print(
    GRAPH_FILE
)


print()
print("Saved error graph:")

print(
    ERROR_GRAPH_FILE
)


print()
print("=" * 80)
print("STEP 117 COMPLETE")
print("=" * 80)