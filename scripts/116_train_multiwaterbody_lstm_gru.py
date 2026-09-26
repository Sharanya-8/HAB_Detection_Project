# ============================================================
# STEP 116
# MULTI-WATERBODY LSTM + GRU TRAINING
# OBJECTIVE 3 - TEMPORAL HAB FORECASTING
#
# Training waterbodies:
#   1. Hussain Sagar
#   2. Saroor Nagar
#   3. Osman Sagar
#   4. Himayat Sagar
#
# UNSEEN TEST WATERBODY:
#   Shamirpet Lake
#
# INPUT:
#   data/processed/multiwaterbody_temporal_dataset.csv
#
# TARGET:
#   NDCI at the NEXT observed Sentinel-2 date
#
# SEQUENCE:
#   Previous 3 observations -> next NDCI
#
# SPLIT:
#   70% chronological training
#   15% chronological validation
#   15% chronological testing
#
# IMPORTANT:
#   The split is performed separately for each waterbody.
#   No random splitting is used.
#
# ACCURACY:
#   Percentage of held-out predictions whose absolute
#   NDCI error is <= 0.05.
#
# ============================================================


from pathlib import Path
import copy
import pickle
import random

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multiwaterbody_temporal_dataset.csv"
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
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TRAINING WATERBODIES
# ============================================================

TRAINING_WATERBODIES = [
    "Hussain_Sagar",
    "Saroor_Nagar",
    "Osman_Sagar",
    "Himayat_Sagar",
]


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


TARGET_COLUMN = "target_next_ndci"


# ============================================================
# SEQUENCE SETTINGS
# ============================================================

SEQUENCE_LENGTH = 3

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# MODEL SETTINGS
# ============================================================

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.20

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

BATCH_SIZE = 16

MAX_EPOCHS = 100
PATIENCE = 12

MIN_DELTA = 0.00001


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# PRINT HEADER
# ============================================================

print()
print("=" * 80)
print("STEP 116 - MULTI-WATERBODY LSTM + GRU TRAINING")
print("=" * 80)

print()
print(f"Device: {DEVICE}")
print(f"Sequence length: {SEQUENCE_LENGTH}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Maximum epochs: {MAX_EPOCHS}")

print()
print("Training waterbodies:")

for waterbody in TRAINING_WATERBODIES:
    print(f"  - {waterbody}")


# ============================================================
# LOAD DATASET
# ============================================================

if not DATASET_FILE.exists():

    raise FileNotFoundError(
        f"\nTemporal dataset not found:\n{DATASET_FILE}\n\n"
        "Run Step 115 first."
    )


df = pd.read_csv(
    DATASET_FILE
)


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = (
    ["waterbody", "date", TARGET_COLUMN]
    + FEATURE_COLUMNS
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(
            f"  - {column}"
            for column in missing_columns
        )
    )


df["date"] = pd.to_datetime(
    df["date"]
)


# Keep only the four training waterbodies.

df = df[
    df["waterbody"].isin(
        TRAINING_WATERBODIES
    )
].copy()


df = df.sort_values(
    [
        "waterbody",
        "date"
    ]
).reset_index(
    drop=True
)


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

for column in FEATURE_COLUMNS:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


df[TARGET_COLUMN] = pd.to_numeric(
    df[TARGET_COLUMN],
    errors="coerce"
)


df = df.dropna(
    subset=FEATURE_COLUMNS + [TARGET_COLUMN]
).copy()


print()
print(
    f"Usable rows after validation: {len(df)}"
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================
#
# We assign every observation to train/validation/test
# separately for each waterbody.
#
# The target date determines the split.
#
# Example:
#
# 70% -> TRAIN
# 15% -> VALIDATION
# 15% -> TEST
#
# We DO NOT randomly shuffle observations between splits.
#
# ============================================================


df["split"] = None


for waterbody in TRAINING_WATERBODIES:

    water_df = (
        df[
            df["waterbody"] == waterbody
        ]
        .sort_values("date")
    )

    n = len(water_df)

    train_end = int(
        n * TRAIN_RATIO
    )

    val_end = int(
        n * (TRAIN_RATIO + VAL_RATIO)
    )

    indices = water_df.index.tolist()

    for index in indices[:train_end]:

        df.loc[
            index,
            "split"
        ] = "train"

    for index in indices[
        train_end:val_end
    ]:

        df.loc[
            index,
            "split"
        ] = "validation"

    for index in indices[
        val_end:
    ]:

        df.loc[
            index,
            "split"
        ] = "test"


# ============================================================
# DISPLAY SPLIT SUMMARY
# ============================================================

print()
print("=" * 80)
print("CHRONOLOGICAL SPLIT")
print("=" * 80)

for waterbody in TRAINING_WATERBODIES:

    print()
    print(
        waterbody
    )

    subset = df[
        df["waterbody"] == waterbody
    ]

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        split_df = subset[
            subset["split"] == split_name
        ]

        print(
            f"  {split_name:<12}: "
            f"{len(split_df):>3} observations"
        )


# ============================================================
# CREATE SEQUENCES
# ============================================================
#
# Important:
#
# For every target observation we take the previous
# SEQUENCE_LENGTH observations from the SAME waterbody.
#
# Example with sequence length 3:
#
# Observation 1
# Observation 2
# Observation 3
#          ↓
#     Predict Obs 4 NDCI
#
# The target observation determines whether the
# sequence belongs to train, validation or test.
#
# This allows validation/test predictions to use
# historical observations that were genuinely available
# before the target date.
#
# ============================================================


def create_sequences(
    full_dataframe,
    target_split
):

    X = []
    y = []

    metadata = []

    for waterbody in TRAINING_WATERBODIES:

        water_df = (
            full_dataframe[
                full_dataframe["waterbody"]
                == waterbody
            ]
            .sort_values("date")
            .reset_index(drop=True)
        )

        for target_position in range(
            SEQUENCE_LENGTH,
            len(water_df)
        ):

            target_row = (
                water_df.iloc[
                    target_position
                ]
            )

            if (
                target_row["split"]
                != target_split
            ):
                continue

            start_position = (
                target_position
                - SEQUENCE_LENGTH
            )

            sequence_df = (
                water_df.iloc[
                    start_position:
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

            target_value = float(
                target_row[
                    TARGET_COLUMN
                ]
            )

            X.append(
                sequence_values
            )

            y.append(
                target_value
            )

            metadata.append({

                "waterbody":
                    waterbody,

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
            })

    if len(X) == 0:

        raise RuntimeError(
            f"No sequences created for "
            f"{target_split}."
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        pd.DataFrame(metadata)
    )


# ============================================================
# CREATE RAW SEQUENCES
# ============================================================

X_train_raw, y_train, train_meta = (
    create_sequences(
        df,
        "train"
    )
)

X_val_raw, y_val, val_meta = (
    create_sequences(
        df,
        "validation"
    )
)

X_test_raw, y_test, test_meta = (
    create_sequences(
        df,
        "test"
    )
)


print()
print("=" * 80)
print("SEQUENCE DATASET")
print("=" * 80)

print(
    f"Train sequences      : {len(X_train_raw)}"
)

print(
    f"Validation sequences : {len(X_val_raw)}"
)

print(
    f"Test sequences       : {len(X_test_raw)}"
)

print()
print(
    f"Input shape           : "
    f"{X_train_raw.shape}"
)

print(
    f"Target shape          : "
    f"{y_train.shape}"
)


# ============================================================
# FEATURE SCALING
# ============================================================
#
# VERY IMPORTANT:
#
# StandardScaler is fitted ONLY on training observations.
#
# Validation and test data are transformed using the
# training scaler.
#
# This prevents information leakage.
#
# ============================================================


scaler = StandardScaler()


train_2d = X_train_raw.reshape(
    -1,
    len(FEATURE_COLUMNS)
)


scaler.fit(
    train_2d
)


def scale_sequences(
    X,
    fitted_scaler
):

    original_shape = X.shape

    X_2d = X.reshape(
        -1,
        len(FEATURE_COLUMNS)
    )

    X_scaled = fitted_scaler.transform(
        X_2d
    )

    return X_scaled.reshape(
        original_shape
    ).astype(
        np.float32
    )


X_train = scale_sequences(
    X_train_raw,
    scaler
)

X_val = scale_sequences(
    X_val_raw,
    scaler
)

X_test = scale_sequences(
    X_test_raw,
    scaler
)


# ============================================================
# SAVE SCALER
# ============================================================

SCALER_FILE = (
    MODEL_DIR
    / "multiwaterbody_temporal_feature_scaler.pkl"
)

with open(
    SCALER_FILE,
    "wb"
) as file:

    pickle.dump(
        scaler,
        file
    )


# ============================================================
# CONVERT TO PYTORCH TENSORS
# ============================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32
).unsqueeze(1)


X_val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_val_tensor = torch.tensor(
    y_val,
    dtype=torch.float32
).unsqueeze(1)


X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.float32
).unsqueeze(1)


# ============================================================
# DATALOADERS
# ============================================================

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

val_dataset = TensorDataset(
    X_val_tensor,
    y_val_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# LSTM MODEL
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


    def forward(
        self,
        x
    ):

        output, _ = self.lstm(
            x
        )

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
# GRU MODEL
# ============================================================


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


    def forward(
        self,
        x
    ):

        output, _ = self.gru(
            x
        )

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
# TRAINING FUNCTION
# ============================================================


def train_model(
    model,
    model_name
):

    model = model.to(
        DEVICE
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_val_loss = float(
        "inf"
    )

    best_state = None

    patience_counter = 0

    history = []

    print()
    print("=" * 80)
    print(
        f"TRAINING {model_name}"
    )
    print("=" * 80)


    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        train_losses = []

        for X_batch, y_batch in (
            train_loader
        ):

            X_batch = X_batch.to(
                DEVICE
            )

            y_batch = y_batch.to(
                DEVICE
            )

            optimizer.zero_grad()

            predictions = model(
                X_batch
            )

            loss = criterion(
                predictions,
                y_batch
            )

            loss.backward()

            # Prevent exploding gradients.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            train_losses.append(
                loss.item()
            )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        val_losses = []

        with torch.no_grad():

            for X_batch, y_batch in (
                val_loader
            ):

                X_batch = X_batch.to(
                    DEVICE
                )

                y_batch = y_batch.to(
                    DEVICE
                )

                predictions = model(
                    X_batch
                )

                loss = criterion(
                    predictions,
                    y_batch
                )

                val_losses.append(
                    loss.item()
                )


        train_loss = float(
            np.mean(
                train_losses
            )
        )

        val_loss = float(
            np.mean(
                val_losses
            )
        )


        history.append({

            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "validation_loss":
                val_loss,
        })


        print(
            f"Epoch "
            f"{epoch:03d}/{MAX_EPOCHS} | "
            f"Train Loss: "
            f"{train_loss:.6f} | "
            f"Val Loss: "
            f"{val_loss:.6f}"
        )


        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if (
            val_loss
            <
            best_val_loss
            - MIN_DELTA
        ):

            best_val_loss = (
                val_loss
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1


        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            patience_counter
            >= PATIENCE
        ):

            print(
                f"\nEarly stopping at "
                f"epoch {epoch}."
            )

            break


    # --------------------------------------------------------
    # RESTORE BEST MODEL
    # --------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )


    history_df = pd.DataFrame(
        history
    )

    history_file = (
        RESULT_DIR
        / f"{model_name.lower()}_training_history.csv"
    )

    history_df.to_csv(
        history_file,
        index=False
    )


    # --------------------------------------------------------
    # SAVE LOSS GRAPH
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        history_df["epoch"],
        history_df["train_loss"],
        label="Training Loss"
    )

    plt.plot(
        history_df["epoch"],
        history_df["validation_loss"],
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "MSE Loss"
    )

    plt.title(
        f"{model_name} Training History"
    )

    plt.legend()

    plt.tight_layout()

    graph_file = (
        RESULT_DIR
        / f"{model_name.lower()}_training_history.png"
    )

    plt.savefig(
        graph_file,
        dpi=150
    )

    plt.close()


    return (
        model,
        history_df,
        best_val_loss
    )


# ============================================================
# CREATE MODELS
# ============================================================

input_size = len(
    FEATURE_COLUMNS
)


lstm_model = LSTMRegressor(
    input_size=input_size,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)


gru_model = GRURegressor(
    input_size=input_size,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)


# ============================================================
# TRAIN LSTM
# ============================================================

lstm_model, lstm_history, lstm_best_val = (
    train_model(
        lstm_model,
        "LSTM"
    )
)


# ============================================================
# TRAIN GRU
# ============================================================

gru_model, gru_history, gru_best_val = (
    train_model(
        gru_model,
        "GRU"
    )
)


# ============================================================
# SAVE MODEL CHECKPOINTS
# ============================================================

LSTM_MODEL_FILE = (
    MODEL_DIR
    / "multiwaterbody_lstm_best.pth"
)

GRU_MODEL_FILE = (
    MODEL_DIR
    / "multiwaterbody_gru_best.pth"
)


torch.save(
    {
        "model_state_dict":
            lstm_model.state_dict(),

        "input_size":
            input_size,

        "hidden_size":
            HIDDEN_SIZE,

        "num_layers":
            NUM_LAYERS,

        "dropout":
            DROPOUT,

        "sequence_length":
            SEQUENCE_LENGTH,

        "feature_columns":
            FEATURE_COLUMNS,

        "target":
            TARGET_COLUMN,

        "model_type":
            "LSTM",
    },
    LSTM_MODEL_FILE
)


torch.save(
    {
        "model_state_dict":
            gru_model.state_dict(),

        "input_size":
            input_size,

        "hidden_size":
            HIDDEN_SIZE,

        "num_layers":
            NUM_LAYERS,

        "dropout":
            DROPOUT,

        "sequence_length":
            SEQUENCE_LENGTH,

        "feature_columns":
            FEATURE_COLUMNS,

        "target":
            TARGET_COLUMN,

        "model_type":
            "GRU",
    },
    GRU_MODEL_FILE
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================


def predict_model(
    model,
    X_tensor,
    y_true,
    metadata,
    model_name
):

    model.eval()

    predictions = []

    with torch.no_grad():

        for X_batch, _ in DataLoader(
            TensorDataset(
                X_tensor,
                torch.zeros(
                    len(X_tensor),
                    1
                )
            ),
            batch_size=BATCH_SIZE,
            shuffle=False
        ):

            X_batch = X_batch.to(
                DEVICE
            )

            output = model(
                X_batch
            )

            predictions.extend(
                output.cpu()
                .numpy()
                .reshape(-1)
                .tolist()
            )


    predictions = np.asarray(
        predictions,
        dtype=np.float64
    )

    actual = np.asarray(
        y_true,
        dtype=np.float64
    )


    result = metadata.copy()

    result["actual_ndci"] = (
        actual
    )

    result["predicted_ndci"] = (
        predictions
    )

    result["absolute_error"] = (
        np.abs(
            actual
            - predictions
        )
    )

    result["model"] = (
        model_name
    )

    return result


# ============================================================
# GENERATE TEST PREDICTIONS
# ============================================================

lstm_predictions = predict_model(
    lstm_model,
    X_test_tensor,
    y_test,
    test_meta,
    "LSTM"
)


gru_predictions = predict_model(
    gru_model,
    X_test_tensor,
    y_test,
    test_meta,
    "GRU"
)


# ============================================================
# METRICS FUNCTION
# ============================================================


def calculate_metrics(
    actual,
    predicted
):

    actual = np.asarray(
        actual,
        dtype=np.float64
    )

    predicted = np.asarray(
        predicted,
        dtype=np.float64
    )

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


    # --------------------------------------------------------
    # ACCURACY-LIKE METRIC
    # --------------------------------------------------------
    #
    # Percentage of predictions whose absolute error
    # is <= 0.05 NDCI.
    #
    # This is NOT classification accuracy.
    #
    # It is explicitly called:
    #
    # "Within ±0.05 NDCI accuracy"
    #
    # --------------------------------------------------------

    tolerance = 0.05

    within_tolerance = (
        np.abs(
            actual
            - predicted
        )
        <= tolerance
    )

    tolerance_accuracy = (
        100.0
        * np.mean(
            within_tolerance
        )
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
            float(tolerance_accuracy),

        "test_samples":
            int(len(actual)),
    }


# ============================================================
# OVERALL TEST METRICS
# ============================================================

lstm_metrics = calculate_metrics(
    lstm_predictions[
        "actual_ndci"
    ],
    lstm_predictions[
        "predicted_ndci"
    ]
)


gru_metrics = calculate_metrics(
    gru_predictions[
        "actual_ndci"
    ],
    gru_predictions[
        "predicted_ndci"
    ]
)


# ============================================================
# PRINT OVERALL RESULTS
# ============================================================

print()
print("=" * 80)
print("OVERALL TEST PERFORMANCE")
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
# WATERBODY-WISE METRICS
# ============================================================


def calculate_waterbody_metrics(
    predictions_df
):

    records = []

    for waterbody in (
        TRAINING_WATERBODIES
    ):

        subset = predictions_df[
            predictions_df[
                "waterbody"
            ]
            == waterbody
        ]

        if subset.empty:
            continue

        metrics = calculate_metrics(
            subset["actual_ndci"],
            subset["predicted_ndci"]
        )

        records.append({

            "waterbody":
                waterbody,

            **metrics
        })

    return pd.DataFrame(
        records
    )


lstm_waterbody_metrics = (
    calculate_waterbody_metrics(
        lstm_predictions
    )
)

gru_waterbody_metrics = (
    calculate_waterbody_metrics(
        gru_predictions
    )
)


# ============================================================
# SAVE WATERBODY METRICS
# ============================================================

lstm_waterbody_metrics.to_csv(
    RESULT_DIR
    / "lstm_waterbody_test_metrics.csv",
    index=False
)

gru_waterbody_metrics.to_csv(
    RESULT_DIR
    / "gru_waterbody_test_metrics.csv",
    index=False
)


# ============================================================
# SAVE OVERALL METRICS
# ============================================================

metrics_table = pd.DataFrame({

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
    / "multiwaterbody_lstm_gru_metrics.csv"
)


metrics_table.to_csv(
    METRICS_FILE,
    index=False
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

lstm_predictions_file = (
    RESULT_DIR
    / "multiwaterbody_lstm_test_predictions.csv"
)

gru_predictions_file = (
    RESULT_DIR
    / "multiwaterbody_gru_test_predictions.csv"
)


lstm_predictions.to_csv(
    lstm_predictions_file,
    index=False
)

gru_predictions.to_csv(
    gru_predictions_file,
    index=False
)


# ============================================================
# COMBINED PREDICTIONS
# ============================================================

combined_predictions = (
    test_meta.copy()
)


combined_predictions[
    "actual_ndci"
] = y_test


combined_predictions[
    "lstm_predicted_ndci"
] = (
    lstm_predictions[
        "predicted_ndci"
    ].to_numpy()
)


combined_predictions[
    "gru_predicted_ndci"
] = (
    gru_predictions[
        "predicted_ndci"
    ].to_numpy()
)


combined_predictions[
    "lstm_absolute_error"
] = (
    np.abs(
        combined_predictions[
            "actual_ndci"
        ]
        -
        combined_predictions[
            "lstm_predicted_ndci"
        ]
    )
)


combined_predictions[
    "gru_absolute_error"
] = (
    np.abs(
        combined_predictions[
            "actual_ndci"
        ]
        -
        combined_predictions[
            "gru_predicted_ndci"
        ]
    )
)


combined_predictions[
    "lstm_within_0.05"
] = (
    combined_predictions[
        "lstm_absolute_error"
    ]
    <= 0.05
)


combined_predictions[
    "gru_within_0.05"
] = (
    combined_predictions[
        "gru_absolute_error"
    ]
    <= 0.05
)


combined_predictions.to_csv(
    RESULT_DIR
    / "multiwaterbody_lstm_gru_test_predictions.csv",
    index=False
)


# ============================================================
# ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(
    figsize=(12, 6)
)


plot_dates = pd.to_datetime(
    test_meta[
        "target_date"
    ]
)


plt.plot(
    plot_dates,
    y_test,
    marker="o",
    label="Actual NDCI"
)


plt.plot(
    plot_dates,
    lstm_predictions[
        "predicted_ndci"
    ],
    marker="x",
    label="LSTM Predicted NDCI"
)


plt.plot(
    plot_dates,
    gru_predictions[
        "predicted_ndci"
    ],
    marker="s",
    label="GRU Predicted NDCI"
)


plt.axhline(
    y=0.30,
    linestyle="--",
    label="NDCI Alert Threshold"
)


plt.xlabel(
    "Target Date"
)

plt.ylabel(
    "NDCI"
)

plt.title(
    "Multi-Waterbody LSTM and GRU — "
    "Actual vs Predicted NDCI"
)

plt.legend()

plt.xticks(
    rotation=45
)

plt.tight_layout()


plt.savefig(
    RESULT_DIR
    / "multiwaterbody_lstm_gru_actual_vs_predicted.png",
    dpi=150
)

plt.close()


# ============================================================
# METRICS BAR GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)


models = [
    "LSTM",
    "GRU"
]


mae_values = [
    lstm_metrics["MAE"],
    gru_metrics["MAE"]
]


rmse_values = [
    lstm_metrics["RMSE"],
    gru_metrics["RMSE"]
]


x = np.arange(
    len(models)
)


width = 0.35


plt.bar(
    x - width / 2,
    mae_values,
    width,
    label="MAE"
)


plt.bar(
    x + width / 2,
    rmse_values,
    width,
    label="RMSE"
)


plt.xticks(
    x,
    models
)

plt.ylabel(
    "Error"
)

plt.title(
    "LSTM vs GRU Test Error"
)

plt.legend()

plt.tight_layout()


plt.savefig(
    RESULT_DIR
    / "multiwaterbody_lstm_gru_error_comparison.png",
    dpi=150
)

plt.close()


# ============================================================
# SAVE MODEL INFORMATION
# ============================================================

model_information = pd.DataFrame({

    "Parameter": [

        "Training waterbodies",

        "Sequence length",

        "Number of input features",

        "Feature columns",

        "Train ratio",

        "Validation ratio",

        "Test ratio",

        "Hidden size",

        "Number of layers",

        "Dropout",

        "Learning rate",

        "Weight decay",

        "Batch size",

        "Maximum epochs",

        "Accuracy definition",

        "HAB-risk threshold",
    ],

    "Value": [

        ", ".join(
            TRAINING_WATERBODIES
        ),

        SEQUENCE_LENGTH,

        len(
            FEATURE_COLUMNS
        ),

        ", ".join(
            FEATURE_COLUMNS
        ),

        TRAIN_RATIO,

        VAL_RATIO,

        TEST_RATIO,

        HIDDEN_SIZE,

        NUM_LAYERS,

        DROPOUT,

        LEARNING_RATE,

        WEIGHT_DECAY,

        BATCH_SIZE,

        MAX_EPOCHS,

        "Percentage of predictions within ±0.05 NDCI",

        "NDCI >= 0.30",
    ]
})


model_information.to_csv(
    RESULT_DIR
    / "multiwaterbody_temporal_model_information.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 80)
print("STEP 116 FINAL SUMMARY")
print("=" * 80)

print()

print(
    f"LSTM MAE  : "
    f"{lstm_metrics['MAE']:.6f}"
)

print(
    f"LSTM RMSE : "
    f"{lstm_metrics['RMSE']:.6f}"
)

print(
    f"LSTM R²   : "
    f"{lstm_metrics['R2']:.6f}"
)

print(
    f"LSTM ±0.05 accuracy : "
    f"{lstm_metrics['accuracy_within_0.05_ndci_percent']:.2f}%"
)


print()

print(
    f"GRU MAE   : "
    f"{gru_metrics['MAE']:.6f}"
)

print(
    f"GRU RMSE  : "
    f"{gru_metrics['RMSE']:.6f}"
)

print(
    f"GRU R²    : "
    f"{gru_metrics['R2']:.6f}"
)

print(
    f"GRU ±0.05 accuracy : "
    f"{gru_metrics['accuracy_within_0.05_ndci_percent']:.2f}%"
)


print()
print("=" * 80)
print("FILES CREATED")
print("=" * 80)

print()
print("Models:")

print(
    LSTM_MODEL_FILE
)

print(
    GRU_MODEL_FILE
)

print()
print("Scaler:")

print(
    SCALER_FILE
)

print()
print("Metrics:")

print(
    METRICS_FILE
)

print()
print("Predictions:")

print(
    lstm_predictions_file
)

print(
    gru_predictions_file
)

print(
    RESULT_DIR
    / "multiwaterbody_lstm_gru_test_predictions.csv"
)

print()
print("Graphs:")

print(
    RESULT_DIR
    / "multiwaterbody_lstm_gru_actual_vs_predicted.png"
)

print(
    RESULT_DIR
    / "multiwaterbody_lstm_gru_error_comparison.png"
)

print()
print("=" * 80)
print("STEP 116 COMPLETE")
print("=" * 80)