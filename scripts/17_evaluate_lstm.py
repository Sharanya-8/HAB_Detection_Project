from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = (
    BASE_DIR /
    "data" /
    "processed" /
    "forecasting"
)

MODEL_FILE = (
    BASE_DIR /
    "models" /
    "lstm" /
    "best_lstm_model.pth"
)

OUTPUT_DIR = (
    BASE_DIR /
    "results" /
    "lstm"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

INPUT_SIZE = 13
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

DEVICE = torch.device("cpu")


# ============================================================
# LSTM MODEL
# ============================================================

class LSTMForecaster(nn.Module):

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
            dropout=dropout
        )

        self.fc = nn.Sequential(
            nn.Linear(
                hidden_size,
                32
            ),
            nn.ReLU(),
            nn.Linear(
                32,
                1
            )
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        return self.fc(last_output)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("=" * 65)
print("EVALUATING LSTM NDCI FORECAST")
print("=" * 65)

X_test = np.load(
    DATA_DIR / "X_test.npy"
)

y_test = np.load(
    DATA_DIR / "y_test.npy"
)

test_dates = pd.read_csv(
    DATA_DIR / "test_dates.csv"
)

test_dates["date"] = pd.to_datetime(
    test_dates["date"]
)


print("\nTest input shape:", X_test.shape)
print("Test target shape:", y_test.shape)


# ============================================================
# LOAD TARGET SCALER
# ============================================================

target_scaler = joblib.load(
    DATA_DIR / "target_scaler.pkl"
)


# ============================================================
# CREATE MODEL
# ============================================================

model = LSTMForecaster(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(DEVICE)


model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=DEVICE
    )
)

model.eval()


# ============================================================
# PREDICTIONS
# ============================================================

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
).to(DEVICE)


with torch.no_grad():

    predictions_scaled = (
        model(X_test_tensor)
        .cpu()
        .numpy()
        .flatten()
    )


# ============================================================
# CONVERT BACK TO ORIGINAL NDCI SCALE
# ============================================================

actual_ndci = target_scaler.inverse_transform(
    y_test.reshape(-1, 1)
).flatten()

predicted_ndci = target_scaler.inverse_transform(
    predictions_scaled.reshape(-1, 1)
).flatten()


# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(
    actual_ndci,
    predicted_ndci
)

rmse = np.sqrt(
    mean_squared_error(
        actual_ndci,
        predicted_ndci
    )
)

r2 = r2_score(
    actual_ndci,
    predicted_ndci
)


# ============================================================
# RESULTS TABLE
# ============================================================

results = pd.DataFrame({

    "date":
        test_dates["date"].values,

    "actual_ndci":
        actual_ndci,

    "predicted_ndci":
        predicted_ndci

})

results["absolute_error"] = (
    abs(
        results["actual_ndci"]
        -
        results["predicted_ndci"]
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    OUTPUT_DIR /
    "lstm_test_predictions.csv",
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 65)
print("LSTM EVALUATION RESULTS")
print("=" * 65)

print(
    "\nMAE:",
    f"{mae:.6f}"
)

print(
    "RMSE:",
    f"{rmse:.6f}"
)

print(
    "R²:",
    f"{r2:.6f}"
)

print(
    "\nTest observations:",
    len(results)
)

print("\nPredictions:")

print(
    results.to_string(
        index=False
    )
)


print("\nAverage actual NDCI:")

print(
    f"{actual_ndci.mean():.6f}"
)

print("\nAverage predicted NDCI:")

print(
    f"{predicted_ndci.mean():.6f}"
)


print("\nResults saved to:")

print(
    OUTPUT_DIR /
    "lstm_test_predictions.csv"
)

print("\n" + "=" * 65)