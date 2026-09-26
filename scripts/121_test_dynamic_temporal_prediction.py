from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dynamic_temporal"
    / "dynamic_temporal_features.csv"
)

LSTM_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "temporal"
    / "multiwaterbody_lstm_best.pth"
)

GRU_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "temporal"
    / "multiwaterbody_gru_best.pth"
)

SCALER_FILE = (
    PROJECT_ROOT
    / "models"
    / "temporal"
    / "multiwaterbody_temporal_feature_scaler.pkl"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 3

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


# ============================================================
# MODEL
# ============================================================

class MultiWaterbodyTemporalModel(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout,
        kind
    ):

        super().__init__()

        self.kind = kind.upper()

        recurrent = (
            nn.LSTM
            if self.kind == "LSTM"
            else nn.GRU
        )

        layer = recurrent(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        if self.kind == "LSTM":

            self.lstm = layer

        else:

            self.gru = layer

        self.fc = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        if self.kind == "LSTM":

            out, _ = self.lstm(x)

        else:

            out, _ = self.gru(x)

        return self.fc(
            out[:, -1, :]
        )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    model_path,
    kind
):

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    model = MultiWaterbodyTemporalModel(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
        kind=kind,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("STEP 121 - DYNAMIC TEMPORAL MODEL TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    for path in [
        DATASET_FILE,
        LSTM_MODEL_FILE,
        GRU_MODEL_FILE,
        SCALER_FILE,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    print()
    print("Loading dynamic temporal dataset...")

    df = pd.read_csv(
        DATASET_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = (
        df.sort_values("date")
        .reset_index(drop=True)
    )

    print(
        f"Total observations: {len(df)}"
    )

    # --------------------------------------------------------
    # REMOVE ROWS WITH MISSING FEATURES
    # --------------------------------------------------------

    df = df.dropna(
        subset=FEATURE_COLUMNS
    ).reset_index(
        drop=True
    )

    if len(df) < SEQUENCE_LENGTH:

        raise ValueError(
            "Not enough observations "
            "for a sequence of length 3."
        )

    # --------------------------------------------------------
    # SHOW LAST OBSERVATIONS
    # --------------------------------------------------------

    print()
    print("Last historical observations:")

    print(
        df[
            [
                "date",
                "ndci_mean"
            ]
        ]
        .tail(SEQUENCE_LENGTH)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # LOAD SCALER
    # --------------------------------------------------------

    print()
    print("Loading feature scaler...")

    with open(
        SCALER_FILE,
        "rb"
    ) as f:

        scaler = pickle.load(f)

    # --------------------------------------------------------
    # PREPARE LAST 3 OBSERVATIONS
    # --------------------------------------------------------

    last_rows = (
        df[
            FEATURE_COLUMNS
        ]
        .tail(SEQUENCE_LENGTH)
        .values
        .astype(np.float32)
    )

    scaled = scaler.transform(
        last_rows
    ).astype(
        np.float32
    )

    x = torch.from_numpy(
        scaled
    ).unsqueeze(0)

    print()
    print(
        f"Model input shape: "
        f"{tuple(x.shape)}"
    )

    # Expected:
    # (1, 3, 10)

    # --------------------------------------------------------
    # LOAD LSTM
    # --------------------------------------------------------

    print()
    print("Loading LSTM...")

    lstm_model, lstm_checkpoint = (
        load_model(
            LSTM_MODEL_FILE,
            "LSTM"
        )
    )

    # --------------------------------------------------------
    # LOAD GRU
    # --------------------------------------------------------

    print("Loading GRU...")

    gru_model, gru_checkpoint = (
        load_model(
            GRU_MODEL_FILE,
            "GRU"
        )
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    with torch.no_grad():

        lstm_scaled = float(
            lstm_model(x)
            .cpu()
            .numpy()
            .reshape(-1)[0]
        )

        gru_scaled = float(
            gru_model(x)
            .cpu()
            .numpy()
            .reshape(-1)[0]
        )

    # --------------------------------------------------------
    # CONVERT BACK TO ORIGINAL NDCI SCALE
    # --------------------------------------------------------

    ndci_mean_index = (
        FEATURE_COLUMNS.index(
            "ndci_mean"
        )
    )

    ndci_mean_value = (
        float(
            scaler.mean_[
                ndci_mean_index
            ]
        )
    )

    ndci_scale_value = (
        float(
            scaler.scale_[
                ndci_mean_index
            ]
        )
    )

    lstm_prediction = (
        lstm_scaled
        * ndci_scale_value
        + ndci_mean_value
    )

    gru_prediction = (
        gru_scaled
        * ndci_scale_value
        + ndci_mean_value
    )

    mean_prediction = (
        lstm_prediction
        + gru_prediction
    ) / 2.0

    # --------------------------------------------------------
    # LAST OBSERVATION
    # --------------------------------------------------------

    last_date = (
        df["date"].iloc[-1]
    )

    last_ndci = (
        float(
            df["ndci_mean"].iloc[-1]
        )
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DYNAMIC TEMPORAL PREDICTION")
    print("=" * 70)

    print()

    print(
        f"Last historical date: "
        f"{last_date.strftime('%Y-%m-%d')}"
    )

    print(
        f"Last observed NDCI: "
        f"{last_ndci:.6f}"
    )

    print()

    print(
        f"LSTM predicted NDCI: "
        f"{lstm_prediction:.6f}"
    )

    print(
        f"GRU predicted NDCI: "
        f"{gru_prediction:.6f}"
    )

    print(
        f"Mean predicted NDCI: "
        f"{mean_prediction:.6f}"
    )

    print()

    print(
        "Prediction test completed successfully."
    )

    print("=" * 70)