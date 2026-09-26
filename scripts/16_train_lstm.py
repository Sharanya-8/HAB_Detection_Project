from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


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

MODEL_DIR = (
    BASE_DIR /
    "models" /
    "lstm"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

INPUT_SIZE = 13
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

DEVICE = torch.device("cpu")


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("LSTM NDCI FORECASTING")
print("=" * 65)

X_train = np.load(
    DATA_DIR / "X_train.npy"
)

y_train = np.load(
    DATA_DIR / "y_train.npy"
)

X_val = np.load(
    DATA_DIR / "X_val.npy"
)

y_val = np.load(
    DATA_DIR / "y_val.npy"
)

X_test = np.load(
    DATA_DIR / "X_test.npy"
)

y_test = np.load(
    DATA_DIR / "y_test.npy"
)


print("\nData shapes:")

print(
    "X_train:",
    X_train.shape
)

print(
    "y_train:",
    y_train.shape
)

print(
    "X_val:",
    X_val.shape
)

print(
    "y_val:",
    y_val.shape
)

print(
    "X_test:",
    X_test.shape
)

print(
    "y_test:",
    y_test.shape
)


# ============================================================
# CONVERT TO PYTORCH
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
    shuffle=False
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

        prediction = self.fc(
            last_output
        )

        return prediction


# ============================================================
# CREATE MODEL
# ============================================================

model = LSTMForecaster(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(DEVICE)


print("\nModel:")
print(model)


# ============================================================
# LOSS AND OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")

best_epoch = 0


print("\n" + "=" * 65)
print("TRAINING")
print("=" * 65)


for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(
            X_batch
        )

        loss = criterion(
            predictions,
            y_batch
        )

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item() *
            X_batch.size(0)
        )


    train_loss /= len(
        train_dataset
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            predictions = model(
                X_batch
            )

            loss = criterion(
                predictions,
                y_batch
            )

            val_loss += (
                loss.item() *
                X_batch.size(0)
            )


    val_loss /= len(
        val_dataset
    )


    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS} "
        f"| Train Loss: {train_loss:.6f} "
        f"| Val Loss: {val_loss:.6f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_epoch = epoch

        torch.save(
            model.state_dict(),
            MODEL_DIR /
            "best_lstm_model.pth"
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_DIR /
        "best_lstm_model.pth",
        map_location=DEVICE
    )
)

model.eval()


# ============================================================
# TEST LOSS
# ============================================================

test_loss = 0.0

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        predictions = model(
            X_batch
        )

        loss = criterion(
            predictions,
            y_batch
        )

        test_loss += (
            loss.item() *
            X_batch.size(0)
        )


test_loss /= len(
    test_dataset
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 65)
print("LSTM TRAINING COMPLETE")
print("=" * 65)

print(
    "\nBest epoch:",
    best_epoch
)

print(
    "Best validation loss:",
    f"{best_val_loss:.6f}"
)

print(
    "Test MSE:",
    f"{test_loss:.6f}"
)

print(
    "\nBest model saved to:"
)

print(
    MODEL_DIR /
    "best_lstm_model.pth"
)

print("\n" + "=" * 65)