from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.segformer.segformer_model import SegFormerHABSegmentation
from utils.dataset_utils import create_datasets


# ============================================================
# PATHS
# ============================================================

TILES_DIR = PROJECT_ROOT / "data" / "tiles"
MODEL_DIR = PROJECT_ROOT / "models" / "segformer"
RESULTS_DIR = PROJECT_ROOT / "results" / "segformer"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 1

# Same number of epochs as Swin
NUM_EPOCHS = 5

LEARNING_RATE = 1e-4

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("SEGFORMER HAB SEGMENTATION TRAINING")
print("=" * 70)

print()
print("Device:", DEVICE)

if DEVICE.type == "cpu":
    print("WARNING: CUDA is not available.")
    print("Training will use CPU.")


# ============================================================
# LOAD DATASETS
# ============================================================

train_dataset, val_dataset, test_dataset = create_datasets(
    TILES_DIR
)

print()
print("DATASET SIZES")
print("-" * 40)

print(
    "Training:",
    len(train_dataset)
)

print(
    "Validation:",
    len(val_dataset)
)

print(
    "Testing:",
    len(test_dataset)
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print()
print(
    "TRAINING BATCHES:",
    len(train_loader)
)

print(
    "VALIDATION BATCHES:",
    len(val_loader)
)


# ============================================================
# CREATE SEGFORMER MODEL
# ============================================================

print()
print("Creating SegFormer model...")

model = SegFormerHABSegmentation(
    num_channels=14,
    num_classes=2
)

model = model.to(DEVICE)

print(
    "Model created successfully."
)


# ============================================================
# LOSS FUNCTION
# ============================================================

# Class 0 = Non-HAB
# Class 1 = HAB
#
# HAB is the minority class.
#
# 255 = outside-water / invalid
# These pixels are ignored during training.

class_weights = torch.tensor(
    [1.0, 10.0],
    dtype=torch.float32,
    device=DEVICE
)

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    ignore_index=255
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")

model_path = (
    MODEL_DIR
    / "best_segformer_hab_model.pth"
)


for epoch in range(NUM_EPOCHS):

    print()
    print("=" * 70)
    print(
        f"EPOCH {epoch + 1}/{NUM_EPOCHS}"
    )
    print("=" * 70)

    # ========================================================
    # TRAINING PHASE
    # ========================================================

    model.train()

    train_loss = 0.0

    for batch_index, (images, masks) in enumerate(
        train_loader
    ):

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            masks
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        print(
            f"\rTraining batch "
            f"{batch_index + 1}/{len(train_loader)} "
            f"- Loss: {loss.item():.4f}",
            end=""
        )

    train_loss /= len(train_loader)

    print()

    print(
        f"Average training loss: "
        f"{train_loss:.4f}"
    )


    # ========================================================
    # VALIDATION PHASE
    # ========================================================

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                masks
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Validation loss: "
        f"{val_loss:.4f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "epoch":
                    epoch + 1,

                "train_loss":
                    train_loss,

                "val_loss":
                    val_loss
            },
            model_path
        )

        print()
        print(
            "New best model saved:"
        )

        print(model_path)

    else:

        print(
            "Validation loss did not improve."
        )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("SEGFORMER TRAINING COMPLETE")
print("=" * 70)

print()

print(
    f"Best validation loss: "
    f"{best_val_loss:.6f}"
)

print()

print("Best model:")
print(model_path)