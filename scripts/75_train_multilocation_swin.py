from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.swin.swin_model import SwinHABSegmentation

# Import the verified dataset from Step 74
from importlib.util import spec_from_file_location, module_from_spec

dataset_script = PROJECT_ROOT / "scripts" / "74_create_pytorch_dataset.py"

spec = spec_from_file_location(
    "multilocation_dataset",
    dataset_script
)

dataset_module = module_from_spec(spec)

# Prevent Step 74 from running its test code during import.
# We will recreate the datasets directly below.
spec.loader.exec_module(dataset_module)

MultiWaterbodyHABDataset = (
    dataset_module.MultiWaterbodyHABDataset
)

import pandas as pd


# ============================================================
# PATHS
# ============================================================

SPLIT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_val_split"
    / "multilocation_train_val_split.csv"
)

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "tiles"
    / "images"
)

MASK_DIR = (
    PROJECT_ROOT
    / "data"
    / "tiles"
    / "masks"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "swin"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "swin"
)

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

NUM_EPOCHS = 5

LEARNING_RATE = 1e-4

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# New checkpoint — do NOT overwrite old model
MODEL_PATH = (
    MODEL_DIR
    / "multilocation_best_swin_hab_model.pth"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("MULTI-WATERBODY SWIN TRANSFORMER TRAINING")
print("=" * 70)

print()
print("Device:", DEVICE)

if DEVICE.type == "cpu":

    print()
    print("WARNING: CUDA is not available.")
    print("Training will use CPU.")
    print("This may take a long time.")


# ============================================================
# LOAD SPLIT
# ============================================================

df = pd.read_csv(
    SPLIT_FILE
)

train_df = df[
    df["split"] == "train"
].copy()

val_df = df[
    df["split"] == "val"
].copy()


print()
print("=" * 70)
print("DATASET")
print("=" * 70)

print(
    "Training tiles   :",
    len(train_df)
)

print(
    "Validation tiles :",
    len(val_df)
)


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = MultiWaterbodyHABDataset(
    train_df,
    IMAGE_DIR,
    MASK_DIR
)

val_dataset = MultiWaterbodyHABDataset(
    val_df,
    IMAGE_DIR,
    MASK_DIR
)


# ============================================================
# CREATE DATALOADERS
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
    "Training batches   :",
    len(train_loader)
)

print(
    "Validation batches :",
    len(val_loader)
)


# ============================================================
# CREATE MODEL
# ============================================================

print()
print("=" * 70)
print("CREATING SWIN MODEL")
print("=" * 70)

model = SwinHABSegmentation(
    num_channels=14,
    num_classes=2
)

model = model.to(DEVICE)

print("Model created successfully.")


# ============================================================
# LOSS
# ============================================================

# Class 0 = Non-HAB
# Class 1 = HAB
# 255 = ignored pixels

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
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    predictions,
    targets
):

    predictions = predictions.cpu()
    targets = targets.cpu()

    valid = targets != 255

    predictions = predictions[valid]
    targets = targets[valid]

    if predictions.numel() == 0:

        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "iou": 0.0,
            "dice": 0.0
        }

    predictions = predictions == 1
    targets = targets == 1

    true_positive = (
        (predictions & targets)
        .sum()
        .item()
    )

    false_positive = (
        (predictions & ~targets)
        .sum()
        .item()
    )

    false_negative = (
        (~predictions & targets)
        .sum()
        .item()
    )

    precision = (
        true_positive
        / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )

    recall = (
        true_positive
        / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    iou = (
        true_positive
        / (
            true_positive
            + false_positive
            + false_negative
        )
        if (
            true_positive
            + false_positive
            + false_negative
        ) > 0
        else 0.0
    )

    dice = (
        2 * true_positive
        / (
            2 * true_positive
            + false_positive
            + false_negative
        )
        if (
            2 * true_positive
            + false_positive
            + false_negative
        ) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dice": dice
    }


# ============================================================
# TRAINING
# ============================================================

best_val_f1 = -1.0

history = []


for epoch in range(NUM_EPOCHS):

    print()
    print("=" * 70)
    print(
        f"EPOCH {epoch + 1}/{NUM_EPOCHS}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

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
        f"{train_loss:.6f}"
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    total_tp = 0
    total_fp = 0
    total_fn = 0

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

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            valid = masks != 255

            pred_hab = (
                predictions == 1
            )

            true_hab = (
                masks == 1
            )

            total_tp += (
                pred_hab
                & true_hab
                & valid
            ).sum().item()

            total_fp += (
                pred_hab
                & ~true_hab
                & valid
            ).sum().item()

            total_fn += (
                ~pred_hab
                & true_hab
                & valid
            ).sum().item()


    val_loss /= len(val_loader)

    precision = (
        total_tp
        / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )

    recall = (
        total_tp
        / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    iou = (
        total_tp
        / (
            total_tp
            + total_fp
            + total_fn
        )
        if (
            total_tp
            + total_fp
            + total_fn
        ) > 0
        else 0.0
    )

    dice = (
        2 * total_tp
        / (
            2 * total_tp
            + total_fp
            + total_fn
        )
        if (
            2 * total_tp
            + total_fp
            + total_fn
        ) > 0
        else 0.0
    )


    # --------------------------------------------------------
    # PRINT METRICS
    # --------------------------------------------------------

    print()
    print("Validation loss :", f"{val_loss:.6f}")
    print("Precision        :", f"{precision:.6f}")
    print("Recall           :", f"{recall:.6f}")
    print("F1 Score         :", f"{f1:.6f}")
    print("IoU              :", f"{iou:.6f}")
    print("Dice             :", f"{dice:.6f}")


    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dice": dice
    })


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if f1 > best_val_f1:

        best_val_f1 = f1

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "epoch":
                    epoch + 1,

                "train_loss":
                    train_loss,

                "val_loss":
                    val_loss,

                "precision":
                    precision,

                "recall":
                    recall,

                "f1":
                    f1,

                "iou":
                    iou,

                "dice":
                    dice
            },
            MODEL_PATH
        )

        print()
        print("NEW BEST MODEL SAVED")
        print(MODEL_PATH)

    else:

        print()
        print("Best F1 did not improve.")


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_file = (
    RESULTS_DIR
    / "multilocation_swin_training_history.csv"
)

history_df.to_csv(
    history_file,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("MULTI-WATERBODY SWIN TRAINING COMPLETE")
print("=" * 70)

print()
print(
    f"Best validation F1: "
    f"{best_val_f1:.6f}"
)

print()
print("Best model:")
print(MODEL_PATH)

print()
print("Training history:")
print(history_file)