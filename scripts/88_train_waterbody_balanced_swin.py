import os
import sys
import random
import importlib.util

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)


# ------------------------------------------------------------
# IMPORT DATASET FROM STEP 74
# ------------------------------------------------------------

dataset_script = os.path.join(
    PROJECT_ROOT,
    "scripts",
    "74_create_pytorch_dataset.py"
)

spec = importlib.util.spec_from_file_location(
    "dataset_module",
    dataset_script
)

dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_module)

MultiWaterbodyHABDataset = (
    dataset_module.MultiWaterbodyHABDataset
)


# ------------------------------------------------------------
# IMPORT SWIN MODEL
# ------------------------------------------------------------

from models.swin.swin_model import SwinHABSegmentation


# ============================================================
# STEP 88
# ============================================================

print("=" * 70)
print("STEP 88 - WATERBODY-BALANCED SWIN TRAINING")
print("=" * 70)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

SPLIT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "train_val_split",
    "multilocation_train_val_split.csv"
)

IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "tiles",
    "images"
)

MASK_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "tiles",
    "masks"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "swin"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "swin"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


CHECKPOINT_PATH = os.path.join(
    MODEL_DIR,
    "waterbody_balanced_best_swin_hab_model.pth"
)

HISTORY_PATH = os.path.join(
    RESULT_DIR,
    "waterbody_balanced_swin_history.csv"
)


# ------------------------------------------------------------
# REPRODUCIBILITY
# ------------------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"\nDevice: {device}")


# ============================================================
# READ SPLIT
# ============================================================

split_df = pd.read_csv(SPLIT_FILE)

train_df = split_df[
    split_df["split"] == "train"
].reset_index(drop=True)

val_df = split_df[
    split_df["split"] == "val"
].reset_index(drop=True)


print("\nOriginal training tiles:")
print(len(train_df))

print("\nValidation tiles:")
print(len(val_df))


# ============================================================
# WATERBODY DISTRIBUTION
# ============================================================

print("\n")
print("=" * 70)
print("ORIGINAL TRAINING DISTRIBUTION")
print("=" * 70)

print(
    train_df["waterbody"]
    .value_counts()
    .to_string()
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
# WATERBODY-BALANCED SAMPLER
# ============================================================

# Each waterbody receives approximately equal probability
# of being selected during training.

waterbody_counts = (
    train_df["waterbody"]
    .value_counts()
)

num_waterbodies = len(
    waterbody_counts
)

total_train = len(train_df)


weights_by_waterbody = {}

for waterbody, count in waterbody_counts.items():

    weights_by_waterbody[waterbody] = (
        total_train /
        (
            num_waterbodies *
            count
        )
    )


sample_weights = train_df[
    "waterbody"
].map(
    weights_by_waterbody
).values


sample_weights = torch.DoubleTensor(
    sample_weights
)


sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=total_train,
    replacement=True
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    sampler=sampler,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)


print("\n")
print("=" * 70)
print("BALANCED SAMPLING WEIGHTS")
print("=" * 70)

for waterbody, weight in weights_by_waterbody.items():

    print(
        f"{waterbody:20s} : "
        f"{weight:.4f}"
    )


# ============================================================
# MODEL
# ============================================================

model = SwinHABSegmentation(
    num_channels=14,
    num_classes=2
)

model = model.to(device)


# ============================================================
# LOSS
# ============================================================

# Moderate class weighting.
# We deliberately avoid the aggressive weighting from Step 79.

class_weights = torch.tensor(
    [1.0, 4.0],
    dtype=torch.float32,
    device=device
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
    lr=1e-4,
    weight_decay=1e-4
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

EPOCHS = 5

best_f1 = -1.0

history = []


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    model,
    loader
):

    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    total_loss = 0.0
    batches = 0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                masks
            )

            total_loss += loss.item()
            batches += 1

            predictions = (
                torch.argmax(
                    outputs,
                    dim=1
                )
            )

            valid = masks != 255

            pred = predictions[valid]
            true = masks[valid]

            total_tp += torch.sum(
                (pred == 1) &
                (true == 1)
            ).item()

            total_fp += torch.sum(
                (pred == 1) &
                (true == 0)
            ).item()

            total_fn += torch.sum(
                (pred == 0) &
                (true == 1)
            ).item()

            total_tn += torch.sum(
                (pred == 0) &
                (true == 0)
            ).item()


    precision = (
        total_tp /
        (total_tp + total_fp)
        if (total_tp + total_fp) > 0
        else 0
    )

    recall = (
        total_tp /
        (total_tp + total_fn)
        if (total_tp + total_fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    iou = (
        total_tp /
        (
            total_tp +
            total_fp +
            total_fn
        )
        if (
            total_tp +
            total_fp +
            total_fn
        ) > 0
        else 0
    )

    total_valid = (
        total_tp +
        total_fp +
        total_fn +
        total_tn
    )

    actual_hab = (
        (total_tp + total_fn) /
        total_valid *
        100
    )

    predicted_hab = (
        (total_tp + total_fp) /
        total_valid *
        100
    )

    return {
        "loss": (
            total_loss / batches
            if batches > 0
            else 0
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "actual_hab_percent": actual_hab,
        "predicted_hab_percent": predicted_hab
    }


# ============================================================
# TRAIN
# ============================================================

print("\n")
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    running_loss = 0.0

    for batch_idx, (
        images,
        masks
    ) in enumerate(
        train_loader,
        start=1
    ):

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            masks
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:

            print(
                f"Epoch {epoch}/{EPOCHS} "
                f"| Batch {batch_idx}/"
                f"{len(train_loader)}",
                end="\r"
            )


    train_loss = (
        running_loss /
        len(train_loader)
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_metrics = calculate_metrics(
        model,
        val_loader
    )


    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_metrics["loss"],
        "precision": val_metrics["precision"],
        "recall": val_metrics["recall"],
        "f1": val_metrics["f1"],
        "iou": val_metrics["iou"],
        "actual_hab_percent":
            val_metrics["actual_hab_percent"],
        "predicted_hab_percent":
            val_metrics["predicted_hab_percent"]
    }

    history.append(row)


    # --------------------------------------------------------
    # SAVE BEST
    # --------------------------------------------------------

    if val_metrics["f1"] > best_f1:

        best_f1 = val_metrics["f1"]

        torch.save(
            model.state_dict(),
            CHECKPOINT_PATH
        )

        best_marker = " <-- BEST"

    else:

        best_marker = ""


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"\nEpoch {epoch}/{EPOCHS} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_metrics['loss']:.6f} | "
        f"Precision: {val_metrics['precision']:.6f} | "
        f"Recall: {val_metrics['recall']:.6f} | "
        f"F1: {val_metrics['f1']:.6f} | "
        f"IoU: {val_metrics['iou']:.6f} | "
        f"Pred HAB: {val_metrics['predicted_hab_percent']:.2f}%"
        f"{best_marker}"
    )


# ============================================================
# SAVE HISTORY
# ============================================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("STEP 88 COMPLETE")
print("=" * 70)

print(
    f"\nBest validation F1: "
    f"{best_f1:.6f}"
)

print(
    f"\nBest checkpoint:"
)

print(CHECKPOINT_PATH)

print(
    f"\nTraining history:"
)

print(HISTORY_PATH)

print(
    "\nIMPORTANT:"
)

print(
    "Shamirpet was NOT used for training."
)

print(
    "Run the next evaluation step only after "
    "reviewing these results."
)