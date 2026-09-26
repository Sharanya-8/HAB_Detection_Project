import os
import sys
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from models.swin.swin_model import SwinHABSegmentation


# ============================================================
# STEP 81
# CORRECTED MULTI-WATERBODY SWIN TRAINING
# ============================================================

print("=" * 70)
print("STEP 81 - CORRECTED MULTI-WATERBODY SWIN TRAINING")
print("=" * 70)


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device("cpu")

IMAGE_DIR = "data/tiles/images"
MASK_DIR = "data/tiles/masks"

SPLIT_FILE = (
    "data/processed/train_val_split/"
    "multilocation_train_val_split.csv"
)

OUTPUT_MODEL = (
    "models/swin/"
    "multilocation_corrected_best_swin_hab_model.pth"
)

OUTPUT_HISTORY = (
    "results/swin/"
    "multilocation_corrected_swin_history.csv"
)

BATCH_SIZE = 1

EPOCHS = 10

LEARNING_RATE = 1e-4

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

print("\nDevice:", DEVICE)

print(
    "\nCUDA is not being used."
)

print(
    "Training will use CPU."
)


# ============================================================
# LOAD TRAIN / VALIDATION SPLIT
# ============================================================

split_df = pd.read_csv(
    SPLIT_FILE
)

train_df = split_df[
    split_df["split"] == "train"
].copy()

val_df = split_df[
    split_df["split"] == "val"
].copy()

print(
    "\nTraining tiles   :",
    len(train_df)
)

print(
    "Validation tiles :",
    len(val_df)
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_image(image):

    image = image.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Sentinel-2 reflectance bands
    # --------------------------------------------------------

    image[:10] = (
        image[:10] / 2.0
    )

    # --------------------------------------------------------
    # Spectral indices:
    # NDWI, MNDWI, NDCI, FAI
    # --------------------------------------------------------

    image[10:] = (
        image[10:] + 1.0
    ) / 2.0

    # --------------------------------------------------------
    # Remove invalid values
    # --------------------------------------------------------

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # --------------------------------------------------------
    # Keep values between 0 and 1
    # --------------------------------------------------------

    image = np.clip(
        image,
        0.0,
        1.0
    )

    return image


# ============================================================
# DATASET
# ============================================================

class MultiWaterbodyHABDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        image_dir,
        mask_dir
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.image_dir = image_dir

        self.mask_dir = mask_dir


    def __len__(self):

        return len(self.df)


    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[index]

        tile_name = str(
            row["tile_id"]
        )

        image_path = os.path.join(
            self.image_dir,
            tile_name + ".npy"
        )

        mask_path = os.path.join(
            self.mask_dir,
            tile_name + ".npy"
        )

        image = np.load(
            image_path
        )

        mask = np.load(
            mask_path
        )

        image = normalize_image(
            image
        )

        image = torch.from_numpy(
            image
        ).float()

        mask = torch.from_numpy(
            mask
        ).long()

        return image, mask


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
# GENTLE BALANCED SAMPLING
# ============================================================

print(
    "\nCreating gentle balanced sampler..."
)

hab_percent = (
    train_df["hab_percent"]
    .astype(float)
    .values
)


# ------------------------------------------------------------
# STEP 79 used:
#
#     weight = HAB percentage
#
# This was too aggressive and resulted in:
#
#     Predicted HAB = 70.43%
#
# Step 81 uses the square-root of HAB percentage instead.
#
# This still favors informative HAB tiles but reduces the
# difference between low-HAB and high-HAB tiles.
# ------------------------------------------------------------

sample_weights = np.sqrt(
    np.maximum(
        hab_percent,
        1.0
    )
)


sample_weights = torch.as_tensor(
    sample_weights,
    dtype=torch.double
)


sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_dataset),
    replacement=True
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print(
    "\nTraining batches   :",
    len(train_loader)
)

print(
    "Validation batches :",
    len(val_loader)
)


# ============================================================
# MODEL
# ============================================================

print(
    "\nCreating Swin Transformer..."
)

model = SwinHABSegmentation(
    num_channels=14,
    num_classes=2
)

model.to(
    DEVICE
)

print(
    "Model created successfully."
)


# ============================================================
# CLASS-WEIGHTED CROSS ENTROPY
# ============================================================

# ------------------------------------------------------------
# Step 79 used HAB weight = 12.
#
# That produced approximately 70% predicted HAB.
#
# Step 81 reduces the HAB weight to 4.
# ------------------------------------------------------------

class_weights = torch.tensor(
    [1.0, 4.0],
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
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    model,
    loader
):

    model.eval()

    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0

    total_loss = 0.0


    with torch.no_grad():

        for images, masks in loader:

            images = images.to(
                DEVICE
            )

            masks = masks.to(
                DEVICE
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                masks
            )

            total_loss += (
                loss.item()
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            valid = (
                masks != 255
            )

            true_values = masks[
                valid
            ]

            pred_values = predictions[
                valid
            ]


            total_tp += (
                (pred_values == 1)
                &
                (true_values == 1)
            ).sum().item()


            total_tn += (
                (pred_values == 0)
                &
                (true_values == 0)
            ).sum().item()


            total_fp += (
                (pred_values == 1)
                &
                (true_values == 0)
            ).sum().item()


            total_fn += (
                (pred_values == 0)
                &
                (true_values == 1)
            ).sum().item()


    precision = (
        total_tp
        /
        max(
            1,
            total_tp + total_fp
        )
    )

    recall = (
        total_tp
        /
        max(
            1,
            total_tp + total_fn
        )
    )

    f1 = (
        2
        * precision
        * recall
        /
        max(
            1e-12,
            precision + recall
        )
    )

    iou = (
        total_tp
        /
        max(
            1,
            total_tp
            + total_fp
            + total_fn
        )
    )

    dice = (
        2
        * total_tp
        /
        max(
            1,
            2 * total_tp
            + total_fp
            + total_fn
        )
    )

    average_loss = (
        total_loss
        /
        max(
            1,
            len(loader)
        )
    )


    predicted_hab = (
        total_tp
        + total_fp
    )

    actual_hab = (
        total_tp
        + total_fn
    )

    total_valid = (
        total_tp
        + total_tn
        + total_fp
        + total_fn
    )


    predicted_hab_percent = (
        100
        * predicted_hab
        /
        max(
            1,
            total_valid
        )
    )

    actual_hab_percent = (
        100
        * actual_hab
        /
        max(
            1,
            total_valid
        )
    )


    return {
        "loss":
            average_loss,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "iou":
            iou,

        "dice":
            dice,

        "predicted_hab_percent":
            predicted_hab_percent,

        "actual_hab_percent":
            actual_hab_percent
    }


# ============================================================
# TRAINING
# ============================================================

history = []

best_f1 = -1.0


for epoch in range(
    1,
    EPOCHS + 1
):

    print("\n")
    print("=" * 70)

    print(
        f"EPOCH {epoch}/{EPOCHS}"
    )

    print("=" * 70)


    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    total_train_loss = 0.0


    for batch_index, (
        images,
        masks
    ) in enumerate(
        train_loader,
        start=1
    ):

        images = images.to(
            DEVICE
        )

        masks = masks.to(
            DEVICE
        )

        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            masks
        )

        loss.backward()

        optimizer.step()

        total_train_loss += (
            loss.item()
        )


        if (
            batch_index % 100 == 0
            or
            batch_index == len(
                train_loader
            )
        ):

            print(
                f"Training batch "
                f"{batch_index}/"
                f"{len(train_loader)} "
                f"- Loss: "
                f"{loss.item():.4f}"
            )


    train_loss = (
        total_train_loss
        /
        len(train_loader)
    )


    print(
        f"Average training loss: "
        f"{train_loss:.6f}"
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    metrics = calculate_metrics(
        model,
        val_loader
    )


    print(
        f"\nValidation loss : "
        f"{metrics['loss']:.6f}"
    )

    print(
        f"Precision        : "
        f"{metrics['precision']:.6f}"
    )

    print(
        f"Recall           : "
        f"{metrics['recall']:.6f}"
    )

    print(
        f"F1 Score         : "
        f"{metrics['f1']:.6f}"
    )

    print(
        f"IoU              : "
        f"{metrics['iou']:.6f}"
    )

    print(
        f"Dice             : "
        f"{metrics['dice']:.6f}"
    )

    print(
        f"Actual HAB %     : "
        f"{metrics['actual_hab_percent']:.2f}%"
    )

    print(
        f"Predicted HAB %  : "
        f"{metrics['predicted_hab_percent']:.2f}%"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if metrics["f1"] > best_f1:

        best_f1 = metrics["f1"]

        torch.save(
            {
                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "best_f1":
                    best_f1
            },
            OUTPUT_MODEL
        )


        print(
            "\nNEW BEST MODEL SAVED"
        )

        print(
            OUTPUT_MODEL
        )

    else:

        print(
            "\nBest F1 did not improve."
        )


    # ========================================================
    # SAVE HISTORY ENTRY
    # ========================================================

    history.append(
        {
            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "val_loss":
                metrics["loss"],

            "precision":
                metrics["precision"],

            "recall":
                metrics["recall"],

            "f1":
                metrics["f1"],

            "iou":
                metrics["iou"],

            "dice":
                metrics["dice"],

            "actual_hab_percent":
                metrics["actual_hab_percent"],

            "predicted_hab_percent":
                metrics["predicted_hab_percent"]
        }
    )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

os.makedirs(
    os.path.dirname(
        OUTPUT_HISTORY
    ),
    exist_ok=True
)

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    OUTPUT_HISTORY,
    index=False
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n")
print("=" * 70)
print("STEP 81 COMPLETE")
print("=" * 70)

print(
    f"\nBest validation F1: "
    f"{best_f1:.6f}"
)

print(
    "\nBest model saved to:"
)

print(
    OUTPUT_MODEL
)

print(
    "\nTraining history saved to:"
)

print(
    OUTPUT_HISTORY
)

print("=" * 70)