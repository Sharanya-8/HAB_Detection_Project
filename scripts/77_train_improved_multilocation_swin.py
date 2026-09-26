from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models.swin.swin_model import SwinHABSegmentation


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


MODEL_PATH = (
    MODEL_DIR
    / "multilocation_improved_best_swin_hab_model.pth"
)

HISTORY_PATH = (
    RESULTS_DIR
    / "multilocation_improved_swin_history.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 1

NUM_EPOCHS = 15

LEARNING_RATE = 1e-4

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CHANNEL NORMALIZATION
# ============================================================

CHANNEL_NAMES = [
    "B2", "B3", "B4", "B5", "B6",
    "B7", "B8", "B8A", "B11", "B12",
    "NDWI", "MNDWI", "NDCI", "FAI"
]

INDEX_RANGES = {
    "NDWI": (-1.0, 1.0),
    "MNDWI": (-1.0, 1.0),
    "NDCI": (-1.0, 1.0),
    "FAI": (-1.0, 1.0)
}


def normalize_image(image):

    image = image.astype(np.float32)

    normalized = np.zeros_like(image)

    # Reflectance bands
    for i in range(10):

        normalized[i] = (
            image[i] / 2.0
        )

    # Index bands
    for i in range(10, 14):

        name = CHANNEL_NAMES[i]

        min_value, max_value = INDEX_RANGES[name]

        normalized[i] = (
            (image[i] - min_value)
            / (max_value - min_value)
        )

    normalized = np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )

    return normalized


# ============================================================
# DATASET
# ============================================================

class MultiWaterbodyHABDataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_dir,
        mask_dir
    ):

        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        tile_id = str(row["tile_id"])

        image_path = (
            self.image_dir
            / f"{tile_id}.npy"
        )

        mask_path = (
            self.mask_dir
            / f"{tile_id}.npy"
        )

        image = np.load(image_path)

        mask = np.load(mask_path)

        image = normalize_image(image)

        image = torch.from_numpy(
            image
        ).float()

        mask = torch.from_numpy(
            mask
        ).long()

        return image, mask


# ============================================================
# DICE LOSS
# ============================================================

def dice_loss(
    logits,
    targets,
    ignore_index=255,
    smooth=1.0
):

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    # Probability of HAB class
    hab_probability = probabilities[:, 1, :, :]

    valid = targets != ignore_index

    target_hab = (
        targets == 1
    ).float()

    hab_probability = (
        hab_probability * valid
    )

    target_hab = (
        target_hab * valid
    )

    intersection = (
        hab_probability * target_hab
    ).sum()

    denominator = (
        hab_probability.sum()
        + target_hab.sum()
    )

    dice = (
        (2.0 * intersection + smooth)
        / (denominator + smooth)
    )

    return 1.0 - dice


# ============================================================
# METRICS
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

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

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

            true_non_hab = (
                masks == 0
            )

            total_tp += (
                pred_hab
                & true_hab
                & valid
            ).sum().item()

            total_fp += (
                pred_hab
                & true_non_hab
                & valid
            ).sum().item()

            total_fn += (
                (~pred_hab)
                & true_hab
                & valid
            ).sum().item()

            total_tn += (
                (~pred_hab)
                & true_non_hab
                & valid
            ).sum().item()

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
        if total_tp + total_fp + total_fn > 0
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

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dice": dice
    }


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 77 - IMPROVED MULTI-WATERBODY SWIN TRAINING")
print("=" * 70)

print()
print("Device:", DEVICE)

if DEVICE.type == "cpu":

    print()
    print("WARNING: CUDA is not available.")
    print("Training will use CPU.")


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
print("Training tiles   :", len(train_df))
print("Validation tiles :", len(val_df))


# ============================================================
# DATASETS
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
# DATALOADERS
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
print("Training batches   :", len(train_loader))
print("Validation batches :", len(val_loader))


# ============================================================
# MODEL
# ============================================================

print()
print("Creating Swin Transformer...")

model = SwinHABSegmentation(
    num_channels=14,
    num_classes=2
)

model = model.to(DEVICE)

print("Model created successfully.")


# ============================================================
# LOSS FUNCTIONS
# ============================================================

class_weights = torch.tensor(
    [1.0, 10.0],
    dtype=torch.float32,
    device=DEVICE
)

cross_entropy = nn.CrossEntropyLoss(
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

best_f1 = -1.0

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

    total_train_loss = 0.0

    for batch_index, (
        images,
        masks
    ) in enumerate(train_loader):

        images = images.to(DEVICE)

        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        ce = cross_entropy(
            outputs,
            masks
        )

        dl = dice_loss(
            outputs,
            masks
        )

        loss = (
            0.7 * ce
            + 0.3 * dl
        )

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()

        print(
            f"\rTraining batch "
            f"{batch_index + 1}/{len(train_loader)} "
            f"- Loss: {loss.item():.4f}",
            end=""
        )


    train_loss = (
        total_train_loss
        / len(train_loader)
    )

    print()

    print(
        f"Average training loss: "
        f"{train_loss:.6f}"
    )


    # --------------------------------------------------------
    # VALIDATION LOSS
    # --------------------------------------------------------

    model.eval()

    total_val_loss = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)

            masks = masks.to(DEVICE)

            outputs = model(images)

            ce = cross_entropy(
                outputs,
                masks
            )

            dl = dice_loss(
                outputs,
                masks
            )

            loss = (
                0.7 * ce
                + 0.3 * dl
            )

            total_val_loss += (
                loss.item()
            )


    val_loss = (
        total_val_loss
        / len(val_loader)
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = calculate_metrics(
        model,
        val_loader
    )


    print()
    print(
        f"Validation loss : "
        f"{val_loss:.6f}"
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


    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        **metrics
    })


    # --------------------------------------------------------
    # SAVE BEST
    # --------------------------------------------------------

    if metrics["f1"] > best_f1:

        best_f1 = metrics["f1"]

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

                **metrics
            },
            MODEL_PATH
        )

        print()
        print(
            "NEW BEST MODEL SAVED"
        )

        print(
            MODEL_PATH
        )

    else:

        print()
        print(
            "Best F1 did not improve."
        )


# ============================================================
# SAVE HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("STEP 77 COMPLETE")
print("=" * 70)

print()
print(
    f"Best validation F1: "
    f"{best_f1:.6f}"
)

print()
print("Best model:")
print(MODEL_PATH)

print()
print("Training history:")
print(HISTORY_PATH)