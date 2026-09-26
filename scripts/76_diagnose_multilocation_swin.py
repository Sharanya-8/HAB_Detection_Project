from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd
from torch.utils.data import DataLoader

from models.swin.swin_model import SwinHABSegmentation

from importlib.util import spec_from_file_location, module_from_spec


# ============================================================
# LOAD DATASET CLASS
# ============================================================

dataset_script = (
    PROJECT_ROOT
    / "scripts"
    / "74_create_pytorch_dataset.py"
)

spec = spec_from_file_location(
    "multilocation_dataset",
    dataset_script
)

dataset_module = module_from_spec(spec)
spec.loader.exec_module(dataset_module)

MultiWaterbodyHABDataset = (
    dataset_module.MultiWaterbodyHABDataset
)


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

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "swin"
    / "multilocation_best_swin_hab_model.pth"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("STEP 76 - MULTI-WATERBODY SWIN DIAGNOSTIC")
print("=" * 70)

print()
print("Device:", DEVICE)

print()
print("Model:")
print(MODEL_PATH)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    SPLIT_FILE
)

val_df = df[
    df["split"] == "val"
].copy()


val_dataset = MultiWaterbodyHABDataset(
    val_df,
    IMAGE_DIR,
    MASK_DIR
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOAD MODEL
# ============================================================

model = SwinHABSegmentation(
    num_channels=14,
    num_classes=2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


print()
print("Checkpoint epoch:")
print(checkpoint.get("epoch"))

print()
print("Checkpoint F1:")
print(checkpoint.get("f1"))


# ============================================================
# COUNTERS
# ============================================================

actual_hab = 0
actual_non_hab = 0

predicted_hab = 0
predicted_non_hab = 0

true_positive = 0
false_positive = 0
false_negative = 0
true_negative = 0

total_valid = 0


# ============================================================
# VALIDATION
# ============================================================

with torch.no_grad():

    for images, masks in val_loader:

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

        pred_non_hab = (
            predictions == 0
        )

        true_hab = (
            masks == 1
        )

        true_non_hab = (
            masks == 0
        )

        actual_hab += (
            true_hab & valid
        ).sum().item()

        actual_non_hab += (
            true_non_hab & valid
        ).sum().item()

        predicted_hab += (
            pred_hab & valid
        ).sum().item()

        predicted_non_hab += (
            pred_non_hab & valid
        ).sum().item()

        true_positive += (
            pred_hab
            & true_hab
            & valid
        ).sum().item()

        false_positive += (
            pred_hab
            & true_non_hab
            & valid
        ).sum().item()

        false_negative += (
            pred_non_hab
            & true_hab
            & valid
        ).sum().item()

        true_negative += (
            pred_non_hab
            & true_non_hab
            & valid
        ).sum().item()

        total_valid += (
            valid.sum().item()
        )


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("VALIDATION PREDICTION DISTRIBUTION")
print("=" * 70)

print()
print(
    f"Total valid pixels     : {total_valid:,}"
)

print(
    f"Actual HAB pixels      : {actual_hab:,}"
)

print(
    f"Actual Non-HAB pixels  : {actual_non_hab:,}"
)

print()
print(
    f"Predicted HAB pixels   : {predicted_hab:,}"
)

print(
    f"Predicted Non-HAB      : {predicted_non_hab:,}"
)


actual_hab_percent = (
    actual_hab
    / total_valid
    * 100
)

predicted_hab_percent = (
    predicted_hab
    / total_valid
    * 100
)


print()
print(
    f"Actual HAB percentage  : "
    f"{actual_hab_percent:.2f}%"
)

print(
    f"Predicted HAB percent  : "
    f"{predicted_hab_percent:.2f}%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()
print(
    f"True Positive  (HAB → HAB)       : "
    f"{true_positive:,}"
)

print(
    f"False Positive (Non-HAB → HAB)   : "
    f"{false_positive:,}"
)

print(
    f"False Negative (HAB → Non-HAB)   : "
    f"{false_negative:,}"
)

print(
    f"True Negative  (Non-HAB → Non-HAB): "
    f"{true_negative:,}"
)


# ============================================================
# METRICS
# ============================================================

precision = (
    true_positive
    / (true_positive + false_positive)
    if true_positive + false_positive > 0
    else 0
)

recall = (
    true_positive
    / (true_positive + false_negative)
    if true_positive + false_negative > 0
    else 0
)

f1 = (
    2 * precision * recall
    / (precision + recall)
    if precision + recall > 0
    else 0
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
    else 0
)


print()
print("=" * 70)
print("FINAL DIAGNOSTIC METRICS")
print("=" * 70)

print()
print(
    f"Precision : {precision:.6f}"
)

print(
    f"Recall    : {recall:.6f}"
)

print(
    f"F1        : {f1:.6f}"
)

print(
    f"IoU       : {iou:.6f}"
)


# ============================================================
# INTERPRETATION
# ============================================================

print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

if predicted_hab_percent < 1:

    print()
    print(
        "The model is predicting almost everything "
        "as Non-HAB."
    )

elif predicted_hab_percent < 5:

    print()
    print(
        "The model is predicting substantially fewer "
        "HAB pixels than the labels."
    )

else:

    print()
    print(
        "The model is predicting a meaningful amount "
        "of HAB pixels."
    )


print()
print("STEP 76 COMPLETE")