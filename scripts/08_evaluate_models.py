from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from torch.utils.data import DataLoader

from models.swin.swin_model import SwinHABSegmentation
from models.segformer.segformer_model import SegFormerHABSegmentation
from utils.dataset_utils import create_datasets


# ============================================================
# PATHS
# ============================================================

TILES_DIR = PROJECT_ROOT / "data" / "tiles"

SWIN_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "swin"
    / "best_swin_hab_model.pth"
)

SEGFORMER_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "segformer"
    / "best_segformer_hab_model.pth"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SWIN_RESULT_FILE = (
    RESULTS_DIR / "swin_evaluation.txt"
)

SEGFORMER_RESULT_FILE = (
    RESULTS_DIR / "segformer_evaluation.txt"
)

COMPARISON_FILE = (
    RESULTS_DIR / "model_comparison.txt"
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
print("HAB MODEL EVALUATION")
print("=" * 70)

print()
print("Device:", DEVICE)


# ============================================================
# CHECK MODEL FILES
# ============================================================

if not SWIN_MODEL_PATH.exists():

    print()
    print("ERROR: Swin checkpoint not found:")
    print(SWIN_MODEL_PATH)
    sys.exit(1)


if not SEGFORMER_MODEL_PATH.exists():

    print()
    print("ERROR: SegFormer checkpoint not found:")
    print(SEGFORMER_MODEL_PATH)
    sys.exit(1)


# ============================================================
# LOAD TEST DATASET
# ============================================================

_, _, test_dataset = create_datasets(
    TILES_DIR
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

print()
print("Test tiles:", len(test_dataset))


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    predictions,
    targets
):

    # Ignore outside-water / invalid pixels
    valid = targets != 255

    predictions = predictions[valid]
    targets = targets[valid]

    if len(targets) == 0:

        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "iou": 0.0,
            "dice": 0.0
        }

    predictions = predictions.astype(
        np.int64
    )

    targets = targets.astype(
        np.int64
    )

    true_positive = np.sum(
        (predictions == 1)
        & (targets == 1)
    )

    false_positive = np.sum(
        (predictions == 1)
        & (targets == 0)
    )

    false_negative = np.sum(
        (predictions == 0)
        & (targets == 1)
    )

    correct = np.sum(
        predictions == targets
    )

    total = len(targets)

    accuracy = correct / total

    precision = (
        true_positive
        / (true_positive + false_positive + 1e-8)
    )

    recall = (
        true_positive
        / (true_positive + false_negative + 1e-8)
    )

    f1 = (
        2 * precision * recall
        / (precision + recall + 1e-8)
    )

    iou = (
        true_positive
        / (
            true_positive
            + false_positive
            + false_negative
            + 1e-8
        )
    )

    dice = (
        2 * true_positive
        / (
            2 * true_positive
            + false_positive
            + false_negative
            + 1e-8
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dice": dice
    }


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate_model(
    model,
    checkpoint_path,
    model_name
):

    print()
    print("=" * 70)
    print(f"EVALUATING {model_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    print()
    print("Model loaded successfully.")

    print(
        "Best checkpoint epoch:",
        checkpoint.get(
            "epoch",
            "unknown"
        )
    )

    print(
        "Validation loss:",
        checkpoint.get(
            "val_loss",
            "unknown"
        )
    )

    # --------------------------------------------------------
    # Run test set
    # --------------------------------------------------------

    all_predictions = []
    all_targets = []

    print()
    print("Evaluating test tiles...")

    with torch.no_grad():

        for index, (images, masks) in enumerate(
            test_loader
        ):

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.append(
                predictions.cpu().numpy()
            )

            all_targets.append(
                masks.numpy()
            )

            print(
                f"\rTest tile "
                f"{index + 1}/{len(test_loader)}",
                end=""
            )

    print()

    # --------------------------------------------------------
    # Combine results
    # --------------------------------------------------------

    predictions = np.concatenate(
        all_predictions,
        axis=0
    )

    targets = np.concatenate(
        all_targets,
        axis=0
    )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        predictions,
        targets
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print()
    print(
        f"{model_name} RESULTS"
    )

    print("-" * 50)

    for name, value in metrics.items():

        print(
            f"{name.capitalize():12s}: "
            f"{value:.4f}"
        )

    return metrics, checkpoint


# ============================================================
# EVALUATE SWIN
# ============================================================

swin_model = SwinHABSegmentation(
    num_channels=14,
    num_classes=2
)

swin_metrics, swin_checkpoint = evaluate_model(
    swin_model,
    SWIN_MODEL_PATH,
    "SWIN TRANSFORMER"
)


# ============================================================
# EVALUATE SEGFORMER
# ============================================================

segformer_model = SegFormerHABSegmentation(
    num_channels=14,
    num_classes=2
)

segformer_metrics, segformer_checkpoint = evaluate_model(
    segformer_model,
    SEGFORMER_MODEL_PATH,
    "SEGFORMER"
)


# ============================================================
# SAVE INDIVIDUAL RESULTS
# ============================================================

with open(
    SWIN_RESULT_FILE,
    "w"
) as file:

    file.write(
        "SWIN TRANSFORMER HAB MODEL EVALUATION\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        f"Device: {DEVICE}\n"
    )

    file.write(
        f"Test tiles: {len(test_dataset)}\n"
    )

    file.write(
        f"Best checkpoint epoch: "
        f"{swin_checkpoint.get('epoch', 'unknown')}\n"
    )

    file.write(
        f"Validation loss: "
        f"{swin_checkpoint.get('val_loss', 'unknown')}\n\n"
    )

    for name, value in swin_metrics.items():

        file.write(
            f"{name}: {value:.6f}\n"
        )


with open(
    SEGFORMER_RESULT_FILE,
    "w"
) as file:

    file.write(
        "SEGFORMER HAB MODEL EVALUATION\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        f"Device: {DEVICE}\n"
    )

    file.write(
        f"Test tiles: {len(test_dataset)}\n"
    )

    file.write(
        f"Best checkpoint epoch: "
        f"{segformer_checkpoint.get('epoch', 'unknown')}\n"
    )

    file.write(
        f"Validation loss: "
        f"{segformer_checkpoint.get('val_loss', 'unknown')}\n\n"
    )

    for name, value in segformer_metrics.items():

        file.write(
            f"{name}: {value:.6f}\n"
        )


# ============================================================
# MODEL COMPARISON
# ============================================================

print()
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print()

print(
    f"{'Metric':<15}"
    f"{'Swin':>15}"
    f"{'SegFormer':>15}"
)

print("-" * 45)

for metric in swin_metrics:

    print(
        f"{metric.capitalize():<15}"
        f"{swin_metrics[metric]:>15.4f}"
        f"{segformer_metrics[metric]:>15.4f}"
    )


# ============================================================
# SELECT BEST MODEL
# ============================================================

if swin_metrics["f1"] >= segformer_metrics["f1"]:

    best_model = "Swin Transformer"

else:

    best_model = "SegFormer"


print()
print(
    "Selected model based on F1 score:",
    best_model
)


# ============================================================
# SAVE COMPARISON
# ============================================================

with open(
    COMPARISON_FILE,
    "w"
) as file:

    file.write(
        "HAB MODEL COMPARISON\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        f"Test tiles: {len(test_dataset)}\n"
    )

    file.write(
        f"Device: {DEVICE}\n\n"
    )

    file.write(
        f"{'Metric':<15}"
        f"{'Swin':>15}"
        f"{'SegFormer':>15}\n"
    )

    file.write(
        "-" * 45 + "\n"
    )

    for metric in swin_metrics:

        file.write(
            f"{metric:<15}"
            f"{swin_metrics[metric]:>15.6f}"
            f"{segformer_metrics[metric]:>15.6f}\n"
        )

    file.write("\n")

    file.write(
        f"Selected model based on F1: "
        f"{best_model}\n"
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("Individual results saved:")
print(SWIN_RESULT_FILE)
print(SEGFORMER_RESULT_FILE)

print()
print("Model comparison saved:")
print(COMPARISON_FILE)

print()
print("=" * 70)
print("MODEL EVALUATION COMPLETE")
print("=" * 70)