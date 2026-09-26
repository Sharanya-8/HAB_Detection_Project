import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import rasterio

# ============================================================
# STEP 83F
# TEST TRAINED SWIN MODEL ON UNSEEN SHAMIRPET LAKE
# ============================================================

# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Make project root available to Python
sys.path.insert(0, PROJECT_ROOT)

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "sentinel2",
    "shamirpet_test"
)

LABEL_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "labels",
    "hab_masks",
    "shamirpet_test"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "swin",
    "multilocation_corrected_best_swin_hab_model.pth"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "swin",
    "shamirpet_test"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

DETAILS_FILE = os.path.join(
    RESULT_DIR,
    "shamirpet_image_results.csv"
)

SUMMARY_FILE = os.path.join(
    RESULT_DIR,
    "shamirpet_overall_results.csv"
)

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 83F - SWIN TEST ON UNSEEN SHAMIRPET LAKE")
print("=" * 70)

# ============================================================
# LOAD MODEL
# ============================================================

from models.swin.swin_model import SwinHABSegmentation

device = torch.device("cpu")

print("\nDevice:", device)

# Create same model architecture used during training
model = SwinHABSegmentation(
    num_channels=14,
    num_classes=2
)

# Load trained checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

# Handle either:
# 1. normal state_dict
# 2. checkpoint containing model_state_dict

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(
    state_dict
)

model.to(device)

model.eval()

print("Model loaded successfully:")
print(MODEL_PATH)

# ============================================================
# NORMALIZATION
# SAME NORMALIZATION USED DURING TRAINING
# ============================================================

def normalize_14_band_image(image):

    image = image.astype(
        np.float32
    )

    # --------------------------------------------------------
    # First 10 bands:
    #
    # B2 B3 B4 B5 B6 B7 B8 B8A B11 B12
    #
    # Training normalization:
    # divide reflectance bands by 2
    # --------------------------------------------------------

    image[:10] = image[:10] / 2.0

    # --------------------------------------------------------
    # Last 4 bands:
    #
    # NDWI
    # MNDWI
    # NDCI
    # FAI
    #
    # Convert approximately:
    # [-1, 1] -> [0, 1]
    # --------------------------------------------------------

    image[10:] = (
        image[10:] + 1.0
    ) / 2.0

    # Remove invalid numerical values

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    # Keep values inside [0,1]

    image = np.clip(
        image,
        0.0,
        1.0
    )

    return image


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    prediction,
    target
):

    # --------------------------------------------------------
    # Ignore invalid pixels
    # 255 = invalid
    # --------------------------------------------------------

    valid = (
        target != 255
    )

    prediction = prediction[valid]
    target = target[valid]

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    tp = np.sum(
        (prediction == 1)
        & (target == 1)
    )

    fp = np.sum(
        (prediction == 1)
        & (target == 0)
    )

    fn = np.sum(
        (prediction == 0)
        & (target == 1)
    )

    tn = np.sum(
        (prediction == 0)
        & (target == 0)
    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    if (tp + fp) > 0:
        precision = (
            tp / (tp + fp)
        )
    else:
        precision = 0.0

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    if (tp + fn) > 0:
        recall = (
            tp / (tp + fn)
        )
    else:
        recall = 0.0

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    if (precision + recall) > 0:

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    else:
        f1 = 0.0

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    if (tp + fp + fn) > 0:

        iou = (
            tp
            / (tp + fp + fn)
        )

    else:
        iou = 0.0

    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

    if (
        2 * tp + fp + fn
    ) > 0:

        dice = (
            2 * tp
            / (2 * tp + fp + fn)
        )

    else:
        dice = 0.0

    # --------------------------------------------------------
    # Pixel accuracy
    # --------------------------------------------------------

    total = (
        tp
        + tn
        + fp
        + fn
    )

    if total > 0:

        accuracy = (
            (tp + tn)
            / total
        )

    else:
        accuracy = 0.0

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "dice": float(dice),
        "accuracy": float(accuracy)
    }


# ============================================================
# FIND SHAMIRPET IMAGES
# ============================================================

image_files = sorted(
    glob.glob(
        os.path.join(
            IMAGE_DIR,
            "*.tif"
        )
    )
)

print(
    "\nShamirpet images found:",
    len(image_files)
)

if len(image_files) == 0:

    raise RuntimeError(
        "No Shamirpet TIFF images found."
    )


# ============================================================
# PROCESS ALL IMAGES
# ============================================================

all_results = []

# Overall confusion matrix
overall_tp = 0
overall_fp = 0
overall_fn = 0
overall_tn = 0

# ============================================================
# LOOP
# ============================================================

for i, image_path in enumerate(
    image_files,
    start=1
):

    filename = os.path.basename(
        image_path
    )

    date = filename.replace(
        ".tif",
        ""
    )

    label_path = os.path.join(
        LABEL_DIR,
        f"{date}_HAB.tif"
    )

    try:

        # ====================================================
        # READ 14-BAND IMAGE
        # ====================================================

        with rasterio.open(
            image_path
        ) as src:

            image = src.read()

        # Check number of bands

        if image.shape[0] != 14:

            raise ValueError(
                f"Expected 14 bands, "
                f"found {image.shape[0]}"
            )

        # ====================================================
        # READ LABEL
        # ====================================================

        if not os.path.exists(
            label_path
        ):

            raise FileNotFoundError(
                f"Label not found: {label_path}"
            )

        with rasterio.open(
            label_path
        ) as src:

            label = src.read(1)

        # ====================================================
        # NORMALIZE IMAGE
        # ====================================================

        image = normalize_14_band_image(
            image
        )

        # ====================================================
        # CONVERT IMAGE TO TORCH
        # ====================================================

        tensor = torch.from_numpy(
            image
        ).unsqueeze(0).float()

        # ====================================================
        # RESIZE IMAGE
        #
        # Training tiles were:
        # 14 x 256 x 256
        #
        # Shamirpet source images have a different
        # spatial size.
        #
        # Therefore resize ONLY the model input.
        # Original TIFF remains unchanged.
        # ====================================================

        tensor = F.interpolate(
            tensor,
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        )

        # ====================================================
        # RESIZE LABEL
        #
        # Use nearest-neighbor so that:
        #
        # 0 remains 0
        # 1 remains 1
        # 255 remains 255
        # ====================================================

        label_tensor = torch.from_numpy(
            label.astype(
                np.float32
            )
        ).unsqueeze(
            0
        ).unsqueeze(
            0
        )

        label_tensor = F.interpolate(
            label_tensor,
            size=(256, 256),
            mode="nearest"
        )

        label = (
            label_tensor
            .squeeze()
            .numpy()
            .astype(np.uint8)
        )

        # ====================================================
        # SWIN INFERENCE
        # ====================================================

        with torch.no_grad():

            logits = model(
                tensor.to(device)
            )

            prediction = torch.argmax(
                logits,
                dim=1
            )

        prediction = (
            prediction
            .squeeze(0)
            .cpu()
            .numpy()
        )

        # ====================================================
        # CALCULATE METRICS
        # ====================================================

        metrics = calculate_metrics(
            prediction,
            label
        )

        # ====================================================
        # HAB COUNTS
        # ====================================================

        valid_pixels = np.sum(
            label != 255
        )

        actual_hab = np.sum(
            (label == 1)
            & (label != 255)
        )

        predicted_hab = np.sum(
            (prediction == 1)
            & (label != 255)
        )

        if valid_pixels > 0:

            actual_hab_percent = (
                actual_hab
                / valid_pixels
                * 100
            )

            predicted_hab_percent = (
                predicted_hab
                / valid_pixels
                * 100
            )

        else:

            actual_hab_percent = 0.0
            predicted_hab_percent = 0.0

        # ====================================================
        # ADD TO OVERALL CONFUSION MATRIX
        # ====================================================

        overall_tp += metrics["tp"]
        overall_fp += metrics["fp"]
        overall_fn += metrics["fn"]
        overall_tn += metrics["tn"]

        # ====================================================
        # STORE IMAGE RESULT
        # ====================================================

        all_results.append({

            "date": date,

            "image": filename,

            "valid_pixels": int(
                valid_pixels
            ),

            "actual_hab_pixels": int(
                actual_hab
            ),

            "predicted_hab_pixels": int(
                predicted_hab
            ),

            "actual_hab_percent":
                actual_hab_percent,

            "predicted_hab_percent":
                predicted_hab_percent,

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

            "accuracy":
                metrics["accuracy"]
        })

        # ====================================================
        # PROGRESS
        # ====================================================

        if (
            i % 10 == 0
            or i == len(image_files)
        ):

            print(
                f"Processed {i}/{len(image_files)} | "
                f"{date} | "
                f"Actual HAB: "
                f"{actual_hab_percent:.2f}% | "
                f"Predicted HAB: "
                f"{predicted_hab_percent:.2f}% | "
                f"F1: "
                f"{metrics['f1']:.4f}"
            )

    except Exception as e:

        print(
            f"\nFAILED: {filename}"
        )

        print(
            f"Reason: {e}"
        )


# ============================================================
# CHECK WHETHER ANY IMAGE SUCCEEDED
# ============================================================

if len(all_results) == 0:

    raise RuntimeError(
        "\nNo Shamirpet images were successfully "
        "processed. Overall metrics cannot be calculated."
    )


# ============================================================
# SAVE IMAGE-LEVEL RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    DETAILS_FILE,
    index=False
)


# ============================================================
# OVERALL CONFUSION MATRIX
# ============================================================

tp = overall_tp
fp = overall_fp
fn = overall_fn
tn = overall_tn


# ============================================================
# OVERALL PRECISION
# ============================================================

if (tp + fp) > 0:

    precision = (
        tp / (tp + fp)
    )

else:

    precision = 0.0


# ============================================================
# OVERALL RECALL
# ============================================================

if (tp + fn) > 0:

    recall = (
        tp / (tp + fn)
    )

else:

    recall = 0.0


# ============================================================
# OVERALL F1
# ============================================================

if (precision + recall) > 0:

    f1 = (
        2
        * precision
        * recall
        / (precision + recall)
    )

else:

    f1 = 0.0


# ============================================================
# OVERALL IOU
# ============================================================

if (tp + fp + fn) > 0:

    iou = (
        tp
        / (tp + fp + fn)
    )

else:

    iou = 0.0


# ============================================================
# OVERALL DICE
# ============================================================

if (
    2 * tp + fp + fn
) > 0:

    dice = (
        2 * tp
        / (2 * tp + fp + fn)
    )

else:

    dice = 0.0


# ============================================================
# OVERALL PIXEL ACCURACY
# ============================================================

total_valid_pixels = (
    tp
    + tn
    + fp
    + fn
)

if total_valid_pixels > 0:

    accuracy = (
        (tp + tn)
        / total_valid_pixels
    )

else:

    accuracy = 0.0


# ============================================================
# HAB PERCENTAGES
# ============================================================

actual_hab_pixels = (
    tp + fn
)

predicted_hab_pixels = (
    tp + fp
)

if total_valid_pixels > 0:

    actual_hab_percent = (
        actual_hab_pixels
        / total_valid_pixels
        * 100
    )

    predicted_hab_percent = (
        predicted_hab_pixels
        / total_valid_pixels
        * 100
    )

else:

    actual_hab_percent = 0.0
    predicted_hab_percent = 0.0


# ============================================================
# OVERALL SUMMARY
# ============================================================

summary = pd.DataFrame([{

    "waterbody":
        "Shamirpet_Lake",

    "images_tested":
        len(all_results),

    "valid_pixels":
        total_valid_pixels,

    "actual_hab_pixels":
        actual_hab_pixels,

    "predicted_hab_pixels":
        predicted_hab_pixels,

    "actual_hab_percent":
        actual_hab_percent,

    "predicted_hab_percent":
        predicted_hab_percent,

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

    "accuracy":
        accuracy
}])


summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("STEP 83F COMPLETE")
print("=" * 70)

print(
    f"\nImages successfully tested : "
    f"{len(all_results)}"
)

print(
    f"Actual HAB                : "
    f"{actual_hab_percent:.2f}%"
)

print(
    f"Predicted HAB             : "
    f"{predicted_hab_percent:.2f}%"
)

print(
    "\n--- UNSEEN SHAMIRPET PERFORMANCE ---"
)

print(
    f"Precision                 : "
    f"{precision:.6f}"
)

print(
    f"Recall                    : "
    f"{recall:.6f}"
)

print(
    f"F1 Score                  : "
    f"{f1:.6f}"
)

print(
    f"IoU                       : "
    f"{iou:.6f}"
)

print(
    f"Dice                      : "
    f"{dice:.6f}"
)

print(
    f"Pixel Accuracy            : "
    f"{accuracy:.6f}"
)

print(
    "\nConfusion Matrix:"
)

print(
    f"TP: {tp:,}"
)

print(
    f"FP: {fp:,}"
)

print(
    f"FN: {fn:,}"
)

print(
    f"TN: {tn:,}"
)

print(
    "\nDetailed results saved to:"
)

print(
    DETAILS_FILE
)

print(
    "\nOverall results saved to:"
)

print(
    SUMMARY_FILE
)

print(
    "\n" + "=" * 70
)