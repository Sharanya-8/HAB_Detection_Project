import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import rasterio

# ============================================================
# STEP 83H
# PROPER TILE-BASED INFERENCE ON UNSEEN SHAMIRPET LAKE
# ============================================================

# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)

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
    "shamirpet_test",
    "tile_based"
)

PREDICTION_DIR = os.path.join(
    RESULT_DIR,
    "predictions"
)

VISUALIZATION_DIR = os.path.join(
    RESULT_DIR,
    "visualizations"
)

os.makedirs(
    PREDICTION_DIR,
    exist_ok=True
)

os.makedirs(
    VISUALIZATION_DIR,
    exist_ok=True
)

DETAILS_FILE = os.path.join(
    RESULT_DIR,
    "shamirpet_tile_based_image_results.csv"
)

SUMMARY_FILE = os.path.join(
    RESULT_DIR,
    "shamirpet_tile_based_overall_results.csv"
)

# ------------------------------------------------------------
# MODEL INPUT SIZE
# ------------------------------------------------------------

TILE_SIZE = 256

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 83H - TILE-BASED SWIN INFERENCE ON SHAMIRPET")
print("=" * 70)

# ============================================================
# LOAD MODEL
# ============================================================

from models.swin.swin_model import SwinHABSegmentation

device = torch.device("cpu")

print("\nDevice:", device)

model = SwinHABSegmentation(
    num_channels=14,
    num_classes=2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

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
# SAME AS TRAINING
# ============================================================

def normalize_14_band_image(image):

    image = image.astype(
        np.float32
    )

    # --------------------------------------------------------
    # First 10 Sentinel-2 bands
    # --------------------------------------------------------

    image[:10] = (
        image[:10] / 2.0
    )

    # --------------------------------------------------------
    # Last 4 spectral indices
    #
    # NDWI
    # MNDWI
    # NDCI
    # FAI
    #
    # [-1,1] -> [0,1]
    # --------------------------------------------------------

    image[10:] = (
        image[10:] + 1.0
    ) / 2.0

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

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

    valid = (
        target != 255
    )

    prediction_valid = prediction[
        valid
    ]

    target_valid = target[
        valid
    ]

    tp = np.sum(
        (prediction_valid == 1)
        & (target_valid == 1)
    )

    fp = np.sum(
        (prediction_valid == 1)
        & (target_valid == 0)
    )

    fn = np.sum(
        (prediction_valid == 0)
        & (target_valid == 1)
    )

    tn = np.sum(
        (prediction_valid == 0)
        & (target_valid == 0)
    )

    # Precision
    if (tp + fp) > 0:
        precision = (
            tp / (tp + fp)
        )
    else:
        precision = 0.0

    # Recall
    if (tp + fn) > 0:
        recall = (
            tp / (tp + fn)
        )
    else:
        recall = 0.0

    # F1
    if (precision + recall) > 0:
        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )
    else:
        f1 = 0.0

    # IoU
    if (tp + fp + fn) > 0:
        iou = (
            tp
            / (tp + fp + fn)
        )
    else:
        iou = 0.0

    # Dice
    if (
        2 * tp + fp + fn
    ) > 0:
        dice = (
            2 * tp
            / (2 * tp + fp + fn)
        )
    else:
        dice = 0.0

    # Pixel accuracy
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
# EDGE PADDING
# ============================================================

def pad_image_to_tile_size(
    image,
    tile_size=256
):

    channels, height, width = image.shape

    padded_height = (
        int(
            np.ceil(
                height / tile_size
            )
        )
        * tile_size
    )

    padded_width = (
        int(
            np.ceil(
                width / tile_size
            )
        )
        * tile_size
    )

    pad_bottom = (
        padded_height - height
    )

    pad_right = (
        padded_width - width
    )

    # Edge padding keeps the boundary values
    # instead of introducing artificial zeros.

    padded = np.pad(
        image,
        (
            (0, 0),
            (0, pad_bottom),
            (0, pad_right)
        ),
        mode="edge"
    )

    return padded


# ============================================================
# TILE-BASED INFERENCE
# ============================================================

def run_tile_inference(
    image
):

    original_channels = image.shape[0]
    original_height = image.shape[1]
    original_width = image.shape[2]

    if original_channels != 14:

        raise ValueError(
            f"Expected 14 channels, "
            f"found {original_channels}"
        )

    # --------------------------------------------------------
    # Normalize entire image first
    # --------------------------------------------------------

    image = normalize_14_band_image(
        image
    )

    # --------------------------------------------------------
    # Pad ONLY edge portions
    # --------------------------------------------------------

    padded_image = pad_image_to_tile_size(
        image,
        TILE_SIZE
    )

    padded_height = padded_image.shape[1]
    padded_width = padded_image.shape[2]

    # --------------------------------------------------------
    # Empty reconstructed prediction
    # --------------------------------------------------------

    full_prediction = np.zeros(
        (
            padded_height,
            padded_width
        ),
        dtype=np.uint8
    )

    tile_count = 0

    # --------------------------------------------------------
    # Process each 256x256 tile
    # --------------------------------------------------------

    for row in range(
        0,
        padded_height,
        TILE_SIZE
    ):

        for col in range(
            0,
            padded_width,
            TILE_SIZE
        ):

            tile = padded_image[
                :,
                row:row + TILE_SIZE,
                col:col + TILE_SIZE
            ]

            # Safety check
            if tile.shape != (
                14,
                TILE_SIZE,
                TILE_SIZE
            ):

                raise ValueError(
                    f"Unexpected tile shape: "
                    f"{tile.shape}"
                )

            # Convert to tensor

            tensor = torch.from_numpy(
                tile
            ).unsqueeze(
                0
            ).float()

            # ------------------------------------------------
            # Swin inference
            # ------------------------------------------------

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
                .astype(np.uint8)
            )

            # ------------------------------------------------
            # Put prediction back into full image
            # ------------------------------------------------

            full_prediction[
                row:row + TILE_SIZE,
                col:col + TILE_SIZE
            ] = prediction

            tile_count += 1

    # --------------------------------------------------------
    # Remove artificial padded area
    # --------------------------------------------------------

    full_prediction = full_prediction[
        :original_height,
        :original_width
    ]

    return (
        full_prediction,
        tile_count
    )


# ============================================================
# FIND ALL SHAMIRPET IMAGES
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
        "No Shamirpet images found."
    )


# ============================================================
# OVERALL CONFUSION MATRIX
# ============================================================

overall_tp = 0
overall_fp = 0
overall_fn = 0
overall_tn = 0

all_results = []


# ============================================================
# PROCESS ALL 100 IMAGES
# ============================================================

for index, image_path in enumerate(
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

    prediction_output = os.path.join(
        PREDICTION_DIR,
        f"{date}_Swin_prediction.tif"
    )

    visualization_output = os.path.join(
        VISUALIZATION_DIR,
        f"{date}_tile_based_comparison.png"
    )

    try:

        # ====================================================
        # READ IMAGE
        # ====================================================

        with rasterio.open(
            image_path
        ) as src:

            image = src.read()

            image_profile = src.profile.copy()

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

            label_profile = src.profile.copy()

        # ====================================================
        # TILE-BASED SWIN INFERENCE
        # ====================================================

        prediction, tile_count = (
            run_tile_inference(
                image
            )
        )

        # ====================================================
        # METRICS
        # ====================================================

        metrics = calculate_metrics(
            prediction,
            label
        )

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
        # UPDATE OVERALL COUNTS
        # ====================================================

        overall_tp += metrics["tp"]
        overall_fp += metrics["fp"]
        overall_fn += metrics["fn"]
        overall_tn += metrics["tn"]

        # ====================================================
        # SAVE PREDICTION TIFF
        # ====================================================

        prediction_profile = label_profile.copy()

        prediction_profile.update(
            dtype=rasterio.uint8,
            count=1,
            nodata=255,
            compress="lzw"
        )

        # Keep invalid pixels invalid

        prediction_to_save = prediction.copy()

        prediction_to_save[
            label == 255
        ] = 255

        with rasterio.open(
            prediction_output,
            "w",
            **prediction_profile
        ) as dst:

            dst.write(
                prediction_to_save,
                1
            )

        # ====================================================
        # SAVE IMAGE-LEVEL RESULT
        # ====================================================

        all_results.append({

            "date":
                date,

            "image":
                filename,

            "tiles_used":
                tile_count,

            "height":
                image.shape[1],

            "width":
                image.shape[2],

            "valid_pixels":
                int(valid_pixels),

            "actual_hab_pixels":
                int(actual_hab),

            "predicted_hab_pixels":
                int(predicted_hab),

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

        print(
            f"Processed {index}/{len(image_files)} | "
            f"{date} | "
            f"Tiles: {tile_count} | "
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
# CHECK RESULTS
# ============================================================

if len(all_results) == 0:

    raise RuntimeError(
        "No Shamirpet images were successfully processed."
    )


# ============================================================
# SAVE IMAGE RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    DETAILS_FILE,
    index=False
)


# ============================================================
# OVERALL METRICS
# ============================================================

tp = overall_tp
fp = overall_fp
fn = overall_fn
tn = overall_tn

# ------------------------------------------------------------
# Precision
# ------------------------------------------------------------

if (tp + fp) > 0:

    precision = (
        tp / (tp + fp)
    )

else:

    precision = 0.0

# ------------------------------------------------------------
# Recall
# ------------------------------------------------------------

if (tp + fn) > 0:

    recall = (
        tp / (tp + fn)
    )

else:

    recall = 0.0

# ------------------------------------------------------------
# F1
# ------------------------------------------------------------

if (precision + recall) > 0:

    f1 = (
        2
        * precision
        * recall
        / (precision + recall)
    )

else:

    f1 = 0.0

# ------------------------------------------------------------
# IoU
# ------------------------------------------------------------

if (tp + fp + fn) > 0:

    iou = (
        tp
        / (tp + fp + fn)
    )

else:

    iou = 0.0

# ------------------------------------------------------------
# Dice
# ------------------------------------------------------------

if (
    2 * tp + fp + fn
) > 0:

    dice = (
        2 * tp
        / (2 * tp + fp + fn)
    )

else:

    dice = 0.0

# ------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# HAB percentages
# ------------------------------------------------------------

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
# SAVE OVERALL SUMMARY
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
print("STEP 83H COMPLETE")
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
    "\n--- TILE-BASED UNSEEN SHAMIRPET PERFORMANCE ---"
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
    "\nImage-level results saved to:"
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
    "\nPrediction masks saved to:"
)

print(
    PREDICTION_DIR
)

print(
    "\n" + "=" * 70
)