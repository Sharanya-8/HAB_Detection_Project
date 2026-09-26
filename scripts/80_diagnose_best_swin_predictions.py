import os
import sys
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from models.swin.swin_model import SwinHABSegmentation


# ============================================================
# STEP 80
# DIAGNOSE BEST MULTI-WATERBODY SWIN MODEL
# ============================================================

print("=" * 70)
print("STEP 80 - DIAGNOSE BEST SWIN PREDICTIONS")
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

MODEL_FILE = (
    "models/swin/"
    "multilocation_balanced_best_swin_hab_model.pth"
)

RESULTS_DIR = (
    "results/swin/step80_diagnosis"
)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking required files...")

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_FILE}"
    )

if not os.path.exists(SPLIT_FILE):
    raise FileNotFoundError(
        f"Split file not found:\n{SPLIT_FILE}"
    )

print("Model found.")
print("Split file found.")


# ============================================================
# LOAD SPLIT
# ============================================================

df = pd.read_csv(
    SPLIT_FILE
)

val_df = df[
    df["split"] == "val"
].copy()

print(
    "\nValidation tiles:",
    len(val_df)
)

print(
    "\nValidation waterbodies:"
)

print(
    val_df["waterbody"]
    .value_counts()
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_image(image):

    image = image.astype(
        np.float32
    )

    # Sentinel-2 reflectance bands
    image[:10] = (
        image[:10] / 2.0
    )

    # NDWI, MNDWI, NDCI, FAI
    image[10:] = (
        image[10:] + 1.0
    ) / 2.0

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    image = np.clip(
        image,
        0.0,
        1.0
    )

    return image


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Swin model...")

model = SwinHABSegmentation(
    num_channels=14,
    num_classes=2
)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)

# Support our checkpoint format
if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch",
            "unknown"
        )
    )

    print(
        "Checkpoint best F1:",
        checkpoint.get(
            "best_f1",
            "unknown"
        )
    )

else:

    model.load_state_dict(
        checkpoint
    )

    print(
        "Loaded raw model checkpoint."
    )


model.to(DEVICE)

model.eval()

print(
    "Model loaded successfully."
)


# ============================================================
# METRIC STORAGE
# ============================================================

overall_tp = 0
overall_tn = 0
overall_fp = 0
overall_fn = 0

waterbody_results = []


# ============================================================
# REPRESENTATIVE TILE SELECTION
# ============================================================

print(
    "\nSelecting representative validation tiles..."
)

selected_tiles = []

for waterbody in sorted(
    val_df["waterbody"].unique()
):

    wb_df = val_df[
        val_df["waterbody"]
        == waterbody
    ].copy()

    # Sort by HAB percentage
    wb_df = wb_df.sort_values(
        "hab_percent"
    )

    if len(wb_df) == 0:
        continue

    # Select:
    # 1 low-HAB tile
    # 1 medium-HAB tile
    # 1 high-HAB tile

    indices = [
        0,
        len(wb_df) // 2,
        len(wb_df) - 1
    ]

    chosen = wb_df.iloc[
        sorted(set(indices))
    ]

    for _, row in chosen.iterrows():

        selected_tiles.append(
            row
        )

        print(
            f"{row['waterbody']} | "
            f"{row['tile_id']} | "
            f"Actual HAB: "
            f"{row['hab_percent']:.2f}%"
        )


# ============================================================
# DIAGNOSE ALL VALIDATION TILES
# ============================================================

print(
    "\nRunning prediction diagnosis..."
)

processed = 0


with torch.no_grad():

    for _, row in val_df.iterrows():

        tile_name = str(
            row["tile_id"]
        )

        image_path = os.path.join(
            IMAGE_DIR,
            tile_name + ".npy"
        )

        mask_path = os.path.join(
            MASK_DIR,
            tile_name + ".npy"
        )

        if not os.path.exists(
            image_path
        ):
            continue

        if not os.path.exists(
            mask_path
        ):
            continue

        image = np.load(
            image_path
        )

        mask = np.load(
            mask_path
        )

        image = normalize_image(
            image
        )

        image_tensor = torch.from_numpy(
            image
        ).float().unsqueeze(0)

        image_tensor = image_tensor.to(
            DEVICE
        )

        output = model(
            image_tensor
        )

        prediction = torch.argmax(
            output,
            dim=1
        )[0].cpu().numpy()

        valid = (
            mask != 255
        )

        actual = mask[
            valid
        ]

        predicted = prediction[
            valid
        ]

        tp = (
            (predicted == 1)
            &
            (actual == 1)
        ).sum()

        tn = (
            (predicted == 0)
            &
            (actual == 0)
        ).sum()

        fp = (
            (predicted == 1)
            &
            (actual == 0)
        ).sum()

        fn = (
            (predicted == 0)
            &
            (actual == 1)
        ).sum()

        overall_tp += int(tp)
        overall_tn += int(tn)
        overall_fp += int(fp)
        overall_fn += int(fn)

        processed += 1

        if (
            processed % 25 == 0
            or processed == len(val_df)
        ):

            print(
                f"Processed "
                f"{processed}/"
                f"{len(val_df)}"
            )


# ============================================================
# OVERALL METRICS
# ============================================================

precision = (
    overall_tp
    /
    max(
        1,
        overall_tp + overall_fp
    )
)

recall = (
    overall_tp
    /
    max(
        1,
        overall_tp + overall_fn
    )
)

f1 = (
    2 * precision * recall
    /
    max(
        1e-12,
        precision + recall
    )
)

iou = (
    overall_tp
    /
    max(
        1,
        overall_tp
        + overall_fp
        + overall_fn
    )
)

dice = (
    2 * overall_tp
    /
    max(
        1,
        2 * overall_tp
        + overall_fp
        + overall_fn
    )
)

actual_hab = (
    overall_tp
    + overall_fn
)

actual_non_hab = (
    overall_tn
    + overall_fp
)

predicted_hab = (
    overall_tp
    + overall_fp
)

predicted_non_hab = (
    overall_tn
    + overall_fn
)


# ============================================================
# PRINT OVERALL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("OVERALL STEP 80 DIAGNOSIS")
print("=" * 70)

print(
    f"\nValid pixels:"
    f" {actual_hab + actual_non_hab:,}"
)

print(
    f"Actual HAB:"
    f" {actual_hab:,}"
)

print(
    f"Actual Non-HAB:"
    f" {actual_non_hab:,}"
)

print(
    f"\nPredicted HAB:"
    f" {predicted_hab:,}"
)

print(
    f"Predicted Non-HAB:"
    f" {predicted_non_hab:,}"
)

print(
    f"\nActual HAB %:"
    f" {100 * actual_hab / max(1, actual_hab + actual_non_hab):.2f}%"
)

print(
    f"Predicted HAB %:"
    f" {100 * predicted_hab / max(1, actual_hab + actual_non_hab):.2f}%"
)

print("\nConfusion Matrix:")

print(
    f"TP: {overall_tp:,}"
)

print(
    f"FP: {overall_fp:,}"
)

print(
    f"FN: {overall_fn:,}"
)

print(
    f"TN: {overall_tn:,}"
)

print("\nMetrics:")

print(
    f"Precision: {precision:.6f}"
)

print(
    f"Recall:    {recall:.6f}"
)

print(
    f"F1:        {f1:.6f}"
)

print(
    f"IoU:       {iou:.6f}"
)

print(
    f"Dice:      {dice:.6f}"
)


# ============================================================
# REPRESENTATIVE VISUALIZATIONS
# ============================================================

print(
    "\nCreating representative prediction visualizations..."
)


for row in selected_tiles:

    tile_name = str(
        row["tile_id"]
    )

    waterbody = str(
        row["waterbody"]
    )

    image_path = os.path.join(
        IMAGE_DIR,
        tile_name + ".npy"
    )

    mask_path = os.path.join(
        MASK_DIR,
        tile_name + ".npy"
    )

    if not os.path.exists(
        image_path
    ):
        continue

    if not os.path.exists(
        mask_path
    ):
        continue

    image = np.load(
        image_path
    )

    mask = np.load(
        mask_path
    )

    normalized = normalize_image(
        image
    )

    image_tensor = torch.from_numpy(
        normalized
    ).float().unsqueeze(0)

    image_tensor = image_tensor.to(
        DEVICE
    )

    with torch.no_grad():

        output = model(
            image_tensor
        )

        prediction = torch.argmax(
            output,
            dim=1
        )[0].cpu().numpy()


    # --------------------------------------------------------
    # Display an RGB composite
    #
    # Sentinel-2:
    # B4 = red
    # B3 = green
    # B2 = blue
    #
    # These are channels 2,1,0.
    # --------------------------------------------------------

    rgb = normalized[
        [2, 1, 0]
    ]

    rgb = np.transpose(
        rgb,
        (1, 2, 0)
    )

    rgb = np.clip(
        rgb,
        0,
        1
    )


    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )


    # RGB
    axes[0].imshow(
        rgb
    )

    axes[0].set_title(
        "Sentinel-2 RGB"
    )

    axes[0].axis(
        "off"
    )


    # Actual mask
    actual_display = np.where(
        mask == 255,
        np.nan,
        mask
    )

    axes[1].imshow(
        actual_display
    )

    axes[1].set_title(
        "Actual HAB Mask"
    )

    axes[1].axis(
        "off"
    )


    # Prediction
    prediction_display = np.where(
        mask == 255,
        np.nan,
        prediction
    )

    axes[2].imshow(
        prediction_display
    )

    axes[2].set_title(
        "Swin Prediction"
    )

    axes[2].axis(
        "off"
    )


    fig.suptitle(
        f"{waterbody} | "
        f"{tile_name}\n"
        f"Actual HAB: "
        f"{row['hab_percent']:.2f}%",
        fontsize=12
    )

    plt.tight_layout()


    safe_name = (
        tile_name
        .replace(
            "\\",
            "_"
        )
        .replace(
            "/",
            "_"
        )
    )

    output_file = os.path.join(
        RESULTS_DIR,
        safe_name
        + "_diagnosis.png"
    )

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output_file
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        {
            "valid_pixels":
                actual_hab
                + actual_non_hab,

            "actual_hab":
                actual_hab,

            "actual_non_hab":
                actual_non_hab,

            "predicted_hab":
                predicted_hab,

            "predicted_non_hab":
                predicted_non_hab,

            "TP":
                overall_tp,

            "FP":
                overall_fp,

            "FN":
                overall_fn,

            "TN":
                overall_tn,

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
        }
    ]
)

summary_file = os.path.join(
    RESULTS_DIR,
    "step80_overall_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 70)
print("STEP 80 COMPLETE")
print("=" * 70)

print(
    "\nDiagnosis results saved to:"
)

print(
    RESULTS_DIR
)

print(
    "\nSummary saved to:"
)

print(
    summary_file
)

print("=" * 70)