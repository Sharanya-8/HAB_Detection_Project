import os
import sys
import torch
import numpy as np
import pandas as pd

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from models.swin.swin_model import SwinHABSegmentation


# ============================================================
# STEP 78 - DIAGNOSE IMPROVED SWIN PREDICTIONS
# ============================================================

print("=" * 70)
print("STEP 78 - DIAGNOSE IMPROVED SWIN PREDICTIONS")
print("=" * 70)


DEVICE = torch.device("cpu")

CHECKPOINT = "models/swin/multilocation_improved_best_swin_hab_model.pth"

SPLIT_FILE = (
    "data/processed/train_val_split/"
    "multilocation_train_val_split.csv"
)

IMAGE_DIR = "data/tiles/images"
MASK_DIR = "data/tiles/masks"


print("\nDevice:", DEVICE)
print("Checkpoint:", CHECKPOINT)


# ============================================================
# LOAD VALIDATION SPLIT
# ============================================================

split_df = pd.read_csv(SPLIT_FILE)

val_df = split_df[
    split_df["split"] == "val"
].copy()

print("\nValidation tiles:", len(val_df))


# ============================================================
# CREATE MODEL
# ============================================================

print("\nCreating Swin Transformer...")

model = SwinHABSegmentation(
    num_channels=14,
    num_classes=2
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_image(image):

    image = image.astype(np.float32)

    # First 10 bands = Sentinel-2 reflectance
    image[:10] = image[:10] / 2.0

    # Last 4 bands = spectral indices
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
# DIAGNOSTIC COUNTERS
# ============================================================

total_pixels = 0

actual_hab = 0
actual_nonhab = 0

predicted_hab = 0
predicted_nonhab = 0

tp = 0
tn = 0
fp = 0
fn = 0

prediction_percentages = []


# ============================================================
# RUN VALIDATION
# ============================================================

print("\nRunning validation diagnosis...\n")


with torch.no_grad():

    for i, row in enumerate(
        val_df.itertuples(index=False),
        start=1
    ):

        # ----------------------------------------------------
        # tile_id ALREADY contains the tile filename stem
        #
        # Example:
        # Himayat_Sagar_2016-03-10_tile_001
        # ----------------------------------------------------

        tile_name = str(row.tile_id)

        image_filename = (
            tile_name + ".npy"
        )

        mask_filename = (
            tile_name + ".npy"
        )

        image_path = os.path.join(
            IMAGE_DIR,
            image_filename
        )

        mask_path = os.path.join(
            MASK_DIR,
            mask_filename
        )


        # ----------------------------------------------------
        # CHECK IMAGE
        # ----------------------------------------------------

        if not os.path.exists(image_path):

            print(
                "\nMissing image:"
            )

            print(image_path)

            continue


        # ----------------------------------------------------
        # CHECK MASK
        # ----------------------------------------------------

        if not os.path.exists(mask_path):

            print(
                "\nMissing mask:"
            )

            print(mask_path)

            continue


        # ----------------------------------------------------
        # LOAD IMAGE AND MASK
        # ----------------------------------------------------

        image = np.load(
            image_path
        )

        mask = np.load(
            mask_path
        )


        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        image = normalize_image(
            image
        )


        # ----------------------------------------------------
        # CONVERT TO TENSORS
        # ----------------------------------------------------

        image_tensor = (
            torch.from_numpy(image)
            .unsqueeze(0)
            .float()
            .to(DEVICE)
        )

        mask_tensor = (
            torch.from_numpy(mask)
            .long()
            .to(DEVICE)
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        outputs = model(
            image_tensor
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        ).squeeze(0)


        # ----------------------------------------------------
        # IGNORE INVALID PIXELS
        # ----------------------------------------------------

        valid = (
            mask_tensor != 255
        )

        true_values = (
            mask_tensor[valid]
        )

        pred_values = (
            predictions[valid]
        )


        # ----------------------------------------------------
        # TOTAL PIXELS
        # ----------------------------------------------------

        total_pixels += (
            true_values.numel()
        )


        # ----------------------------------------------------
        # ACTUAL LABELS
        # ----------------------------------------------------

        actual_hab += (
            true_values == 1
        ).sum().item()

        actual_nonhab += (
            true_values == 0
        ).sum().item()


        # ----------------------------------------------------
        # PREDICTED LABELS
        # ----------------------------------------------------

        predicted_hab += (
            pred_values == 1
        ).sum().item()

        predicted_nonhab += (
            pred_values == 0
        ).sum().item()


        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        tp += (
            (pred_values == 1)
            & (true_values == 1)
        ).sum().item()

        tn += (
            (pred_values == 0)
            & (true_values == 0)
        ).sum().item()

        fp += (
            (pred_values == 1)
            & (true_values == 0)
        ).sum().item()

        fn += (
            (pred_values == 0)
            & (true_values == 1)
        ).sum().item()


        # ----------------------------------------------------
        # PREDICTED HAB PERCENTAGE
        # ----------------------------------------------------

        predicted_hab_pixels = (
            pred_values == 1
        ).sum().item()

        hab_percent = (
            predicted_hab_pixels
            / max(
                1,
                true_values.numel()
            )
        ) * 100

        prediction_percentages.append(
            hab_percent
        )


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            i % 25 == 0
            or i == len(val_df)
        ):

            print(
                f"Processed "
                f"{i}/{len(val_df)} "
                f"validation tiles"
            )


# ============================================================
# CALCULATE METRICS
# ============================================================

precision = (
    tp /
    max(
        1,
        tp + fp
    )
)

recall = (
    tp /
    max(
        1,
        tp + fn
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
    tp /
    max(
        1,
        tp + fp + fn
    )
)

dice = (
    2 * tp /
    max(
        1,
        2 * tp + fp + fn
    )
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("STEP 78 DIAGNOSTIC RESULTS")
print("=" * 70)


print("\nTOTAL VALID PIXELS")
print("------------------")

print(
    f"Total pixels       : "
    f"{total_pixels:,}"
)


print("\nACTUAL LABELS")
print("-------------")

print(
    f"Actual HAB         : "
    f"{actual_hab:,}"
)

print(
    f"Actual Non-HAB     : "
    f"{actual_nonhab:,}"
)

print(
    f"Actual HAB %       : "
    f"{actual_hab / max(1, total_pixels) * 100:.2f}%"
)


print("\nMODEL PREDICTIONS")
print("-----------------")

print(
    f"Predicted HAB      : "
    f"{predicted_hab:,}"
)

print(
    f"Predicted Non-HAB  : "
    f"{predicted_nonhab:,}"
)

print(
    f"Predicted HAB %    : "
    f"{predicted_hab / max(1, total_pixels) * 100:.2f}%"
)


print("\nCONFUSION MATRIX")
print("----------------")

print(
    f"True Positive      : "
    f"{tp:,}"
)

print(
    f"False Positive     : "
    f"{fp:,}"
)

print(
    f"False Negative     : "
    f"{fn:,}"
)

print(
    f"True Negative      : "
    f"{tn:,}"
)


print("\nMETRICS")
print("-------")

print(
    f"Precision           : "
    f"{precision:.6f}"
)

print(
    f"Recall              : "
    f"{recall:.6f}"
)

print(
    f"F1 Score            : "
    f"{f1:.6f}"
)

print(
    f"IoU                 : "
    f"{iou:.6f}"
)

print(
    f"Dice                : "
    f"{dice:.6f}"
)


# ============================================================
# TILE-LEVEL PREDICTION STATISTICS
# ============================================================

if prediction_percentages:

    print(
        "\nPREDICTED HAB % PER TILE"
    )

    print(
        "-------------------------"
    )

    print(
        f"Minimum             : "
        f"{min(prediction_percentages):.2f}%"
    )

    print(
        f"Maximum             : "
        f"{max(prediction_percentages):.2f}%"
    )

    print(
        f"Mean                : "
        f"{np.mean(prediction_percentages):.2f}%"
    )

    print(
        f"Median              : "
        f"{np.median(prediction_percentages):.2f}%"
    )


# ============================================================
# FINAL DIAGNOSIS
# ============================================================

print("\n")
print("=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)

if predicted_hab == 0:

    print(
        "\nWARNING:"
    )

    print(
        "The model predicted ZERO HAB pixels."
    )

    print(
        "The training strategy needs to be changed."
    )

elif f1 == 0:

    print(
        "\nWARNING:"
    )

    print(
        "The model produced predictions, "
        "but no correct HAB predictions."
    )

else:

    print(
        "\nThe model is predicting HAB pixels."
    )

print("=" * 70)
