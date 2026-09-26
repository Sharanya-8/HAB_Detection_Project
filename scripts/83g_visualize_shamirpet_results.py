import os
import glob
import numpy as np
import rasterio

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
# ============================================================
# STEP 83G
# VISUALIZE SHAMIRPET SWIN RESULTS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

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

PREDICTION_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "swin",
    "shamirpet_test",
    "predictions"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "swin",
    "shamirpet_test",
    "visualizations"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("=" * 70)
print("STEP 83G - VISUALIZE SHAMIRPET SWIN RESULTS")
print("=" * 70)

# ============================================================
# IMPORTANT
# ============================================================
#
# Step 83F currently calculated predictions but did not save
# the prediction masks.
#
# Therefore this script will recreate the predictions using
# the same trained Swin model.
#
# ============================================================

import sys

sys.path.insert(
    0,
    PROJECT_ROOT
)

import torch
import torch.nn.functional as F

from models.swin.swin_model import SwinHABSegmentation


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")

print("\nDevice:", device)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "swin",
    "multilocation_corrected_best_swin_hab_model.pth"
)

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

    state_dict = checkpoint[
        "model_state_dict"
    ]

else:

    state_dict = checkpoint

model.load_state_dict(
    state_dict
)

model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# NORMALIZATION
# SAME AS TRAINING
# ============================================================

def normalize_14_band_image(image):

    image = image.astype(
        np.float32
    )

    # First 10 Sentinel-2 bands
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
# SELECT 5 REPRESENTATIVE DATES
# ============================================================

all_images = sorted(
    glob.glob(
        os.path.join(
            IMAGE_DIR,
            "*.tif"
        )
    )
)

if len(all_images) == 0:

    raise RuntimeError(
        "No Shamirpet images found."
    )

# Select approximately:
# early period
# 2018
# 2020
# 2022
# latest period

selected_indices = [
    0,
    len(all_images) // 4,
    len(all_images) // 2,
    (3 * len(all_images)) // 4,
    len(all_images) - 1
]

selected_images = [
    all_images[i]
    for i in selected_indices
]

print(
    f"\nTotal Shamirpet images: "
    f"{len(all_images)}"
)

print("\nSelected dates:")

for path in selected_images:

    print(
        " -",
        os.path.basename(path)
    )


# ============================================================
# PROCESS SELECTED DATES
# ============================================================

for image_path in selected_images:

    filename = os.path.basename(
        image_path
    )

    date = filename.replace(
        ".tif",
        ""
    )

    print(
        f"\nProcessing {date}..."
    )

    label_path = os.path.join(
        LABEL_DIR,
        f"{date}_HAB.tif"
    )

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    with rasterio.open(
        image_path
    ) as src:

        image = src.read()

    if image.shape[0] != 14:

        raise ValueError(
            f"{filename} has "
            f"{image.shape[0]} bands. "
            f"Expected 14."
        )

    # --------------------------------------------------------
    # READ LABEL
    # --------------------------------------------------------

    with rasterio.open(
        label_path
    ) as src:

        label = src.read(1)

    # --------------------------------------------------------
    # SAVE ORIGINAL SIZE
    # --------------------------------------------------------

    original_height = image.shape[1]
    original_width = image.shape[2]

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    image = normalize_14_band_image(
        image
    )

    # --------------------------------------------------------
    # CONVERT TO TORCH
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0).float()

    # --------------------------------------------------------
    # RESIZE TO MODEL INPUT
    # --------------------------------------------------------

    tensor = F.interpolate(
        tensor,
        size=(256, 256),
        mode="bilinear",
        align_corners=False
    )

    # --------------------------------------------------------
    # RESIZE LABEL TO SAME SIZE
    # --------------------------------------------------------

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

    label_256 = (
        label_tensor
        .squeeze()
        .numpy()
        .astype(np.uint8)
    )

    # --------------------------------------------------------
    # SWIN PREDICTION
    # --------------------------------------------------------

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

    # ========================================================
    # CREATE RGB IMAGE
    #
    # Sentinel-2:
    # B4 = Red
    # B3 = Green
    # B2 = Blue
    #
    # These are bands 3,2,1 in zero-based Python indexing.
    # ========================================================

    red = image[2]
    green = image[1]
    blue = image[0]

    rgb = np.stack(
        [
            red,
            green,
            blue
        ],
        axis=-1
    )

    # Improve display contrast

    rgb = np.clip(
        rgb * 2.5,
        0,
        1
    )

    # ========================================================
    # CREATE MASKS
    # ========================================================

    actual_hab = (
        label_256 == 1
    )

    predicted_hab = (
        prediction == 1
    )

    # ========================================================
    # CREATE COMPARISON FIGURE
    # ========================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6)
    )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    axes[0].imshow(
        rgb
    )

    axes[0].set_title(
        f"Shamirpet Sentinel-2\n{date}"
    )

    axes[0].axis(
        "off"
    )

    # --------------------------------------------------------
    # PSEUDO LABEL
    # --------------------------------------------------------

    axes[1].imshow(
        actual_hab,
        cmap="hot",
        vmin=0,
        vmax=1
    )

    axes[1].set_title(
        "Pseudo-label HAB"
    )

    axes[1].axis(
        "off"
    )

    # --------------------------------------------------------
    # SWIN PREDICTION
    # --------------------------------------------------------

    axes[2].imshow(
        predicted_hab,
        cmap="hot",
        vmin=0,
        vmax=1
    )

    axes[2].set_title(
        "Swin HAB Prediction"
    )

    axes[2].axis(
        "off"
    )

    # --------------------------------------------------------
    # MAIN TITLE
    # --------------------------------------------------------

    fig.suptitle(
        f"Shamirpet Lake HAB Detection - {date}",
        fontsize=16
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{date}_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output_path
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("STEP 83G COMPLETE")
print("=" * 70)

print(
    "\nVisualizations saved in:"
)

print(
    OUTPUT_DIR
)

print(
    "\nOpen the generated PNG files and inspect:"
)

print(
    "1. Sentinel-2 image"
)

print(
    "2. Pseudo-label HAB"
)

print(
    "3. Swin HAB prediction"
)

print(
    "\nDo not retrain the model yet."
)