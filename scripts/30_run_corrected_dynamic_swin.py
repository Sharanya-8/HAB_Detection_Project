from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.swin.swin_model import SwinHABSegmentation


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path(
    "models/swin/best_swin_hab_model.pth"
)

TILES_DIR = Path(
    "data/tiles/dynamic_corrected"
)

OUTPUT_DIR = Path(
    "results/maps/dynamic_corrected"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEVICE = torch.device("cpu")


# ============================================================
# START
# ============================================================

print("=" * 70)
print("CORRECTED DYNAMIC SWIN INFERENCE")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained Swin model...")

model = SwinHABSegmentation(
    num_channels=14,
    num_classes=2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
)
elif "state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["state_dict"]
)
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")
print(f"Device: {DEVICE}")


# ============================================================
# FIND TILES
# ============================================================

tile_files = sorted(
    TILES_DIR.glob(
        "dynamic_corrected_tile_*.npy"
    )
)

if not tile_files:
    raise FileNotFoundError(
        "No corrected dynamic tiles found."
    )

print(
    f"\nTiles found: {len(tile_files)}"
)


# ============================================================
# INFERENCE
# ============================================================

total_hab = 0
total_pixels = 0

prediction_tiles = []


print("\nRunning inference...")


with torch.no_grad():

    for number, tile_file in enumerate(
        tile_files,
        start=1
    ):

        print(
            f"\nProcessing tile "
            f"{number}/{len(tile_files)}: "
            f"{tile_file.name}"
        )

        image = np.load(
            tile_file
        ).astype(np.float32)


        if image.shape != (
            14,
            256,
            256
        ):

            raise RuntimeError(
                f"Invalid tile shape: "
                f"{image.shape}"
            )


        image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )


        tensor = torch.from_numpy(
            image
        ).unsqueeze(0).to(DEVICE)


        logits = model(tensor)


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


        hab_pixels = int(
            np.sum(prediction == 1)
        )

        total = prediction.size

        percentage = (
            hab_pixels /
            total *
            100
        )


        print(
            f"HAB pixels: {hab_pixels}"
        )

        print(
            f"Tile HAB percentage: "
            f"{percentage:.2f}%"
        )


        output_file = (
            OUTPUT_DIR /
            f"{tile_file.stem}_prediction.npy"
        )


        np.save(
            output_file,
            prediction
        )


        prediction_tiles.append(
            prediction
        )


        total_hab += hab_pixels
        total_pixels += total


        print(
            f"Saved: {output_file}"
        )


# ============================================================
# OVERALL
# ============================================================

overall_percentage = (
    total_hab /
    total_pixels *
    100
)


print("\n" + "=" * 70)
print("OVERALL RESULT")
print("=" * 70)

print(
    f"Tiles processed: "
    f"{len(tile_files)}"
)

print(
    f"Total pixels: "
    f"{total_pixels}"
)

print(
    f"HAB pixels: "
    f"{total_hab}"
)

print(
    f"Predicted HAB percentage: "
    f"{overall_percentage:.2f}%"
)


# ============================================================
# VERIFY
# ============================================================

all_values = np.concatenate(
    [
        tile.flatten()
        for tile in prediction_tiles
    ]
)

unique_values = np.unique(
    all_values
)

print(
    f"\nPrediction values: "
    f"{unique_values}"
)


if not set(
    unique_values.tolist()
).issubset({0, 1}):

    raise RuntimeError(
        "Unexpected prediction values."
    )


print(
    "Prediction verification: PASSED"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print(
    "Corrected dynamic Sentinel-2 data "
    "was successfully processed by Swin."
)

print(
    "Pixel-level HAB predictions were generated."
)

print("=" * 70)