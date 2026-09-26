from pathlib import Path
import sys

import numpy as np
import torch


# Add project root to Python import path
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
    "data/tiles/dynamic_test"
)

OUTPUT_DIR = Path(
    "results/maps/dynamic_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TILE_SIZE = 256

DEVICE = torch.device("cpu")


# ============================================================
# START
# ============================================================

print("=" * 70)
print("DYNAMIC SWIN HAB INFERENCE TEST")
print("=" * 70)


# ============================================================
# STEP 1: CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


print("\nLoading trained Swin model...")


model = SwinHABSegmentation(
    num_channels=14,
    num_classes=2
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


if isinstance(checkpoint, dict):

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

else:

    model.load_state_dict(checkpoint)


model.to(DEVICE)

model.eval()


print("Trained Swin model loaded successfully.")
print(f"Device: {DEVICE}")


# ============================================================
# STEP 2: FIND DYNAMIC TILES
# ============================================================

tile_files = sorted(
    TILES_DIR.glob("dynamic_tile_*.npy")
)


if not tile_files:

    raise FileNotFoundError(
        "No dynamic tiles found."
    )


print(
    f"\nDynamic tiles found: "
    f"{len(tile_files)}"
)


# ============================================================
# STEP 3: RUN INFERENCE
# ============================================================

prediction_tiles = []

total_hab_pixels = 0
total_valid_pixels = 0


print("\nRunning Swin inference...")


with torch.no_grad():

    for tile_number, tile_path in enumerate(
        tile_files,
        start=1
    ):

        print(
            f"\nProcessing tile "
            f"{tile_number}/{len(tile_files)}:"
        )

        print(
            f"  {tile_path.name}"
        )


        # ----------------------------------------------------
        # LOAD TILE
        # ----------------------------------------------------

        image = np.load(tile_path)


        if image.shape != (
            14,
            TILE_SIZE,
            TILE_SIZE
        ):

            raise RuntimeError(
                f"Invalid tile shape: "
                f"{image.shape}"
            )


        image = image.astype(
            np.float32
        )


        # ----------------------------------------------------
        # HANDLE INVALID VALUES
        # ----------------------------------------------------

        image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )


        # ----------------------------------------------------
        # CONVERT TO PYTORCH
        # ----------------------------------------------------

        tensor = torch.from_numpy(
            image
        ).unsqueeze(0).to(DEVICE)


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        hab_pixels = int(
            np.sum(prediction == 1)
        )

        valid_pixels = int(
            prediction.size
        )


        hab_percentage = (
            hab_pixels /
            valid_pixels *
            100
        )


        print(
            f"  HAB pixels: "
            f"{hab_pixels}"
        )

        print(
            f"  HAB percentage: "
            f"{hab_percentage:.2f}%"
        )


        total_hab_pixels += hab_pixels

        total_valid_pixels += valid_pixels


        # ----------------------------------------------------
        # SAVE PREDICTION
        # ----------------------------------------------------

        output_name = (
            tile_path.stem +
            "_prediction.npy"
        )

        output_path = (
            OUTPUT_DIR /
            output_name
        )


        np.save(
            output_path,
            prediction
        )


        prediction_tiles.append(
            prediction
        )


        print(
            f"  Saved: "
            f"{output_path}"
        )


# ============================================================
# STEP 4: OVERALL STATISTICS
# ============================================================

overall_hab_percentage = (
    total_hab_pixels /
    total_valid_pixels *
    100
)


print("\n" + "=" * 70)
print("OVERALL SWIN RESULT")
print("=" * 70)

print(
    f"Tiles processed: "
    f"{len(prediction_tiles)}"
)

print(
    f"Total pixels: "
    f"{total_valid_pixels}"
)

print(
    f"HAB pixels predicted: "
    f"{total_hab_pixels}"
)

print(
    f"Predicted HAB percentage: "
    f"{overall_hab_percentage:.2f}%"
)


# ============================================================
# STEP 5: VERIFY PREDICTIONS
# ============================================================

print("\nChecking prediction values...")


unique_values = np.unique(
    np.concatenate(
        [
            prediction.flatten()
            for prediction in prediction_tiles
        ]
    )
)


print(
    f"Unique prediction values: "
    f"{unique_values}"
)


allowed_values = {
    0,
    1
}


if not set(
    unique_values.tolist()
).issubset(allowed_values):

    raise RuntimeError(
        "Unexpected prediction values detected."
    )


print(
    "Prediction value verification: PASSED"
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print(
    "Dynamic Sentinel-2 tiles were successfully "
    "processed by the trained Swin Transformer."
)

print(
    "The model produced pixel-level HAB predictions."
)

print("=" * 70)