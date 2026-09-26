from pathlib import Path
import sys

import numpy as np
import torch


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT EXISTING SWIN MODEL
# ============================================================

from models.swin.swin_model import SwinHABSegmentation


# ============================================================
# PATHS
# ============================================================

TILE_DIR = (
    PROJECT_ROOT
    / "data"
    / "tiles"
    / "dynamic_swin"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "swin"
    / "best_swin_hab_model.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "swin"
    / "dynamic"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

NUM_CHANNELS = 14
NUM_CLASSES = 2
IMAGE_SIZE = 256


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DYNAMIC SWIN TRANSFORMER INFERENCE")
    print("=" * 70)

    print()
    print(f"Device: {DEVICE}")

    # ========================================================
    # CHECK MODEL
    # ========================================================

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Trained Swin model not found:\n"
            f"{MODEL_PATH}"
        )

    print()
    print("Loading trained Swin model...")

    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = SwinHABSegmentation(
        num_channels=NUM_CHANNELS,
        num_classes=NUM_CLASSES
    )

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Detect checkpoint format
    # --------------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

    # ========================================================
    # LOAD MODEL WEIGHTS
    # ========================================================

    model.load_state_dict(
        state_dict
    )

    model.to(
        DEVICE
    )

    model.eval()

    print(
        "Trained Swin model loaded successfully."
    )

    # ========================================================
    # FIND DYNAMIC TILES
    # ========================================================

    tile_files = sorted(
        TILE_DIR.glob(
            "dynamic_tile_*.npy"
        )
    )

    if len(tile_files) == 0:

        raise FileNotFoundError(
            "No dynamic Swin tiles found in:\n"
            f"{TILE_DIR}"
        )

    print()
    print(
        f"Dynamic tiles found: {len(tile_files)}"
    )

    # ========================================================
    # PROCESS EACH TILE
    # ========================================================

    prediction_maps = []

    for tile_number, tile_path in enumerate(
        tile_files,
        start=1
    ):

        print()
        print(
            "-" * 70
        )

        print(
            f"Processing tile {tile_number}: "
            f"{tile_path.name}"
        )

        # ----------------------------------------------------
        # LOAD TILE
        # ----------------------------------------------------

        tile = np.load(
            tile_path
        )

        print(
            f"Input shape: {tile.shape}"
        )

        # ----------------------------------------------------
        # VERIFY INPUT
        # ----------------------------------------------------

        expected_shape = (
            NUM_CHANNELS,
            IMAGE_SIZE,
            IMAGE_SIZE
        )

        if tile.shape != expected_shape:

            raise ValueError(
                f"Unexpected tile shape: "
                f"{tile.shape}\n"
                f"Expected: {expected_shape}"
            )

        # ----------------------------------------------------
        # CLEAN INVALID VALUES
        # ----------------------------------------------------

        tile = np.nan_to_num(
            tile,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        tile = tile.astype(
            np.float32
        )

        # ----------------------------------------------------
        # INPUT STATISTICS
        # ----------------------------------------------------

        print(
            f"Input minimum: {tile.min():.6f}"
        )

        print(
            f"Input maximum: {tile.max():.6f}"
        )

        # ----------------------------------------------------
        # CONVERT TO TORCH
        # ----------------------------------------------------

        tensor = torch.from_numpy(
            tile
        )

        tensor = tensor.unsqueeze(
            0
        )

        tensor = tensor.to(
            DEVICE
        )

        # ====================================================
        # SWIN INFERENCE
        # ====================================================

        with torch.no_grad():

            logits = model(
                tensor
            )

        # ----------------------------------------------------
        # VERIFY OUTPUT
        # ----------------------------------------------------

        print(
            f"Model output shape: "
            f"{tuple(logits.shape)}"
        )

        # ====================================================
        # CONVERT LOGITS TO CLASS PREDICTION
        # ====================================================

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

        # ====================================================
        # CALCULATE TILE STATISTICS
        # ====================================================

        total_pixels = prediction.size

        hab_pixels = int(
            np.sum(
                prediction == 1
            )
        )

        non_hab_pixels = int(
            np.sum(
                prediction == 0
            )
        )

        hab_percentage = (
            hab_pixels
            / total_pixels
            * 100.0
        )

        print()
        print(
            f"HAB pixels:     {hab_pixels}"
        )

        print(
            f"Non-HAB pixels: {non_hab_pixels}"
        )

        print(
            f"HAB percentage: "
            f"{hab_percentage:.2f}%"
        )

        # ====================================================
        # SAVE TILE PREDICTION
        # ====================================================

        prediction_path = (
            OUTPUT_DIR
            / (
                f"{tile_path.stem}"
                f"_prediction.npy"
            )
        )

        np.save(
            prediction_path,
            prediction
        )

        print(
            f"Prediction saved:"
        )

        print(
            prediction_path
        )

        prediction_maps.append(
            prediction
        )

    # ========================================================
    # COMBINE TILE PREDICTIONS
    # ========================================================

    print()
    print("=" * 70)
    print("COMBINING TILE PREDICTIONS")
    print("=" * 70)

    # The four tiles represent a 2 × 2 arrangement:
    #
    # Tile 1 | Tile 2
    # ----------------
    # Tile 3 | Tile 4
    #
    # Each tile is 256 × 256.

    combined_prediction = np.zeros(
        (
            IMAGE_SIZE * 2,
            IMAGE_SIZE * 2
        ),
        dtype=np.uint8
    )

    combined_prediction[
        0:IMAGE_SIZE,
        0:IMAGE_SIZE
    ] = prediction_maps[0]

    combined_prediction[
        0:IMAGE_SIZE,
        IMAGE_SIZE:IMAGE_SIZE * 2
    ] = prediction_maps[1]

    combined_prediction[
        IMAGE_SIZE:IMAGE_SIZE * 2,
        0:IMAGE_SIZE
    ] = prediction_maps[2]

    combined_prediction[
        IMAGE_SIZE:IMAGE_SIZE * 2,
        IMAGE_SIZE:IMAGE_SIZE * 2
    ] = prediction_maps[3]

    # ========================================================
    # SAVE COMBINED PREDICTION
    # ========================================================

    combined_path = (
        OUTPUT_DIR
        / "dynamic_swin_prediction_512x512.npy"
    )

    np.save(
        combined_path,
        combined_prediction
    )

    # ========================================================
    # OVERALL STATISTICS
    # ========================================================

    total_pixels = (
        combined_prediction.size
    )

    total_hab_pixels = int(
        np.sum(
            combined_prediction == 1
        )
    )

    total_non_hab_pixels = int(
        np.sum(
            combined_prediction == 0
        )
    )

    overall_hab_percentage = (
        total_hab_pixels
        / total_pixels
        * 100.0
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print("=" * 70)
    print("SUCCESS")
    print("=" * 70)

    print(
        f"Tiles processed:     {len(tile_files)}"
    )

    print(
        f"Combined dimensions:  "
        f"{combined_prediction.shape}"
    )

    print(
        f"Total pixels:        "
        f"{total_pixels}"
    )

    print(
        f"Total HAB pixels:    "
        f"{total_hab_pixels}"
    )

    print(
        f"Total Non-HAB pixels:"
        f" {total_non_hab_pixels}"
    )

    print(
        f"Raw HAB percentage:   "
        f"{overall_hab_percentage:.2f}%"
    )

    print()
    print(
        "Combined prediction saved to:"
    )

    print(
        combined_path
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This is a raw model prediction."
    )

    print(
        "The actual waterbody polygon has NOT "
        "yet been applied."
    )

    print(
        "Therefore the HAB percentage above "
        "is NOT the final waterbody HAB percentage."
    )

    print(
        "The next stage will apply the geographic "
        "waterbody mask."
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()