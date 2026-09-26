from pathlib import Path
import sys
import json

import numpy as np
import rasterio
from rasterio.features import geometry_mask

import torch


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# MODEL IMPORT
# ============================================================

from models.swin.swin_model import SwinHABSegmentation


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "swin"
    / "best_swin_hab_model.pth"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# LOAD MODEL
# ============================================================

def load_swin_model():

    model = SwinHABSegmentation(
        num_channels=14,
        num_classes=2
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(
            checkpoint
        )

    model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# PREPARE TILES
# ============================================================

def create_tiles(image):

    channels, height, width = image.shape

    tile_size = 256

    padded_height = (
        (height + tile_size - 1)
        // tile_size
        * tile_size
    )

    padded_width = (
        (width + tile_size - 1)
        // tile_size
        * tile_size
    )

    padded = np.zeros(
        (
            channels,
            padded_height,
            padded_width
        ),
        dtype=np.float32
    )

    padded[
        :,
        :height,
        :width
    ] = image

    tiles = []

    positions = []

    for row in range(
        0,
        padded_height,
        tile_size
    ):

        for col in range(
            0,
            padded_width,
            tile_size
        ):

            tile = padded[
                :,
                row:row + tile_size,
                col:col + tile_size
            ]

            tiles.append(tile)

            positions.append(
                (
                    row,
                    col
                )
            )

    return (
        tiles,
        positions,
        height,
        width
    )


# ============================================================
# RUN SWIN INFERENCE
# ============================================================

def run_swin_inference(
    sentinel_file,
    boundary_geometry
):

    print(
        "Loading Sentinel-2 image..."
    )

    with rasterio.open(
        sentinel_file
    ) as src:

        image = src.read().astype(
            np.float32
        )

        transform = src.transform

        height = src.height

        width = src.width

        profile = src.profile.copy()

    # --------------------------------------------------------
    # CLEAN DATA
    # --------------------------------------------------------

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # --------------------------------------------------------
    # CREATE TILES
    # --------------------------------------------------------

    (
        tiles,
        positions,
        original_height,
        original_width
    ) = create_tiles(image)

    print(
        f"Created {len(tiles)} tiles."
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print(
        "Loading trained Swin Transformer..."
    )

    model = load_swin_model()

    # --------------------------------------------------------
    # RECONSTRUCT PREDICTION
    # --------------------------------------------------------

    padded_height = (
        max(
            row
            for row, col in positions
        )
        + 256
    )

    padded_width = (
        max(
            col
            for row, col in positions
        )
        + 256
    )

    prediction = np.zeros(
        (
            padded_height,
            padded_width
        ),
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # TILE INFERENCE
    # --------------------------------------------------------

    for index, (
        tile,
        (row, col)
    ) in enumerate(
        zip(
            tiles,
            positions
        ),
        start=1
    ):

        tensor = torch.from_numpy(
            tile
        ).unsqueeze(0)

        tensor = tensor.to(
            DEVICE
        )

        with torch.no_grad():

            output = model(
                tensor
            )

            predicted = torch.argmax(
                output,
                dim=1
            )

        predicted = (
            predicted
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        prediction[
            row:row + 256,
            col:col + 256
        ] = predicted

        print(
            f"  Tile {index}/{len(tiles)} complete"
        )

    # --------------------------------------------------------
    # CROP TO ORIGINAL SIZE
    # --------------------------------------------------------

    prediction = prediction[
        :original_height,
        :original_width
    ]

    # --------------------------------------------------------
    # CREATE WATERBODY MASK
    # --------------------------------------------------------

    print(
        "Applying actual waterbody boundary..."
    )

    water_mask = geometry_mask(
        [
            boundary_geometry
        ],
        out_shape=(
            original_height,
            original_width
        ),
        transform=transform,
        invert=True
    )

    # --------------------------------------------------------
    # CALCULATE STATISTICS
    # --------------------------------------------------------

    water_predictions = prediction[
        water_mask
    ]

    hab_pixels = int(
        np.sum(
            water_predictions == 1
        )
    )

    non_hab_pixels = int(
        np.sum(
            water_predictions == 0
        )
    )

    total_pixels = (
        hab_pixels
        + non_hab_pixels
    )

    if total_pixels > 0:

        hab_percentage = (
            hab_pixels
            / total_pixels
            * 100.0
        )

    else:

        hab_percentage = 0.0

    # --------------------------------------------------------
    # CREATE DISPLAY MAP
    # --------------------------------------------------------

    display_map = np.zeros(
        (
            original_height,
            original_width
        ),
        dtype=np.uint8
    )

    # 0 = outside waterbody
    # 1 = Non-HAB
    # 2 = HAB

    display_map[
        water_mask
        & (prediction == 0)
    ] = 1

    display_map[
        water_mask
        & (prediction == 1)
    ] = 2

    return {
        "prediction": display_map,
        "water_mask": water_mask,
        "hab_pixels": hab_pixels,
        "non_hab_pixels": non_hab_pixels,
        "water_pixels": total_pixels,
        "hab_percentage": hab_percentage,
        "transform": transform,
        "profile": profile
    }


# ============================================================
# TEST MODE
# ============================================================

def main():

    sentinel_file = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "sentinel2"
        / "Dynamic_Hussain_Sagar_2025-01-02_14Channel.tif"
    )

    boundary_file = (
        PROJECT_ROOT
        / "data"
        / "labels"
        / "hab_masks"
        / "Hussain_Sagar_boundary.geojson"
    )

    if not sentinel_file.exists():

        raise FileNotFoundError(
            f"Sentinel-2 file not found:\n"
            f"{sentinel_file}"
        )

    if not boundary_file.exists():

        raise FileNotFoundError(
            f"Boundary file not found:\n"
            f"{boundary_file}"
        )

    with open(
        boundary_file,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if data["type"] == "FeatureCollection":

        geometry = data[
            "features"
        ][0][
            "geometry"
        ]

    elif data["type"] == "Feature":

        geometry = data[
            "geometry"
        ]

    else:

        geometry = data

    print("=" * 70)
    print("DASHBOARD SWIN DETECTION TEST")
    print("=" * 70)

    result = run_swin_inference(
        sentinel_file,
        geometry
    )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        f"Waterbody pixels: "
        f"{result['water_pixels']}"
    )

    print(
        f"Non-HAB pixels: "
        f"{result['non_hab_pixels']}"
    )

    print(
        f"HAB pixels: "
        f"{result['hab_pixels']}"
    )

    print(
        f"Estimated HAB: "
        f"{result['hab_percentage']:.2f}%"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()