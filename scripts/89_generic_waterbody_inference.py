"""
STEP 89
Generic HAB inference engine for ANY suitable waterbody.

Pipeline:
Waterbody boundary
    -> Sentinel-2 14-channel image
    -> 256x256 tiles
    -> Swin Transformer
    -> reconstructed HAB prediction
    -> waterbody mask
    -> HAB area and percentage
"""

import sys
import math
import numpy as np
import rasterio
import torch
import torch.nn.functional as F

from pathlib import Path
from pyproj import Geod


# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "swin"
    / "multilocation_corrected_best_swin_hab_model.pth"
)

sys.path.insert(0, str(PROJECT_ROOT))

from models.swin.swin_model import SwinHABSegmentation


# ---------------------------------------------------------------------
# MODEL SETTINGS
# ---------------------------------------------------------------------

DEVICE = torch.device("cpu")

TILE_SIZE = 256
NUM_CHANNELS = 14


# ---------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------

def load_model():
    """Load the trained Swin HAB segmentation model."""

    print("=" * 70)
    print("LOADING SWIN MODEL")
    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{MODEL_PATH}"
        )

    model = SwinHABSegmentation(
        num_channels=14,
        num_classes=2
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")
    print(f"Checkpoint: {MODEL_PATH}")
    print(f"Device: {DEVICE}")

    return model


# ---------------------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------------------

def normalize_14_channels(image):
    """
    Normalize the 14 Sentinel-2 channels using the same
    normalization used during training.

    First 10 channels:
        reflectance -> divide by 2

    Last 4 channels:
        indices -> [-1, 1] mapped to [0, 1]
    """

    image = image.astype(np.float32)

    # First 10 channels: reflectance
    image[:10] = image[:10] / 2.0

    # Last 4 channels: spectral indices
    image[10:] = (image[10:] + 1.0) / 2.0

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


# ---------------------------------------------------------------------
# TILE CREATION
# ---------------------------------------------------------------------

def create_tiles(image):
    """
    Split an image into 256x256 tiles.

    Edge tiles are padded with zeros.

    Returns:
        tiles
        positions
        original_height
        original_width
    """

    channels, height, width = image.shape

    tiles = []
    positions = []

    rows = math.ceil(
        height / TILE_SIZE
    )

    cols = math.ceil(
        width / TILE_SIZE
    )

    for row in range(rows):

        for col in range(cols):

            y0 = row * TILE_SIZE
            x0 = col * TILE_SIZE

            y1 = min(
                y0 + TILE_SIZE,
                height
            )

            x1 = min(
                x0 + TILE_SIZE,
                width
            )

            tile = np.zeros(
                (
                    channels,
                    TILE_SIZE,
                    TILE_SIZE
                ),
                dtype=np.float32
            )

            tile_height = y1 - y0
            tile_width = x1 - x0

            tile[
                :,
                :tile_height,
                :tile_width
            ] = image[
                :,
                y0:y1,
                x0:x1
            ]

            tiles.append(tile)

            positions.append(
                (
                    y0,
                    x0,
                    tile_height,
                    tile_width
                )
            )

    return (
        tiles,
        positions,
        height,
        width
    )


# ---------------------------------------------------------------------
# MODEL INFERENCE
# ---------------------------------------------------------------------

def predict_tiles(model, tiles):
    """
    Run Swin inference on all tiles.

    Returns binary prediction masks.
    """

    predictions = []

    print(
        f"Running inference on {len(tiles)} tiles..."
    )

    with torch.no_grad():

        for index, tile in enumerate(tiles):

            tensor = torch.from_numpy(
                tile
            ).unsqueeze(0)

            tensor = tensor.to(DEVICE)

            logits = model(tensor)

            probabilities = F.softmax(
                logits,
                dim=1
            )

            # Class 1 = HAB
            hab_probability = probabilities[:, 1]

            prediction = (
                hab_probability >= 0.5
            ).cpu().numpy()[0].astype(
                np.uint8
            )

            predictions.append(
                prediction
            )

            print(
                f"  Tile {index + 1}/{len(tiles)}"
            )

    return predictions


# ---------------------------------------------------------------------
# RECONSTRUCTION
# ---------------------------------------------------------------------

def reconstruct_prediction(
    predictions,
    positions,
    height,
    width
):
    """
    Reconstruct full-size prediction from
    256x256 tiles.
    """

    full_prediction = np.zeros(
        (
            height,
            width
        ),
        dtype=np.uint8
    )

    for prediction, position in zip(
        predictions,
        positions
    ):

        y0, x0, tile_height, tile_width = position

        full_prediction[
            y0:y0 + tile_height,
            x0:x0 + tile_width
        ] = prediction[
            :tile_height,
            :tile_width
        ]

    return full_prediction


# ---------------------------------------------------------------------
# WATERBODY MASK
# ---------------------------------------------------------------------

def apply_waterbody_mask(
    prediction,
    water_mask
):
    """
    Remove predictions outside the selected waterbody.
    """

    prediction = prediction.copy()

    prediction[
        water_mask == 0
    ] = 0

    return prediction


# ---------------------------------------------------------------------
# AREA CALCULATION
# ---------------------------------------------------------------------

def calculate_pixel_area_m2(
    transform,
    crs
):
    """
    Calculate the area of one raster pixel in square metres.

    Handles:

    1. Projected CRS
       Example: metres-based CRS.

    2. Geographic CRS
       Example: EPSG:4326 latitude/longitude.

    For geographic CRS, the four corners of one pixel
    are used with WGS84 geodesic calculation.
    """

    # ---------------------------------------------------------
    # PROJECTED CRS
    # ---------------------------------------------------------

    if crs is not None and not crs.is_geographic:

        pixel_width = abs(
            transform.a
        )

        pixel_height = abs(
            transform.e
        )

        return (
            pixel_width *
            pixel_height
        )

    # ---------------------------------------------------------
    # GEOGRAPHIC CRS
    # ---------------------------------------------------------

    geod = Geod(
        ellps="WGS84"
    )

    # Pixel corner 1
    lon1 = transform.c
    lat1 = transform.f

    # Pixel corner 2
    lon2 = (
        transform.c +
        transform.a
    )

    lat2 = (
        transform.f +
        transform.e
    )

    lons = [
        lon1,
        lon2,
        lon2,
        lon1
    ]

    lats = [
        lat1,
        lat1,
        lat2,
        lat2
    ]

    area_m2, _ = (
        geod.polygon_area_perimeter(
            lons,
            lats
        )
    )

    return abs(area_m2)


def calculate_hab_statistics(
    prediction,
    water_mask,
    transform,
    crs
):
    """
    Calculate:

    - water pixels
    - HAB pixels
    - HAB coverage percentage
    - HAB area in square metres
    - HAB area in hectares
    """

    water_pixels = np.sum(
        water_mask > 0
    )

    hab_pixels = np.sum(
        (prediction == 1) &
        (water_mask > 0)
    )

    if water_pixels == 0:

        return {
            "water_pixels": 0,
            "hab_pixels": 0,
            "hab_percentage": 0.0,
            "hab_area_m2": 0.0,
            "hab_area_ha": 0.0
        }

    # ---------------------------------------------------------
    # HAB COVERAGE
    # ---------------------------------------------------------

    hab_percentage = (
        hab_pixels /
        water_pixels
    ) * 100.0

    # ---------------------------------------------------------
    # PIXEL AREA
    # ---------------------------------------------------------

    pixel_area_m2 = calculate_pixel_area_m2(
        transform,
        crs
    )

    # ---------------------------------------------------------
    # HAB AREA
    # ---------------------------------------------------------

    hab_area_m2 = (
        hab_pixels *
        pixel_area_m2
    )

    hab_area_ha = (
        hab_area_m2 /
        10000.0
    )

    return {
        "water_pixels": int(
            water_pixels
        ),

        "hab_pixels": int(
            hab_pixels
        ),

        "hab_percentage": float(
            hab_percentage
        ),

        "hab_area_m2": float(
            hab_area_m2
        ),

        "hab_area_ha": float(
            hab_area_ha
        )
    }


# ---------------------------------------------------------------------
# MAIN GENERIC FUNCTION
# ---------------------------------------------------------------------

def run_waterbody_inference(
    image_path,
    water_mask_path,
    output_prediction_path=None
):
    """
    Run HAB detection for any compatible
    14-band waterbody image.

    Parameters
    ----------
    image_path:
        Path to 14-band Sentinel-2 image.

    water_mask_path:
        Raster water mask aligned with
        the Sentinel-2 image.

    output_prediction_path:
        Optional path for saving the
        HAB prediction raster.
    """

    print("\n")
    print("=" * 70)
    print("GENERIC WATERBODY HAB INFERENCE")
    print("=" * 70)

    image_path = Path(
        image_path
    )

    water_mask_path = Path(
        water_mask_path
    )

    # ---------------------------------------------------------
    # CHECK FILES
    # ---------------------------------------------------------

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    if not water_mask_path.exists():

        raise FileNotFoundError(
            f"Water mask not found:\n{water_mask_path}"
        )

    print(
        f"Image: {image_path}"
    )

    print(
        f"Water mask: {water_mask_path}"
    )

    # ---------------------------------------------------------
    # READ SENTINEL-2 IMAGE
    # ---------------------------------------------------------

    with rasterio.open(
        image_path
    ) as src:

        image = src.read()

        transform = src.transform

        crs = src.crs

        height = src.height

        width = src.width

    print("\nImage information:")

    print(
        f"  Channels: {image.shape[0]}"
    )

    print(
        f"  Height:   {height}"
    )

    print(
        f"  Width:    {width}"
    )

    print(
        f"  CRS:      {crs}"
    )

    if image.shape[0] != NUM_CHANNELS:

        raise ValueError(
            f"Expected 14 channels, "
            f"found {image.shape[0]}"
        )

    # ---------------------------------------------------------
    # READ WATER MASK
    # ---------------------------------------------------------

    with rasterio.open(
        water_mask_path
    ) as mask_src:

        water_mask = mask_src.read(1)

    if water_mask.shape != (
        height,
        width
    ):

        raise ValueError(
            "Water mask dimensions "
            "do not match image dimensions."
        )

    water_mask = (
        water_mask > 0
    ).astype(np.uint8)

    print(
        f"Water pixels: "
        f"{np.sum(water_mask):,}"
    )

    # ---------------------------------------------------------
    # NORMALIZE
    # ---------------------------------------------------------

    print(
        "\nNormalizing 14 channels..."
    )

    image = normalize_14_channels(
        image
    )

    # ---------------------------------------------------------
    # CREATE TILES
    # ---------------------------------------------------------

    print(
        "\nCreating 256x256 tiles..."
    )

    (
        tiles,
        positions,
        original_height,
        original_width
    ) = create_tiles(
        image
    )

    print(
        f"Tiles created: "
        f"{len(tiles)}"
    )

    # ---------------------------------------------------------
    # LOAD MODEL
    # ---------------------------------------------------------

    model = load_model()

    # ---------------------------------------------------------
    # PREDICT
    # ---------------------------------------------------------

    predictions = predict_tiles(
        model,
        tiles
    )

    # ---------------------------------------------------------
    # RECONSTRUCT
    # ---------------------------------------------------------

    print(
        "\nReconstructing full prediction..."
    )

    full_prediction = (
        reconstruct_prediction(
            predictions,
            positions,
            original_height,
            original_width
        )
    )

    # ---------------------------------------------------------
    # APPLY WATERBODY MASK
    # ---------------------------------------------------------

    print(
        "Applying waterbody mask..."
    )

    full_prediction = (
        apply_waterbody_mask(
            full_prediction,
            water_mask
        )
    )

    # ---------------------------------------------------------
    # STATISTICS
    # ---------------------------------------------------------

    statistics = (
        calculate_hab_statistics(
            full_prediction,
            water_mask,
            transform,
            crs
        )
    )

    # ---------------------------------------------------------
    # DISPLAY RESULT
    # ---------------------------------------------------------

    print("\n")

    print("=" * 70)

    print(
        "HAB DETECTION RESULT"
    )

    print("=" * 70)

    print(
        f"Water pixels : "
        f"{statistics['water_pixels']:,}"
    )

    print(
        f"HAB pixels   : "
        f"{statistics['hab_pixels']:,}"
    )

    print(
        f"HAB coverage : "
        f"{statistics['hab_percentage']:.2f}%"
    )

    print(
        f"HAB area     : "
        f"{statistics['hab_area_ha']:.4f} ha"
    )

    print(
        f"HAB area     : "
        f"{statistics['hab_area_m2']:.2f} m²"
    )

    # ---------------------------------------------------------
    # SAVE PREDICTION
    # ---------------------------------------------------------

    if output_prediction_path is not None:

        output_prediction_path = Path(
            output_prediction_path
        )

        output_prediction_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with rasterio.open(
            output_prediction_path,
            "w",
            driver="GTiff",
            height=original_height,
            width=original_width,
            count=1,
            dtype="uint8",
            crs=crs,
            transform=transform,
            nodata=0
        ) as dst:

            dst.write(
                full_prediction,
                1
            )

        print(
            "\nPrediction saved to:"
        )

        print(
            output_prediction_path
        )

    print("=" * 70)

    return (
        full_prediction,
        statistics
    )


# ---------------------------------------------------------------------
# TEST ENTRY POINT
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print(
        "\nSTEP 89 TEST"
    )

    print(
        "Testing the generic inference engine."
    )

    print()

    print(
        "Generic inference script "
        "loaded successfully."
    )

    print()

    print(
        "Next step: connect it to an actual"
    )

    print(
        "Sentinel-2 image + aligned water mask."
    )