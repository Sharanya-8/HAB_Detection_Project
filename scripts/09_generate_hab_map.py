from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import rasterio
from rasterio.warp import reproject, Resampling

from models.swin.swin_model import SwinHABSegmentation


# ============================================================
# PATHS
# ============================================================

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sentinel2"
)

WATER_MASK_PATH = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "hab_masks"
    / "Hussain_Sagar_water_mask.tif"
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
    / "maps"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

TILE_SIZE = 256

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 70)
print("HUSSAIN SAGAR HAB MAP GENERATION")
print("=" * 70)

print()
print("Device:", DEVICE)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not MODEL_PATH.exists():

    print()
    print("ERROR: Swin model checkpoint not found:")
    print(MODEL_PATH)

    sys.exit(1)


if not WATER_MASK_PATH.exists():

    print()
    print("ERROR: Hussain Sagar water mask not found:")
    print(WATER_MASK_PATH)

    sys.exit(1)


# ============================================================
# LOAD WATER MASK ONCE
# ============================================================

print()
print("Loading Hussain Sagar water mask...")

with rasterio.open(WATER_MASK_PATH) as water_src:

    water_mask = water_src.read(1)

    water_transform = water_src.transform
    water_crs = water_src.crs

print("Water mask loaded successfully.")


# ============================================================
# LOAD SWIN MODEL
# ============================================================

print()
print("Loading best Swin Transformer model...")

model = SwinHABSegmentation(
    num_channels=14,
    num_classes=2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")

print(
    "Checkpoint epoch:",
    checkpoint.get(
        "epoch",
        "unknown"
    )
)

print(
    "Validation loss:",
    checkpoint.get(
        "val_loss",
        "unknown"
    )
)


# ============================================================
# FIND PROCESSED SENTINEL-2 IMAGES
# ============================================================

image_files = sorted(
    PROCESSED_DIR.glob(
        "Hussain_Sagar_Large_*.tif"
    )
)

print()
print(
    "Processed Sentinel-2 images found:",
    len(image_files)
)

if len(image_files) == 0:

    print()
    print(
        "ERROR: No Hussain Sagar processed images found."
    )

    sys.exit(1)


# ============================================================
# PROCESS EACH DATE
# ============================================================

for image_path in image_files:

    print()
    print("=" * 70)

    print(
        "Processing:",
        image_path.name
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Extract date
    # --------------------------------------------------------

    date_part = (
        image_path.stem.replace(
            "Hussain_Sagar_Large_",
            ""
        )
    )

    output_path = (
        OUTPUT_DIR
        / f"Hussain_Sagar_HAB_Swin_{date_part}.tif"
    )


    # ========================================================
    # OPEN SENTINEL-2 IMAGE
    # ========================================================

    with rasterio.open(
        image_path
    ) as src:

        width = src.width
        height = src.height
        bands = src.count

        profile = src.profile.copy()

        print()
        print("Input raster:")
        print("Width:", width)
        print("Height:", height)
        print("Bands:", bands)
        print("CRS:", src.crs)
        print("Resolution:", src.res)


        # ----------------------------------------------------
        # Verify number of channels
        # ----------------------------------------------------

        if bands != 14:

            print()
            print(
                "ERROR: Expected 14 bands."
            )

            print(
                "Found:",
                bands
            )

            sys.exit(1)


        # ====================================================
        # ALIGN WATER MASK TO CURRENT IMAGE
        # ====================================================

        aligned_water_mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        reproject(
            source=water_mask,
            destination=aligned_water_mask,
            src_transform=water_transform,
            src_crs=water_crs,
            dst_transform=src.transform,
            dst_crs=src.crs,
            resampling=Resampling.nearest
        )


        lake_pixels = np.sum(
            aligned_water_mask == 1
        )

        print()
        print(
            "Hussain Sagar water pixels:",
            lake_pixels
        )


        # ====================================================
        # CREATE EMPTY PREDICTION MAP
        # ====================================================

        # 255 = outside water / ignored

        prediction_map = np.full(
            (height, width),
            255,
            dtype=np.uint8
        )


        # ====================================================
        # PROCESS COMPLETE 256 x 256 TILES
        # ====================================================

        rows = range(
            0,
            height - TILE_SIZE + 1,
            TILE_SIZE
        )

        cols = range(
            0,
            width - TILE_SIZE + 1,
            TILE_SIZE
        )

        total_tiles = (
            len(list(rows))
            * len(list(cols))
        )

        tile_number = 0


        for row in range(
            0,
            height - TILE_SIZE + 1,
            TILE_SIZE
        ):

            for col in range(
                0,
                width - TILE_SIZE + 1,
                TILE_SIZE
            ):

                tile_number += 1


                # ------------------------------------------------
                # Read image tile
                # ------------------------------------------------

                window = rasterio.windows.Window(
                    col,
                    row,
                    TILE_SIZE,
                    TILE_SIZE
                )

                image = src.read(
                    window=window
                )


                # ------------------------------------------------
                # Clean invalid values
                # ------------------------------------------------

                image = image.astype(
                    np.float32
                )

                image = np.nan_to_num(
                    image,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                )


                # ------------------------------------------------
                # Convert image to tensor
                # ------------------------------------------------

                tensor = torch.from_numpy(
                    image
                ).unsqueeze(0)

                tensor = tensor.to(
                    DEVICE
                )


                # ------------------------------------------------
                # Swin prediction
                # ------------------------------------------------

                with torch.no_grad():

                    output = model(
                        tensor
                    )

                    prediction = torch.argmax(
                        output,
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
                # Store prediction
                # ------------------------------------------------

                prediction_map[
                    row:row + TILE_SIZE,
                    col:col + TILE_SIZE
                ] = prediction


                print(
                    f"\rTile "
                    f"{tile_number}/{total_tiles}",
                    end=""
                )


        print()


        # ====================================================
        # APPLY HUSSAIN SAGAR WATER MASK
        # ====================================================

        # Everything outside the lake becomes 255.
        #
        # Only pixels inside Hussain Sagar are considered
        # valid HAB/non-HAB predictions.

        prediction_map[
            aligned_water_mask != 1
        ] = 255


        # ====================================================
        # SAVE MAP
        # ====================================================

        profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress="lzw",
            nodata=255
        )

        with rasterio.open(
            output_path,
            "w",
            **profile
        ) as dst:

            dst.write(
                prediction_map,
                1
            )


        # ====================================================
        # FINAL WATER-ONLY STATISTICS
        # ====================================================

        valid = prediction_map[
            prediction_map != 255
        ]

        hab_pixels = np.sum(
            valid == 1
        )

        non_hab_pixels = np.sum(
            valid == 0
        )


        print()
        print(
            "FINAL WATER-ONLY PREDICTION STATISTICS"
        )

        print(
            "Water pixels:",
            len(valid)
        )

        print(
            "Non-HAB pixels:",
            non_hab_pixels
        )

        print(
            "HAB pixels:",
            hab_pixels
        )


        if len(valid) > 0:

            hab_percentage = (
                100
                * hab_pixels
                / len(valid)
            )

            print(
                "Predicted HAB percentage:",
                f"{hab_percentage:.2f}%"
            )


        print()
        print("Saved:")
        print(output_path)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print(
    "HUSSAIN SAGAR HAB MAP GENERATION COMPLETE"
)
print("=" * 70)

print()
print("All maps saved in:")
print(OUTPUT_DIR)