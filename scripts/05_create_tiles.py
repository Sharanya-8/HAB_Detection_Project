# ============================================================
# STEP 5: CREATE MULTI-DATE TRAINING TILES
# HUSSAIN SAGAR LARGE-AOI DATASET
# ============================================================

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sentinel2"
)

LABEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "hab_masks"
)

IMAGE_TILE_DIR = (
    PROJECT_ROOT
    / "data"
    / "tiles"
    / "images"
)

MASK_TILE_DIR = (
    PROJECT_ROOT
    / "data"
    / "tiles"
    / "masks"
)


# ============================================================
# SETTINGS
# ============================================================

TILE_SIZE = 256

# No overlap between tiles.
STRIDE = 256

# A tile must contain at least 10% Hussain Sagar water.
MIN_WATER_RATIO = 0.10


DATES = [
    "2023_01_08",
    "2023_02_12",
    "2023_03_24",
    "2023_05_28",
    "2023_10_05",
    "2024_01_28",
    "2024_03_08",
    "2024_04_22",
    "2024_06_26",
    "2025_01_22",
    "2025_02_26",
    "2025_04_04",
    "2025_06_01",
    "2025_11_13",
    "2025_12_28",
]


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def prepare_directories():

    IMAGE_TILE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MASK_TILE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# REMOVE OLD TOP-LEVEL TILES
# ============================================================

def clear_old_tiles():

    print()
    print("Removing old tiles...")

    for file in IMAGE_TILE_DIR.glob("*.npy"):
        file.unlink()

    for file in MASK_TILE_DIR.glob("*.npy"):
        file.unlink()

    print("Old tiles removed.")


# ============================================================
# CREATE TILES
# ============================================================

def create_tiles():

    print("=" * 80)
    print("HUSSAIN SAGAR MULTI-DATE TILE CREATION")
    print("=" * 80)

    prepare_directories()
    clear_old_tiles()

    total_tiles = 0
    total_hab_pixels = 0
    total_water_pixels = 0

    successful_dates = 0

    # ========================================================
    # PROCESS EACH DATE
    # ========================================================

    for date in DATES:

        print()
        print("-" * 80)
        print(f"PROCESSING DATE: {date}")
        print("-" * 80)

        image_file = (
            PROCESSED_DIR
            / f"Hussain_Sagar_Large_{date}.tif"
        )

        label_file = (
            LABEL_DIR
            / f"Hussain_Sagar_HAB_{date}.tif"
        )

        if not image_file.exists():

            print(
                f"WARNING: Image not found:\n"
                f"{image_file}"
            )

            continue

        if not label_file.exists():

            print(
                f"WARNING: Label not found:\n"
                f"{label_file}"
            )

            continue

        # ====================================================
        # OPEN IMAGE
        # ====================================================

        with rasterio.open(image_file) as src:

            image = src.read()

            image_height = src.height
            image_width = src.width

            image_transform = src.transform
            image_crs = src.crs

        # ====================================================
        # OPEN LABEL
        # ====================================================

        with rasterio.open(label_file) as label_src:

            label = label_src.read(1)

            label_transform = label_src.transform
            label_crs = label_src.crs

        # ====================================================
        # ALIGN LABEL TO IMAGE
        # ====================================================

        if (
            label.shape !=
            (image_height, image_width)
            or label_transform != image_transform
            or label_crs != image_crs
        ):

            print("Aligning label to image...")

            aligned_label = np.full(
                (image_height, image_width),
                255,
                dtype=np.uint8
            )

            reproject(
                source=label,
                destination=aligned_label,
                src_transform=label_transform,
                src_crs=label_crs,
                dst_transform=image_transform,
                dst_crs=image_crs,
                resampling=Resampling.nearest,
                src_nodata=255,
                dst_nodata=255
            )

            label = aligned_label

        # ====================================================
        # IMAGE VALIDITY
        # ====================================================

        image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        ).astype(np.float32)

        # ====================================================
        # TILE COUNTER
        # ====================================================

        date_tile_count = 0

        # ====================================================
        # LOOP THROUGH FULL 256 x 256 TILES
        # ====================================================

        for row in range(
            0,
            image_height - TILE_SIZE + 1,
            STRIDE
        ):

            for col in range(
                0,
                image_width - TILE_SIZE + 1,
                STRIDE
            ):

                image_tile = image[
                    :,
                    row:row + TILE_SIZE,
                    col:col + TILE_SIZE
                ]

                mask_tile = label[
                    row:row + TILE_SIZE,
                    col:col + TILE_SIZE
                ]

                # =================================================
                # WATER PIXELS
                # =================================================

                water_pixels = (
                    mask_tile != 255
                )

                water_count = np.count_nonzero(
                    water_pixels
                )

                water_ratio = (
                    water_count
                    / (TILE_SIZE * TILE_SIZE)
                )

                # =================================================
                # SKIP TILES WITH TOO LITTLE WATER
                # =================================================

                if water_ratio < MIN_WATER_RATIO:
                    continue

                # =================================================
                # SAVE TILE
                # =================================================

                tile_number = date_tile_count + 1

                tile_name = (
                    f"Hussain_Sagar_{date}"
                    f"_tile_{tile_number:03d}.npy"
                )

                image_output = (
                    IMAGE_TILE_DIR
                    / tile_name
                )

                mask_output = (
                    MASK_TILE_DIR
                    / tile_name
                )

                np.save(
                    image_output,
                    image_tile
                )

                np.save(
                    mask_output,
                    mask_tile
                )

                # =================================================
                # STATISTICS
                # =================================================

                hab_pixels = np.count_nonzero(
                    mask_tile == 1
                )

                total_hab_pixels += hab_pixels
                total_water_pixels += water_count

                date_tile_count += 1
                total_tiles += 1

        print(
            f"Image dimensions: "
            f"{image_width} x {image_height}"
        )

        print(
            f"Tiles generated: "
            f"{date_tile_count}"
        )

        successful_dates += 1

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("MULTI-DATE TILE CREATION COMPLETE")
    print("=" * 80)

    print(
        f"Dates processed       : "
        f"{successful_dates}/{len(DATES)}"
    )

    print(
        f"Total image tiles     : "
        f"{total_tiles}"
    )

    print(
        f"Total mask tiles      : "
        f"{len(list(MASK_TILE_DIR.glob('*.npy')))}"
    )

    print(
        f"Total water pixels    : "
        f"{total_water_pixels:,}"
    )

    print(
        f"Total HAB pixels      : "
        f"{total_hab_pixels:,}"
    )

    # ========================================================
    # VERIFY IMAGE/MASK MATCHING
    # ========================================================

    image_files = sorted(
        IMAGE_TILE_DIR.glob("*.npy")
    )

    mask_files = sorted(
        MASK_TILE_DIR.glob("*.npy")
    )

    print()
    print("VERIFICATION")
    print("-" * 40)

    print(
        f"Image tiles: "
        f"{len(image_files)}"
    )

    print(
        f"Mask tiles : "
        f"{len(mask_files)}"
    )

    if (
        len(image_files) !=
        len(mask_files)
    ):

        print(
            "STATUS: FAILED - tile counts do not match"
        )

        return False

    image_names = [
        file.name
        for file in image_files
    ]

    mask_names = [
        file.name
        for file in mask_files
    ]

    if image_names != mask_names:

        print(
            "STATUS: FAILED - image/mask filenames do not match"
        )

        return False

    # ========================================================
    # SAMPLE TILE
    # ========================================================

    if len(image_files) > 0:

        sample_image = np.load(
            image_files[0]
        )

        sample_mask = np.load(
            mask_files[0]
        )

        print()
        print(
            "Sample image shape:",
            sample_image.shape
        )

        print(
            "Sample image dtype:",
            sample_image.dtype
        )

        print(
            "Sample mask shape:",
            sample_mask.shape
        )

        print(
            "Sample mask dtype:",
            sample_mask.dtype
        )

        print(
            "Sample mask values:",
            np.unique(sample_mask)
        )

    print()
    print("STATUS: SUCCESS")

    return True


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    create_tiles()