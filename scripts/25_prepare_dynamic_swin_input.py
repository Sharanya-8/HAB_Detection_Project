from pathlib import Path

import numpy as np
import rasterio


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/raw/sentinel2/"
    "Dynamic_Hussain_Sagar_14Channel_Test.tif"
)

OUTPUT_DIR = Path(
    "data/tiles/dynamic_test"
)

TILE_SIZE = 256


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("DYNAMIC SWIN INPUT PREPARATION TEST")
print("=" * 70)


# ============================================================
# STEP 1: READ GEOTIFF
# ============================================================

print("\nReading dynamic Sentinel-2 image...")

with rasterio.open(INPUT_FILE) as src:

    image = src.read()

    profile = src.profile.copy()

    print(f"Original shape: {image.shape}")
    print(f"Width: {src.width}")
    print(f"Height: {src.height}")
    print(f"Bands: {src.count}")
    print(f"CRS: {src.crs}")
    print(f"Resolution: {src.res}")


# ============================================================
# STEP 2: VERIFY 14 CHANNELS
# ============================================================

if image.shape[0] != 14:

    raise RuntimeError(
        f"Expected 14 channels, "
        f"but found {image.shape[0]}"
    )


print("\n14-channel verification: PASSED")


# ============================================================
# STEP 3: CONVERT TO FLOAT32
# ============================================================

image = image.astype(np.float32)


# ============================================================
# STEP 4: HANDLE INVALID VALUES
# ============================================================

print("\nChecking invalid values...")

nan_count = np.isnan(image).sum()
inf_count = np.isinf(image).sum()

print(f"NaN values: {nan_count}")
print(f"Infinite values: {inf_count}")


image = np.nan_to_num(
    image,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


# ============================================================
# STEP 5: ORIGINAL DIMENSIONS
# ============================================================

channels, height, width = image.shape

print("\nOriginal dimensions:")
print(f"Channels: {channels}")
print(f"Height: {height}")
print(f"Width: {width}")


# ============================================================
# STEP 6: CALCULATE NUMBER OF TILES
# ============================================================

rows = int(np.ceil(height / TILE_SIZE))
cols = int(np.ceil(width / TILE_SIZE))

padded_height = rows * TILE_SIZE
padded_width = cols * TILE_SIZE


print("\nTile configuration:")
print(f"Tile size: {TILE_SIZE} × {TILE_SIZE}")
print(f"Tile rows: {rows}")
print(f"Tile columns: {cols}")
print(f"Padded height: {padded_height}")
print(f"Padded width: {padded_width}")


# ============================================================
# STEP 7: PAD IMAGE
# ============================================================

print("\nPadding image...")


padded_image = np.pad(
    image,
    (
        (0, 0),
        (0, padded_height - height),
        (0, padded_width - width)
    ),
    mode="constant",
    constant_values=0
)


print(
    f"Padded image shape: "
    f"{padded_image.shape}"
)


# ============================================================
# STEP 8: CREATE TILES
# ============================================================

print("\nCreating 256 × 256 tiles...")


tile_count = 0


for row in range(rows):

    for col in range(cols):

        y_start = row * TILE_SIZE
        x_start = col * TILE_SIZE

        y_end = y_start + TILE_SIZE
        x_end = x_start + TILE_SIZE

        tile = padded_image[
            :,
            y_start:y_end,
            x_start:x_end
        ]


        # ----------------------------------------------------
        # VERIFY TILE
        # ----------------------------------------------------

        if tile.shape != (
            14,
            TILE_SIZE,
            TILE_SIZE
        ):

            raise RuntimeError(
                f"Invalid tile shape: {tile.shape}"
            )


        # ----------------------------------------------------
        # SAVE TILE
        # ----------------------------------------------------

        tile_count += 1

        tile_name = (
            f"dynamic_tile_{tile_count:03d}.npy"
        )

        tile_path = OUTPUT_DIR / tile_name

        np.save(
            tile_path,
            tile.astype(np.float32)
        )


        print(
            f"Created: {tile_name} "
            f"shape={tile.shape}"
        )


# ============================================================
# STEP 9: VERIFY SAVED TILES
# ============================================================

print("\nVerifying saved tiles...")


saved_tiles = sorted(
    OUTPUT_DIR.glob("*.npy")
)


print(
    f"Number of saved tiles: "
    f"{len(saved_tiles)}"
)


for tile_path in saved_tiles:

    tile = np.load(tile_path)

    if tile.shape != (
        14,
        256,
        256
    ):

        raise RuntimeError(
            f"Invalid tile: "
            f"{tile_path.name}"
        )

    if tile.dtype != np.float32:

        raise RuntimeError(
            f"Invalid dtype: "
            f"{tile.dtype}"
        )


# ============================================================
# SAMPLE TILE STATISTICS
# ============================================================

sample = np.load(
    saved_tiles[0]
)


print("\nSample tile:")
print(f"Shape: {sample.shape}")
print(f"Dtype: {sample.dtype}")
print(f"Minimum: {sample.min()}")
print(f"Maximum: {sample.max()}")
print(f"Mean: {sample.mean()}")


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print(
    "Dynamic Sentinel-2 image successfully "
    "prepared for Swin input."
)

print(
    f"Total 256 × 256 tiles: {len(saved_tiles)}"
)

print(
    "Every tile contains 14 channels."
)

print("=" * 70)