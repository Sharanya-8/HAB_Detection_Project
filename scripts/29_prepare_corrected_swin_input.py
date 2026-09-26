from pathlib import Path

import numpy as np
import rasterio


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/processed/sentinel2/"
    "Dynamic_Hussain_Sagar_14Channel_Corrected.tif"
)

OUTPUT_DIR = Path(
    "data/tiles/dynamic_corrected"
)

TILE_SIZE = 256


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Remove previous test tiles
for file in OUTPUT_DIR.glob("*.npy"):
    file.unlink()


# ============================================================
# START
# ============================================================

print("=" * 70)
print("CORRECTED DYNAMIC SWIN INPUT PREPARATION")
print("=" * 70)


# ============================================================
# READ IMAGE
# ============================================================

print("\nReading corrected Sentinel-2 image...")

with rasterio.open(INPUT_FILE) as src:

    image = src.read()

    print(f"Shape: {image.shape}")
    print(f"Width: {src.width}")
    print(f"Height: {src.height}")
    print(f"Bands: {src.count}")
    print(f"CRS: {src.crs}")
    print(f"Resolution: {src.res}")


# ============================================================
# VERIFY
# ============================================================

if image.shape[0] != 14:

    raise RuntimeError(
        f"Expected 14 channels, found {image.shape[0]}"
    )


image = image.astype(np.float32)

image = np.nan_to_num(
    image,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


# ============================================================
# DIMENSIONS
# ============================================================

channels, height, width = image.shape

rows = int(np.ceil(height / TILE_SIZE))
cols = int(np.ceil(width / TILE_SIZE))

padded_height = rows * TILE_SIZE
padded_width = cols * TILE_SIZE


print("\nTile configuration:")
print(f"Tile size: {TILE_SIZE} × {TILE_SIZE}")
print(f"Rows: {rows}")
print(f"Columns: {cols}")
print(f"Padded dimensions: {padded_height} × {padded_width}")


# ============================================================
# PAD
# ============================================================

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


# ============================================================
# CREATE TILES
# ============================================================

print("\nCreating tiles...")

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

        if tile.shape != (
            14,
            256,
            256
        ):

            raise RuntimeError(
                f"Invalid tile shape: {tile.shape}"
            )

        tile_count += 1

        output_file = (
            OUTPUT_DIR /
            f"dynamic_corrected_tile_{tile_count:03d}.npy"
        )

        np.save(
            output_file,
            tile.astype(np.float32)
        )

        print(
            f"Created: {output_file.name}"
        )


# ============================================================
# VERIFY
# ============================================================

tiles = sorted(
    OUTPUT_DIR.glob("*.npy")
)


print("\nVerifying tiles...")

for tile_file in tiles:

    tile = np.load(tile_file)

    if tile.shape != (
        14,
        256,
        256
    ):

        raise RuntimeError(
            f"Invalid tile: {tile_file.name}"
        )

    if tile.dtype != np.float32:

        raise RuntimeError(
            f"Invalid dtype: {tile.dtype}"
        )


# ============================================================
# SAMPLE
# ============================================================

sample = np.load(tiles[0])

print("\nSample tile:")
print(f"Shape: {sample.shape}")
print(f"Dtype: {sample.dtype}")
print(f"Minimum: {sample.min():.6f}")
print(f"Maximum: {sample.max():.6f}")
print(f"Mean: {sample.mean():.6f}")


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print(
    f"Created {len(tiles)} corrected Swin input tiles."
)

print(
    "All tiles contain 14 channels and are 256 × 256."
)

print("=" * 70)