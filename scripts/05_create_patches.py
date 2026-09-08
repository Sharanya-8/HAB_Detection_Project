import rasterio
import numpy as np
from pathlib import Path

# Input Sentinel-2 GeoTIFF
input_file = "data/raw/sentinel2/HAB_Sentinel2_2023_2025.tif"

# Output folder
output_dir = Path("data/patches")
output_dir.mkdir(parents=True, exist_ok=True)

# Patch size
PATCH_SIZE = 256

with rasterio.open(input_file) as src:

    print("Image width:", src.width)
    print("Image height:", src.height)
    print("Number of bands:", src.count)
    print("Patch size:", PATCH_SIZE)

    # Read all 14 bands
    image = src.read()

    height = src.height
    width = src.width

    patch_count = 0

    # Create 256 × 256 patches
    for row in range(0, height - PATCH_SIZE + 1, PATCH_SIZE):

        for col in range(0, width - PATCH_SIZE + 1, PATCH_SIZE):

            patch = image[
                :,
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE
            ]

            # Skip patches containing too many invalid values
            valid_pixels = np.isfinite(patch).sum()

            total_pixels = patch.size

            valid_ratio = valid_pixels / total_pixels

            if valid_ratio < 0.5:
                continue

            patch_count += 1

            output_file = output_dir / f"patch_{patch_count:04d}.npy"

            np.save(output_file, patch)

            print(f"Saved: {output_file}")

print()
print("Patch creation completed.")
print("Total patches:", patch_count)