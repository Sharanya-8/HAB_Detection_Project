from pathlib import Path
import sys

import numpy as np
import rasterio


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# FILES
# ============================================================

DYNAMIC_FILE = Path(
    "data/raw/sentinel2/"
    "Dynamic_Hussain_Sagar_14Channel_Test.tif"
)

TRAIN_TILE_DIR = Path(
    "data/tiles/train/images"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("DYNAMIC VS TRAINING DATA VALUE CHECK")
print("=" * 70)


# ============================================================
# DYNAMIC IMAGE
# ============================================================

print("\nDYNAMIC SENTINEL-2 IMAGE")
print("-" * 70)


with rasterio.open(DYNAMIC_FILE) as src:

    dynamic = src.read().astype(np.float32)


print(f"Shape: {dynamic.shape}")


for i in range(14):

    band = dynamic[i]

    finite = band[np.isfinite(band)]

    if len(finite) == 0:
        print(f"Channel {i+1}: no finite values")
        continue

    print(
        f"Channel {i+1:2d}: "
        f"min={finite.min():.6f}, "
        f"max={finite.max():.6f}, "
        f"mean={finite.mean():.6f}, "
        f"median={np.median(finite):.6f}"
    )


# ============================================================
# TRAINING TILE
# ============================================================

train_files = sorted(
    TRAIN_TILE_DIR.glob("*.npy")
)


if not train_files:

    raise FileNotFoundError(
        "No training tiles found."
    )


train_file = train_files[0]

print("\n\nTRAINING TILE")
print("-" * 70)

print(f"File: {train_file.name}")


training = np.load(
    train_file
).astype(np.float32)


print(f"Shape: {training.shape}")


for i in range(14):

    band = training[i]

    finite = band[np.isfinite(band)]

    if len(finite) == 0:
        print(f"Channel {i+1}: no finite values")
        continue

    print(
        f"Channel {i+1:2d}: "
        f"min={finite.min():.6f}, "
        f"max={finite.max():.6f}, "
        f"mean={finite.mean():.6f}, "
        f"median={np.median(finite):.6f}"
    )


# ============================================================
# COMPARISON
# ============================================================

print("\n\nCOMPARISON")
print("-" * 70)

print(
    "The values above will tell us whether the "
    "dynamic Sentinel-2 image is on the same scale "
    "as the data used during Swin training."
)


print("\n" + "=" * 70)
print("CHECK COMPLETED")
print("=" * 70)