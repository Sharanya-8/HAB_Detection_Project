from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PATHS
# ============================================================

TILES_DIR = PROJECT_ROOT / "data" / "tiles"

IMAGE_DIR = TILES_DIR / "images"
MASK_DIR = TILES_DIR / "masks"

SPLIT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_val_split"
    / "multilocation_train_val_split.csv"
)


# ============================================================
# CHANNEL INFORMATION
# ============================================================

CHANNEL_NAMES = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
    "NDWI",
    "MNDWI",
    "NDCI",
    "FAI"
]


# ============================================================
# NORMALIZATION
# ============================================================

REFLECTANCE_MIN = 0.0
REFLECTANCE_MAX = 2.0

INDEX_RANGES = {
    "NDWI": (-1.0, 1.0),
    "MNDWI": (-1.0, 1.0),
    "NDCI": (-1.0, 1.0),
    "FAI": (-1.0, 1.0)
}


def normalize_image(image):

    image = image.astype(np.float32)

    normalized = np.zeros_like(image)

    # --------------------------------------------------------
    # Reflectance bands
    # --------------------------------------------------------

    for i in range(10):

        normalized[i] = (
            image[i] - REFLECTANCE_MIN
        ) / (
            REFLECTANCE_MAX - REFLECTANCE_MIN
        )

    # --------------------------------------------------------
    # Spectral index bands
    # --------------------------------------------------------

    for i in range(10, 14):

        name = CHANNEL_NAMES[i]

        min_value, max_value = INDEX_RANGES[name]

        normalized[i] = (
            image[i] - min_value
        ) / (
            max_value - min_value
        )

    # --------------------------------------------------------
    # Replace invalid values
    # --------------------------------------------------------

    normalized = np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    # --------------------------------------------------------
    # Keep values in [0, 1]
    # --------------------------------------------------------

    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )

    return normalized


# ============================================================
# DATASET CLASS
# ============================================================

class MultiWaterbodyHABDataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_dir,
        mask_dir
    ):

        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        tile_id = str(row["tile_id"])

        # ----------------------------------------------------
        # Locate image and mask
        # ----------------------------------------------------

        image_path = (
            self.image_dir
            / f"{tile_id}.npy"
        )

        mask_path = (
            self.mask_dir
            / f"{tile_id}.npy"
        )

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        image = np.load(image_path)

        mask = np.load(mask_path)

        # ----------------------------------------------------
        # Normalize image
        # ----------------------------------------------------

        image = normalize_image(image)

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        image = torch.from_numpy(
            image
        ).float()

        mask = torch.from_numpy(
            mask
        ).long()

        return image, mask


# ============================================================
# LOAD SPLIT FILE
# ============================================================

print("=" * 70)
print("STEP 74 - PYTORCH DATASET / DATALOADER TEST")
print("=" * 70)

print()
print("Loading split file:")
print(SPLIT_FILE)

df = pd.read_csv(
    SPLIT_FILE
)

print()
print(
    f"Total tiles in split file: {len(df)}"
)


# ============================================================
# CREATE TRAIN / VALIDATION DATASETS
# ============================================================

train_df = df[
    df["split"] == "train"
].copy()

val_df = df[
    df["split"] == "val"
].copy()


print()
print("Dataset sizes:")
print(
    f"Training   : {len(train_df)}"
)

print(
    f"Validation : {len(val_df)}"
)


train_dataset = MultiWaterbodyHABDataset(
    train_df,
    IMAGE_DIR,
    MASK_DIR
)

val_dataset = MultiWaterbodyHABDataset(
    val_df,
    IMAGE_DIR,
    MASK_DIR
)


# ============================================================
# CREATE DATALOADERS
# ============================================================

BATCH_SIZE = 1

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print()
print("DataLoader sizes:")
print(
    f"Training batches   : {len(train_loader)}"
)

print(
    f"Validation batches : {len(val_loader)}"
)


# ============================================================
# TEST ONE SAMPLE
# ============================================================

print()
print("=" * 70)
print("TESTING ONE TRAINING BATCH")
print("=" * 70)

images, masks = next(
    iter(train_loader)
)


print()
print("Image tensor shape:")
print(images.shape)

print()
print("Mask tensor shape:")
print(masks.shape)

print()
print("Image dtype:")
print(images.dtype)

print()
print("Mask dtype:")
print(masks.dtype)

print()
print("Image minimum:")
print(images.min().item())

print()
print("Image maximum:")
print(images.max().item())

print()
print("Mask unique values:")
print(torch.unique(masks))


# ============================================================
# CHECK SHAPES
# ============================================================

print()
print("=" * 70)
print("DATASET CHECKS")
print("=" * 70)


if images.shape == (1, 14, 256, 256):

    print(
        "Image shape check : PASS"
    )

else:

    print(
        "Image shape check : FAIL"
    )


if masks.shape == (1, 256, 256):

    print(
        "Mask shape check  : PASS"
    )

else:

    print(
        "Mask shape check  : FAIL"
    )


if images.dtype == torch.float32:

    print(
        "Image dtype check : PASS"
    )

else:

    print(
        "Image dtype check : FAIL"
    )


if masks.dtype == torch.int64:

    print(
        "Mask dtype check  : PASS"
    )

else:

    print(
        "Mask dtype check  : FAIL"
    )


allowed_values = {
    0,
    1,
    255
}

actual_values = set(
    torch.unique(masks).tolist()
)

print()

print(
    "Actual mask values:",
    sorted(actual_values)
)


if actual_values.issubset(
    allowed_values
):

    print(
        "Mask value check  : PASS"
    )

else:

    print(
        "Mask value check  : FAIL"
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("STEP 74 COMPLETE")
print("=" * 70)

print()
print(
    "PyTorch Dataset and DataLoader are ready."
)

print()
print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(val_dataset)
)