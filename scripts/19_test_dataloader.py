import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

from pathlib import Path


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
# DATASET CLASS
# ============================================================

class HABDataset(Dataset):

    def __init__(self, image_dir, label_dir):

        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

        self.image_files = sorted(
            self.image_dir.glob("*.npy")
        )

        if len(self.image_files) == 0:

            raise RuntimeError(
                f"No image tiles found in {image_dir}"
            )

    def __len__(self):

        return len(self.image_files)

    def normalize_image(self, image):

        image = image.astype(np.float32)

        normalized = np.zeros_like(image)

        # ----------------------------------------------------
        # REFLECTANCE BANDS
        # ----------------------------------------------------

        for i in range(10):

            normalized[i] = image[i] / 2.0

        # ----------------------------------------------------
        # SPECTRAL INDICES
        # ----------------------------------------------------

        for i in range(10, 14):

            normalized[i] = (
                image[i] + 1.0
            ) / 2.0

        # ----------------------------------------------------
        # HANDLE INVALID VALUES
        # ----------------------------------------------------

        normalized = np.nan_to_num(
            normalized,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        # ----------------------------------------------------
        # LIMIT RANGE
        # ----------------------------------------------------

        normalized = np.clip(
            normalized,
            0.0,
            1.0
        )

        return normalized

    def __getitem__(self, index):

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        image_file = self.image_files[index]

        image = np.load(image_file)

        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        label_file = (
            self.label_dir /
            image_file.name
        )

        if not label_file.exists():

            raise FileNotFoundError(
                f"Missing label: {label_file}"
            )

        label = np.load(label_file)

        # ----------------------------------------------------
        # NORMALIZE IMAGE
        # ----------------------------------------------------

        image = self.normalize_image(image)

        # ----------------------------------------------------
        # CONVERT TO PYTORCH
        # ----------------------------------------------------

        image = torch.from_numpy(
            image
        ).float()

        label = torch.from_numpy(
            label
        ).long()

        return image, label


# ============================================================
# NEW HUSSAIN SAGAR TRAINING DATASET
# ============================================================

train_dataset = HABDataset(
    "data/tiles/train/images",
    "data/tiles/train/masks"
)


# ============================================================
# DATALOADER
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)


# ============================================================
# PRINT DATASET INFORMATION
# ============================================================

print("HUSSAIN SAGAR DATASET / DATALOADER TEST")
print("========================================")

print()

print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Batch size:",
    train_loader.batch_size
)

print(
    "Number of batches:",
    len(train_loader)
)


# ============================================================
# LOAD ONE BATCH
# ============================================================

images, labels = next(
    iter(train_loader)
)


# ============================================================
# DISPLAY BATCH INFORMATION
# ============================================================

print()

print("Image batch shape:")
print(images.shape)

print()

print("Label batch shape:")
print(labels.shape)

print()

print("Image data type:")
print(images.dtype)

print()

print("Label data type:")
print(labels.dtype)

print()

print("Image minimum:")
print(images.min().item())

print()

print("Image maximum:")
print(images.max().item())

print()

print("Label values:")
print(torch.unique(labels))


# ============================================================
# FINAL CHECKS
# ============================================================

print()

if images.shape == (4, 14, 256, 256):

    print("Batch image shape: PASS")

else:

    print("Batch image shape: FAIL")


if labels.shape == (4, 256, 256):

    print("Batch label shape: PASS")

else:

    print("Batch label shape: FAIL")


if images.dtype == torch.float32:

    print("Image dtype: PASS")

else:

    print("Image dtype: FAIL")


if labels.dtype == torch.int64:

    print("Label dtype: PASS")

else:

    print("Label dtype: FAIL")


if torch.all(
    (images >= 0.0) &
    (images <= 1.0)
):

    print("Image normalization: PASS")

else:

    print("Image normalization: FAIL")


allowed_labels = {
    0,
    1,
    255
}


actual_labels = set(
    torch.unique(labels).tolist()
)


if actual_labels.issubset(
    allowed_labels
):

    print("Label values: PASS")

else:

    print("Label values: FAIL")


print()

print(
    "Dataset/DataLoader test completed."
)