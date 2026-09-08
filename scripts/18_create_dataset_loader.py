import numpy as np
import torch
from pathlib import Path


# --------------------------------------------------
# DIRECTORIES
# --------------------------------------------------

IMAGE_DIR = Path("data/patches/train")
LABEL_DIR = Path("data/labels/train")


# --------------------------------------------------
# NORMALIZATION RANGES
# --------------------------------------------------

# Reflectance channels
REFLECTANCE_MIN = 0.0
REFLECTANCE_MAX = 2.0

# Index channels
INDEX_RANGES = {
    "NDWI": (-1.0, 1.0),
    "MNDWI": (-1.0, 1.0),
    "NDCI": (-1.0, 1.0),
    "FAI": (-1.0, 1.0)
}


# --------------------------------------------------
# CHANNEL NAMES
# --------------------------------------------------

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


# --------------------------------------------------
# NORMALIZATION FUNCTION
# --------------------------------------------------

def normalize_image(image):

    image = image.astype(np.float32)

    normalized = np.zeros_like(image)


    # Reflectance bands
    for i in range(10):

        normalized[i] = (
            image[i] - REFLECTANCE_MIN
        ) / (
            REFLECTANCE_MAX - REFLECTANCE_MIN
        )


    # Index bands
    for i in range(10, 14):

        name = CHANNEL_NAMES[i]

        min_value, max_value = INDEX_RANGES[name]

        normalized[i] = (
            image[i] - min_value
        ) / (
            max_value - min_value
        )

# Replace invalid NaN and infinite values
# with 0 before sending data to the model

        normalized = np.nan_to_num(
            normalized,
            nan=0.0,
            posinf=1.0,
        neginf=0.0
    )


    # Keep values inside [0, 1]
    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )


    return normalized


# --------------------------------------------------
# LOAD ONE SAMPLE
# --------------------------------------------------

image_files = sorted(
    IMAGE_DIR.glob("*.npy")
)


if len(image_files) == 0:

    raise RuntimeError(
        "No training patches found."
    )


# Pick first patch
image_file = image_files[0]

label_file = LABEL_DIR / image_file.name


print("DATASET LOADER TEST")
print("===================")

print()
print("Image file:")
print(image_file)

print()
print("Label file:")
print(label_file)


# --------------------------------------------------
# LOAD
# --------------------------------------------------

image = np.load(image_file)

label = np.load(label_file)


print()
print("Original image shape:")
print(image.shape)

print()
print("Original label shape:")
print(label.shape)


# --------------------------------------------------
# NORMALIZE
# --------------------------------------------------

normalized_image = normalize_image(image)


# --------------------------------------------------
# CONVERT TO PYTORCH
# --------------------------------------------------

image_tensor = torch.from_numpy(
    normalized_image
).float()


# PyTorch segmentation labels
# Keep 255 as ignore index

label_tensor = torch.from_numpy(
    label
).long()


# --------------------------------------------------
# OUTPUT
# --------------------------------------------------

print()
print("PyTorch image shape:")
print(image_tensor.shape)

print()
print("PyTorch label shape:")
print(label_tensor.shape)

print()
print("Image data type:")
print(image_tensor.dtype)

print()
print("Label data type:")
print(label_tensor.dtype)


print()
print("Normalized image range:")
print(
    image_tensor.min().item(),
    "to",
    image_tensor.max().item()
)


print()
print("Unique label values:")
print(
    torch.unique(label_tensor)
)


print()
print("Number of image channels:")
print(image_tensor.shape[0])


# --------------------------------------------------
# FINAL CHECKS
# --------------------------------------------------

if image_tensor.shape == (14, 256, 256):

    print()
    print("Image shape check: PASS")

else:

    print()
    print("Image shape check: FAIL")


if label_tensor.shape == (256, 256):

    print("Label shape check: PASS")

else:

    print("Label shape check: FAIL")


if image_tensor.shape[0] == 14:

    print("Channel count check: PASS")

else:

    print("Channel count check: FAIL")


allowed_labels = {0, 1, 255}

actual_labels = set(
    torch.unique(label_tensor).tolist()
)


if actual_labels.issubset(allowed_labels):

    print("Label value check: PASS")

else:

    print("Label value check: FAIL")


print()
print("Dataset loader test completed.")