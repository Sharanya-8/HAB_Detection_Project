import rasterio
import numpy as np
import os


# --------------------------------------------------
# INPUT FILE
# --------------------------------------------------

input_file = "data/raw/sentinel2/HAB_Sentinel2_2023_2025.tif"

# Output directory
output_dir = "data/labels"

os.makedirs(output_dir, exist_ok=True)


# --------------------------------------------------
# THRESHOLDS
# --------------------------------------------------

MNDWI_THRESHOLD = 0.05
NDCI_THRESHOLD = 0.10
FAI_THRESHOLD = 0.05


# --------------------------------------------------
# READ DATA
# --------------------------------------------------

with rasterio.open(input_file) as src:

    # Read indices
    ndwi = src.read(11)
    mndwi = src.read(12)
    ndci = src.read(13)
    fai = src.read(14)

    # Save raster profile for output
    profile = src.profile.copy()


# --------------------------------------------------
# VALID PIXELS
# --------------------------------------------------

valid = (
    np.isfinite(ndwi) &
    np.isfinite(mndwi) &
    np.isfinite(ndci) &
    np.isfinite(fai)
)


# --------------------------------------------------
# WATER MASK
# --------------------------------------------------

water_mask = (
    mndwi > MNDWI_THRESHOLD
)


# --------------------------------------------------
# HAB CANDIDATE MASK
# --------------------------------------------------

hab_indicator = (
    (ndci > NDCI_THRESHOLD) |
    (fai > FAI_THRESHOLD)
)


# --------------------------------------------------
# FINAL PSEUDO-LABEL
# --------------------------------------------------

pseudo_label = (
    valid &
    water_mask &
    hab_indicator
)

# Convert boolean mask to uint8
# 0 = Non-HAB
# 1 = HAB candidate
# 255 = NoData

label = np.zeros(pseudo_label.shape, dtype=np.uint8)

# HAB candidate
label[pseudo_label] = 1

# NoData pixels
label[~valid] = 255

# --------------------------------------------------
# STATISTICS
# --------------------------------------------------

total_valid = np.sum(valid)
water_pixels = np.sum(valid & water_mask)
hab_pixels = np.sum(valid & pseudo_label)
non_hab_pixels = np.sum(valid & ~pseudo_label)

print("HAB PSEUDO-LABEL GENERATION")
print("============================")

print()
print("Thresholds")
print("----------------------------")
print("MNDWI threshold:", MNDWI_THRESHOLD)
print("NDCI threshold:", NDCI_THRESHOLD)
print("FAI threshold:", FAI_THRESHOLD)

print()
print("Pixel statistics")
print("----------------------------")

print("Valid pixels:", total_valid)

print(
    "Water pixels:",
    water_pixels,
    f"({water_pixels / total_valid * 100:.2f}%)"
)

print(
    "HAB candidate pixels:",
    hab_pixels,
    f"({hab_pixels / total_valid * 100:.2f}%)"
)

print(
    "Non-HAB pixels:",
    non_hab_pixels,
    f"({non_hab_pixels / total_valid * 100:.2f}%)"
)


# --------------------------------------------------
# SAVE LABEL RASTER
# --------------------------------------------------

output_file = os.path.join(
    output_dir,
    "HAB_pseudo_labels.tif"
)

profile.update(
    dtype=rasterio.uint8,
    count=1,
    compress="lzw",
    nodata=255
)


with rasterio.open(
    output_file,
    "w",
    **profile
) as dst:

    dst.write(label, 1)


print()
print("Pseudo-label file saved:")
print(output_file)

print()
print("Label meaning:")
print("0 = Non-HAB")
print("1 = HAB candidate")
print("255 = NoData")