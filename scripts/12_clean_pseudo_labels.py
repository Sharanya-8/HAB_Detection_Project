import rasterio
import numpy as np
import os

from scipy import ndimage


# --------------------------------------------------
# INPUT / OUTPUT
# --------------------------------------------------

input_file = "data/labels/HAB_pseudo_labels.tif"

output_file = "data/labels/HAB_clean_labels.tif"


# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------

# Minimum number of connected HAB pixels
# required to keep a region.
MIN_REGION_SIZE = 20


# --------------------------------------------------
# READ LABELS
# --------------------------------------------------

with rasterio.open(input_file) as src:

    label = src.read(1)

    profile = src.profile.copy()


# --------------------------------------------------
# SEPARATE VALID PIXELS
# --------------------------------------------------

valid = label != 255

hab_mask = label == 1


# --------------------------------------------------
# CONNECTED COMPONENT ANALYSIS
# --------------------------------------------------

structure = np.ones((3, 3), dtype=np.uint8)

labeled_regions, number_of_regions = ndimage.label(
    hab_mask,
    structure=structure
)


# --------------------------------------------------
# CALCULATE REGION SIZES
# --------------------------------------------------

region_sizes = np.bincount(
    labeled_regions.ravel()
)


# --------------------------------------------------
# REMOVE SMALL REGIONS
# --------------------------------------------------

clean_hab_mask = np.zeros_like(hab_mask, dtype=bool)


for region_id in range(1, number_of_regions + 1):

    region_size = region_sizes[region_id]

    if region_size >= MIN_REGION_SIZE:

        clean_hab_mask[labeled_regions == region_id] = True


# --------------------------------------------------
# CREATE CLEAN LABEL
# --------------------------------------------------

clean_label = np.zeros_like(label, dtype=np.uint8)

# HAB
clean_label[clean_hab_mask] = 1

# NoData
clean_label[~valid] = 255


# --------------------------------------------------
# STATISTICS
# --------------------------------------------------

valid_pixels = np.sum(valid)

original_hab_pixels = np.sum(hab_mask & valid)

clean_hab_pixels = np.sum(clean_hab_mask & valid)

removed_pixels = original_hab_pixels - clean_hab_pixels


print("HAB PSEUDO-LABEL CLEANING")
print("=========================")

print()
print("Minimum region size:", MIN_REGION_SIZE)

print()
print("Original HAB pixels:", original_hab_pixels)

print(
    "Original HAB percentage:",
    f"{original_hab_pixels / valid_pixels * 100:.2f}%"
)

print()
print("Clean HAB pixels:", clean_hab_pixels)

print(
    "Clean HAB percentage:",
    f"{clean_hab_pixels / valid_pixels * 100:.2f}%"
)

print()
print("Removed noisy pixels:", removed_pixels)

print()
print("Number of original HAB regions:", number_of_regions)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

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

    dst.write(clean_label, 1)


print()
print("Clean label file saved:")
print(output_file)