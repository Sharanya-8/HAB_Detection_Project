"""
STEP 103 — CREATE GENERIC WATER MASK

Automatically finds an existing Sentinel-2 TIFF for the
selected waterbody and creates a perfectly aligned water mask.

Test waterbody:
    Hussain Sagar

The script does NOT assume a filename such as 2025-06-21.tif.
"""

import os
import glob

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize


# ============================================================
# WATERBODY
# ============================================================

WATERBODY_NAME = "Hussain_Sagar"

BOUNDARY_PATH = os.path.join(
    "results",
    "waterbody_discovery",
    "selected_waterbody",
    "Hussain_Sagar_validated_boundary.geojson"
)


# ============================================================
# FIND EXISTING HISTORICAL IMAGE
# ============================================================

HISTORICAL_DIRECTORY = os.path.join(
    "data",
    "raw",
    "sentinel2",
    "historical"
)


# We want the 2025 historical image if available.
TARGET_YEAR = "2025"


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIRECTORY = os.path.join(
    "results",
    "water_masks",
    "generic"
)

os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 103 — CREATE GENERIC WATER MASK")
print("=" * 70)

print()

print(
    f"Waterbody : {WATERBODY_NAME}"
)

print(
    f"Target year: {TARGET_YEAR}"
)

print()


# ============================================================
# CHECK BOUNDARY
# ============================================================

if not os.path.exists(BOUNDARY_PATH):

    print("ERROR: Boundary file does not exist.")

    raise SystemExit


# ============================================================
# FIND HISTORICAL TIFFS
# ============================================================

search_pattern = os.path.join(
    HISTORICAL_DIRECTORY,
    f"*{TARGET_YEAR}*.tif"
)

candidate_images = sorted(
    glob.glob(search_pattern)
)


print("=" * 70)
print("SEARCHING FOR SENTINEL-2 IMAGE")
print("=" * 70)

print()

print(
    f"Search pattern:"
)

print(
    search_pattern
)

print()

print(
    f"Images found: {len(candidate_images)}"
)

print()


if not candidate_images:

    print(
        "ERROR: No historical Sentinel-2 image "
        f"was found for {TARGET_YEAR}."
    )

    raise SystemExit


# ============================================================
# SHOW CANDIDATES
# ============================================================

for index, path in enumerate(
    candidate_images,
    start=1
):

    print(
        f"{index:02d}. "
        f"{os.path.basename(path)}"
    )

print()


# ============================================================
# SELECT FIRST IMAGE
# ============================================================

IMAGE_PATH = candidate_images[0]


print("=" * 70)
print("SELECTED REFERENCE IMAGE")
print("=" * 70)

print()

print(
    os.path.abspath(
        IMAGE_PATH
    )
)

print()


# ============================================================
# READ IMAGE
# ============================================================

with rasterio.open(
    IMAGE_PATH
) as src:

    image_width = src.width
    image_height = src.height

    image_transform = src.transform
    image_crs = src.crs

    image_bounds = src.bounds


print("=" * 70)
print("REFERENCE IMAGE")
print("=" * 70)

print()

print(
    f"Width  : {image_width}"
)

print(
    f"Height : {image_height}"
)

print(
    f"CRS    : {image_crs}"
)

print(
    f"Bounds : {image_bounds}"
)

print()


# ============================================================
# READ BOUNDARY
# ============================================================

gdf = gpd.read_file(
    BOUNDARY_PATH
)


if gdf.empty:

    print(
        "ERROR: Boundary contains no features."
    )

    raise SystemExit


print("=" * 70)
print("BOUNDARY")
print("=" * 70)

print()

print(
    f"Features : {len(gdf)}"
)

print(
    f"CRS      : {gdf.crs}"
)

print()


# ============================================================
# REPROJECT BOUNDARY
# ============================================================

if gdf.crs is None:

    print(
        "ERROR: Boundary CRS is missing."
    )

    raise SystemExit


boundary_gdf = gdf.to_crs(
    image_crs
)


# ============================================================
# FIX INVALID GEOMETRIES
# ============================================================

boundary_gdf["geometry"] = (
    boundary_gdf.geometry.buffer(0)
)


# ============================================================
# CREATE RASTERIZATION SHAPES
# ============================================================

shapes = []

for geometry in boundary_gdf.geometry:

    if geometry is None:
        continue

    if geometry.is_empty:
        continue

    shapes.append(
        (
            geometry,
            1
        )
    )


if not shapes:

    print(
        "ERROR: No valid boundary geometry."
    )

    raise SystemExit


# ============================================================
# RASTERIZE
# ============================================================

print(
    "Creating aligned water mask..."
)

print()


water_mask = rasterize(
    shapes=shapes,

    out_shape=(
        image_height,
        image_width
    ),

    transform=image_transform,

    fill=0,

    dtype="uint8"
)


# ============================================================
# STATISTICS
# ============================================================

water_pixels = int(
    np.sum(
        water_mask == 1
    )
)

total_pixels = (
    image_width
    * image_height
)

water_percentage = (
    water_pixels
    / total_pixels
    * 100
)


print("=" * 70)
print("WATER MASK STATISTICS")
print("=" * 70)

print()

print(
    f"Total pixels : {total_pixels:,}"
)

print(
    f"Water pixels : {water_pixels:,}"
)

print(
    f"Water coverage: "
    f"{water_percentage:.2f}%"
)

print()


# ============================================================
# OUTPUT NAME
# ============================================================

image_name = os.path.splitext(
    os.path.basename(
        IMAGE_PATH
    )
)[0]


OUTPUT_PATH = os.path.join(
    OUTPUT_DIRECTORY,
    f"{WATERBODY_NAME}_{image_name}_water_mask.tif"
)


# ============================================================
# SAVE
# ============================================================

profile = {

    "driver": "GTiff",

    "height": image_height,

    "width": image_width,

    "count": 1,

    "dtype": "uint8",

    "crs": image_crs,

    "transform": image_transform,

    "nodata": 0
}


with rasterio.open(
    OUTPUT_PATH,
    "w",
    **profile
) as dst:

    dst.write(
        water_mask,
        1
    )


# ============================================================
# VALIDATE
# ============================================================

with rasterio.open(
    OUTPUT_PATH
) as check:

    saved_mask = check.read(1)

    print("=" * 70)
    print("OUTPUT VALIDATION")
    print("=" * 70)

    print()

    print(
        f"Mask shape : "
        f"{saved_mask.shape}"
    )

    print(
        f"Mask CRS   : "
        f"{check.crs}"
    )

    print(
        f"Mask values: "
        f"{np.unique(saved_mask)}"
    )

    print()


# ============================================================
# COMPLETE
# ============================================================

print("=" * 70)
print("STEP 103 COMPLETE")
print("=" * 70)

print()

print(
    "Water mask saved:"
)

print(
    os.path.abspath(
        OUTPUT_PATH
    )
)

print()