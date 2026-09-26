import os
import rasterio
import geopandas as gpd
from rasterio.features import rasterize


print("=" * 70)
print("STEP 115 — CREATE OSMAN SAGAR WATER MASK")
print("=" * 70)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

IMAGE_PATH = (
    "data/raw/sentinel2/second_waterbody/"
    "Osman_Sagar_2025_14Channel.tif"
)

GEOMETRY_PATH = (
    "results/waterbody_discovery/second_waterbody/"
    "selected_waterbody_geometry.geojson"
)

OUTPUT_DIR = "results/water_masks/second_waterbody"

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "Osman_Sagar_water_mask_2025-01-02.tif"
)


# ------------------------------------------------------------
# CREATE OUTPUT DIRECTORY
# ------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# LOAD SENTINEL-2 IMAGE
# ------------------------------------------------------------

print("\nLoading Sentinel-2 image...")

with rasterio.open(IMAGE_PATH) as src:

    image_width = src.width
    image_height = src.height
    image_transform = src.transform
    image_crs = src.crs

    print(f"Image size : {image_width} x {image_height}")
    print(f"CRS        : {image_crs}")


# ------------------------------------------------------------
# LOAD OSMAN SAGAR GEOMETRY
# ------------------------------------------------------------

print("\nLoading Osman Sagar geometry...")

gdf = gpd.read_file(GEOMETRY_PATH)

print(f"Geometry type : {gdf.geometry.iloc[0].geom_type}")
print(f"Geometry CRS  : {gdf.crs}")


# ------------------------------------------------------------
# REPROJECT TO IMAGE CRS
# ------------------------------------------------------------

if gdf.crs != image_crs:

    print("\nReprojecting geometry to image CRS...")

    gdf = gdf.to_crs(image_crs)


# ------------------------------------------------------------
# CREATE RASTER MASK
# ------------------------------------------------------------

print("\nCreating raster water mask...")

shapes = [
    (geometry, 1)
    for geometry in gdf.geometry
    if geometry is not None
    and not geometry.is_empty
]

water_mask = rasterize(
    shapes=shapes,
    out_shape=(image_height, image_width),
    transform=image_transform,
    fill=0,
    dtype="uint8"
)


# ------------------------------------------------------------
# SAVE MASK
# ------------------------------------------------------------

with rasterio.open(
    OUTPUT_PATH,
    "w",
    driver="GTiff",
    height=image_height,
    width=image_width,
    count=1,
    dtype="uint8",
    crs=image_crs,
    transform=image_transform,
    nodata=0
) as dst:

    dst.write(water_mask, 1)


# ------------------------------------------------------------
# STATISTICS
# ------------------------------------------------------------

water_pixels = int(
    (water_mask == 1).sum()
)

total_pixels = water_mask.size

coverage = (
    water_pixels / total_pixels * 100
    if total_pixels > 0
    else 0
)


# ------------------------------------------------------------
# RESULT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 115 COMPLETE")
print("=" * 70)

print(f"\nWaterbody      : Osman Sagar")
print(f"Image size     : {image_width} x {image_height}")
print(f"Water pixels   : {water_pixels:,}")
print(f"Water coverage : {coverage:.2f}%")
print(f"CRS            : {image_crs}")

print("\nWater mask saved to:")
print(os.path.abspath(OUTPUT_PATH))

print("=" * 70)