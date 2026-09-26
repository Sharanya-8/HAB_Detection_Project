import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BOUNDARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "hab_masks"
    / "training_waterbodies"
    / "Shamirpet_Lake_OSM_boundary.geojson"
)

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sentinel2"
    / "shamirpet_test"
    / "2016-01-30.tif"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "hab_masks"
    / "shamirpet_test"
    / "Shamirpet_water_mask_2016-01-30.tif"
)


print("=" * 70)
print("STEP 90 — CREATE SHAMIRPET WATER MASK")
print("=" * 70)


# ------------------------------------------------------------
# CHECK FILES
# ------------------------------------------------------------

if not BOUNDARY_PATH.exists():
    raise FileNotFoundError(
        f"Boundary not found:\n{BOUNDARY_PATH}"
    )

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Sentinel-2 image not found:\n{IMAGE_PATH}"
    )


# ------------------------------------------------------------
# READ BOUNDARY
# ------------------------------------------------------------

print("\nReading Shamirpet boundary...")

gdf = gpd.read_file(BOUNDARY_PATH)

print("Geometry:", gdf.geometry.iloc[0].geom_type)
print("CRS:", gdf.crs)
print("Valid:", gdf.geometry.iloc[0].is_valid)


# ------------------------------------------------------------
# READ SENTINEL-2 IMAGE
# ------------------------------------------------------------

print("\nReading Sentinel-2 reference image...")

with rasterio.open(IMAGE_PATH) as src:

    height = src.height
    width = src.width
    transform = src.transform
    image_crs = src.crs

print("Image size:", width, "x", height)
print("Image CRS:", image_crs)


# ------------------------------------------------------------
# REPROJECT BOUNDARY TO IMAGE CRS
# ------------------------------------------------------------

print("\nAligning boundary to image CRS...")

gdf = gdf.to_crs(image_crs)

geometries = list(gdf.geometry)


# ------------------------------------------------------------
# CREATE MASK
# ------------------------------------------------------------

print("\nCreating raster water mask...")

mask = geometry_mask(
    geometries,
    out_shape=(height, width),
    transform=transform,
    invert=True
)


mask = mask.astype("uint8")


print("Water pixels:", int(mask.sum()))
print(
    "Water percentage:",
    f"{(mask.sum() / mask.size) * 100:.2f}%"
)


# ------------------------------------------------------------
# SAVE MASK
# ------------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


with rasterio.open(
    OUTPUT_PATH,
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=1,
    dtype="uint8",
    crs=image_crs,
    transform=transform,
    nodata=0
) as dst:

    dst.write(mask, 1)


print("\nMask saved successfully:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("STEP 90 COMPLETE")
print("=" * 70)