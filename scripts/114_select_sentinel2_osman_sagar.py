import os
import importlib.util
import geopandas as gpd
import ee


print("=" * 70)
print("STEP 114 — SELECT SENTINEL-2 IMAGE")
print("=" * 70)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

GEOMETRY_PATH = (
    "results/waterbody_discovery/second_waterbody/"
    "selected_waterbody_geometry.geojson"
)

OUTPUT_DIR = (
    "data/raw/sentinel2/second_waterbody"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "Osman_Sagar_2025_14Channel.tif"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# LOAD GEE UTILITIES
# ------------------------------------------------------------

print("\nLoading GEE utilities...")

spec = importlib.util.spec_from_file_location(
    "gee_utils",
    "utils/gee_utils.py"
)

gee_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gee_utils)


# ------------------------------------------------------------
# INITIALIZE GEE
# ------------------------------------------------------------

gee_utils.initialize_gee()


# ------------------------------------------------------------
# LOAD GEOMETRY
# ------------------------------------------------------------

print("\nLoading Osman Sagar geometry...")

gdf = gpd.read_file(GEOMETRY_PATH)

print(f"Geometry type : {gdf.geometry.iloc[0].geom_type}")
print(f"CRS           : {gdf.crs}")

# Convert MultiPolygon GeoJSON to an Earth Engine geometry
geojson = gdf.geometry.iloc[0].__geo_interface__

geometry = ee.Geometry(geojson)


# ------------------------------------------------------------
# GET SENTINEL-2 COLLECTION
# ------------------------------------------------------------

print("\nSearching Sentinel-2 observations...")

collection = gee_utils.get_masked_sentinel2_collection(
    geometry,
    "2025-01-01",
    "2025-12-31",
    cloud_threshold=40
)

count = collection.size().getInfo()

print(f"\nUsable Sentinel-2 images : {count}")


if count == 0:

    raise RuntimeError(
        "No usable Sentinel-2 images found for Osman Sagar."
    )


# ------------------------------------------------------------
# SELECT IMAGE WITH LOWEST CLOUD PROBABILITY
# ------------------------------------------------------------

# ------------------------------------------------------------
# SELECT FIRST USABLE LOW-CLOUD IMAGE
# ------------------------------------------------------------

# The GEE utility has already applied the cloud-probability
# filtering, so we select the first available image.
selected = ee.Image(
    collection.sort("system:time_start").first()
)


# ------------------------------------------------------------
# GET SELECTED IMAGE INFORMATION
# ------------------------------------------------------------

selected_date = ee.Date(
    selected.get("system:time_start")
).format("YYYY-MM-dd").getInfo()

system_index = selected.get(
    "system:index"
).getInfo()


print("\n" + "=" * 70)
print("SELECTED SENTINEL-2 IMAGE")
print("=" * 70)

print(f"\nDate         : {selected_date}")
print(f"System index : {system_index}")


# ------------------------------------------------------------
# GET SELECTED IMAGE INFORMATION
# ------------------------------------------------------------

selected_date = ee.Date(
    selected.get("system:time_start")
).format("YYYY-MM-dd").getInfo()

cloud_probability = selected.get(
    "mean_cloud_probability"
).getInfo()

system_index = selected.get(
    "system:index"
).getInfo()


print("\n" + "=" * 70)
print("SELECTED SENTINEL-2 IMAGE")
print("=" * 70)

print(f"\nDate             : {selected_date}")

print(f"System index     : {system_index}")


# ------------------------------------------------------------
# CREATE 14-CHANNEL IMAGE
# ------------------------------------------------------------

print("\nCreating 14-channel image...")

image_14 = gee_utils.create_14_channel_image(
    selected
)


# ------------------------------------------------------------
# DOWNLOAD
# ------------------------------------------------------------

print("Downloading 14-channel image...")
print("Please wait...")

gee_utils.download_14_channel_image(
    image_14,
    geometry,
    OUTPUT_PATH,
    scale=10
)


# ------------------------------------------------------------
# VERIFY OUTPUT
# ------------------------------------------------------------

import rasterio

with rasterio.open(OUTPUT_PATH) as src:

    width = src.width
    height = src.height
    bands = src.count
    crs = src.crs


print("\n" + "=" * 70)
print("STEP 114 COMPLETE")
print("=" * 70)

print(f"\nWaterbody        : Osman Sagar")
print(f"Selected date    : {selected_date}")

print(f"Image size       : {width} x {height}")
print(f"Channels         : {bands}")
print(f"CRS              : {crs}")

print("\nImage saved to:")
print(os.path.abspath(OUTPUT_PATH))

print("=" * 70)