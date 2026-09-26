"""
STEP 108 — SELECT SENTINEL-2 IMAGE

Finds a suitable Sentinel-2 observation for the selected
waterbody analysis geometry and downloads the 14-channel image.

Input:
    results/waterbody_discovery/selected_waterbody/
    analysis_geometry.geojson

Output:
    data/raw/sentinel2/generic_selected/
        <date>_14Channel.tif

Selection:
    - Sentinel-2 SR Harmonized
    - Cloud probability < 40%
    - 2025-01-01 to 2025-12-31
    - Selects the observation with the lowest cloud probability
"""

import os
import sys
import importlib.util

import ee
import geopandas as gpd


# ============================================================
# PATHS
# ============================================================

GEOMETRY_PATH = os.path.join(
    "results",
    "waterbody_discovery",
    "selected_waterbody",
    "analysis_geometry.geojson"
)

OUTPUT_DIRECTORY = os.path.join(
    "data",
    "raw",
    "sentinel2",
    "generic_selected"
)

os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

START_DATE = "2025-01-01"
END_DATE = "2025-12-31"

CLOUD_THRESHOLD = 40

# Use a 10 m Sentinel-2 resolution.
SCALE = 10


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 108 — SELECT SENTINEL-2 IMAGE")
print("=" * 70)

print()


# ============================================================
# CHECK GEOMETRY
# ============================================================

if not os.path.exists(GEOMETRY_PATH):

    print(
        "ERROR: Analysis geometry not found:"
    )

    print(
        os.path.abspath(
            GEOMETRY_PATH
        )
    )

    raise SystemExit


# ============================================================
# LOAD GEOMETRY
# ============================================================

print(
    "Loading selected waterbody analysis geometry..."
)

gdf = gpd.read_file(
    GEOMETRY_PATH
)

if gdf.empty:

    print(
        "ERROR: Analysis geometry is empty."
    )

    raise SystemExit


if gdf.crs is None:

    print(
        "ERROR: Analysis geometry has no CRS."
    )

    raise SystemExit


gdf = gdf.to_crs(
    "EPSG:4326"
)

geometry = gdf.geometry.iloc[0]


if geometry.is_empty:

    print(
        "ERROR: Geometry is empty."
    )

    raise SystemExit


print(
    f"Geometry type : {geometry.geom_type}"
)

print(
    f"CRS           : {gdf.crs}"
)

print()


# ============================================================
# WATERBODY NAME
# ============================================================

waterbody_name = "Selected_Waterbody"

if "name" in gdf.columns:

    value = gdf.iloc[0]["name"]

    if value:

        waterbody_name = str(
            value
        )


print(
    f"Waterbody : {waterbody_name}"
)

print()


# ============================================================
# LOAD EXISTING GEE UTILITIES
# ============================================================

print(
    "Loading GEE utilities..."
)

spec = importlib.util.spec_from_file_location(
    "gee_utils",
    os.path.join(
        "utils",
        "gee_utils.py"
    )
)

gee_utils = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    gee_utils
)


# ============================================================
# INITIALIZE GEE
# ============================================================

print(
    "Initializing Google Earth Engine..."
)

gee_utils.initialize_gee()

print()


# ============================================================
# CONVERT GEOMETRY TO EARTH ENGINE
# ============================================================

print(
    "Converting analysis geometry to Earth Engine..."
)

geometry_json = geometry.__geo_interface__

aoi = ee.Geometry(
    geometry_json
)


# ============================================================
# GET SENTINEL-2 COLLECTION
# ============================================================

print(
    "Searching Sentinel-2 observations..."
)

print()

collection = (
    ee.ImageCollection(
        "COPERNICUS/S2_SR_HARMONIZED"
    )
    .filterBounds(aoi)
    .filterDate(
        START_DATE,
        END_DATE
    )
)


cloud_collection = (
    ee.ImageCollection(
        "COPERNICUS/S2_CLOUD_PROBABILITY"
    )
    .filterBounds(aoi)
    .filterDate(
        START_DATE,
        END_DATE
    )
)


# ============================================================
# JOIN CLOUD PROBABILITY
# ============================================================

joined = ee.Join.saveFirst(
    "cloud_probability"
).apply(
    primary=collection,
    secondary=cloud_collection,
    condition=ee.Filter.equals(
        leftField="system:index",
        rightField="system:index"
    )
)


# ============================================================
# CLOUD FILTER
# ============================================================

def add_cloud_property(image):

    cloud_image = ee.Image(
        image.get(
            "cloud_probability"
        )
    )

    cloud_probability = (
        cloud_image
        .select("probability")
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=20,
            maxPixels=1e8
        )
        .get("probability")
    )

    return image.set(
        "mean_cloud_probability",
        cloud_probability
    )


collection_with_cloud = (
    ee.ImageCollection(
        joined
    )
    .map(
        add_cloud_property
    )
)


usable = (
    collection_with_cloud
    .filter(
        ee.Filter.notNull(
            ["mean_cloud_probability"]
        )
    )
    .filter(
        ee.Filter.lte(
            "mean_cloud_probability",
            CLOUD_THRESHOLD
        )
    )
)


# ============================================================
# COUNT
# ============================================================

total_count = collection.size().getInfo()

usable_count = usable.size().getInfo()

print(
    f"Total Sentinel-2 images : {total_count}"
)

print(
    f"Usable images           : {usable_count}"
)

print()


if usable_count == 0:

    print("=" * 70)
    print("NO SUITABLE SENTINEL-2 IMAGE FOUND")
    print("=" * 70)

    print()

    print(
        f"No image with mean cloud probability "
        f"<= {CLOUD_THRESHOLD}% was found."
    )

    raise SystemExit


# ============================================================
# SELECT LOWEST-CLOUD IMAGE
# ============================================================

selected_image = (
    usable
    .sort(
        "mean_cloud_probability"
    )
    .first()
)


# ============================================================
# GET METADATA
# ============================================================

selected_date = ee.Date(
    selected_image.get(
        "system:time_start"
    )
).format(
    "YYYY-MM-dd"
).getInfo()


selected_cloud = selected_image.get(
    "mean_cloud_probability"
).getInfo()


selected_index = selected_image.get(
    "system:index"
).getInfo()


print("=" * 70)
print("SELECTED SENTINEL-2 IMAGE")
print("=" * 70)

print()

print(
    f"Date            : {selected_date}"
)

print(
    f"Mean cloud prob : {selected_cloud:.2f}%"
)

print(
    f"System index    : {selected_index}"
)

print()


# ============================================================
# GET IMAGE USING EXISTING UTILITY
# ============================================================

print(
    "Creating 14-channel image..."
)

# The selected image is already an ee.Image.
# We use the existing utility functions from gee_utils.py.

image_14 = (
    gee_utils.create_14_channel_image(
        selected_image
    )
)


# ============================================================
# OUTPUT PATH
# ============================================================

output_filename = (
    f"{selected_date}_14Channel.tif"
)

output_path = os.path.join(
    OUTPUT_DIRECTORY,
    output_filename
)


# ============================================================
# DOWNLOAD
# ============================================================

print(
    "Downloading 14-channel image..."
)

print()

gee_utils.download_14_channel_image(
    image_14,
    aoi,
    output_path,
    scale=SCALE
)


# ============================================================
# VERIFY OUTPUT
# ============================================================

if not os.path.exists(
    output_path
):

    print(
        "ERROR: Download completed but "
        "output file was not found."
    )

    raise SystemExit


# ============================================================
# READ OUTPUT METADATA
# ============================================================

import rasterio

with rasterio.open(
    output_path
) as src:

    width = src.width
    height = src.height
    bands = src.count
    crs = src.crs


# ============================================================
# FINAL RESULT
# ============================================================

print()

print("=" * 70)
print("STEP 108 COMPLETE")
print("=" * 70)

print()

print(
    f"Waterbody       : {waterbody_name}"
)

print(
    f"Selected date   : {selected_date}"
)

print(
    f"Cloud probability: {selected_cloud:.2f}%"
)

print(
    f"Image size      : {width} x {height}"
)

print(
    f"Channels        : {bands}"
)

print(
    f"CRS             : {crs}"
)

print()

print(
    "Image saved to:"
)

print(
    os.path.abspath(
        output_path
    )
)

print()

print("=" * 70)