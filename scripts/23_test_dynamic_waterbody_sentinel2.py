from pathlib import Path
import json
import requests
import ee


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = "remote-sensing-project-507517"

LATITUDE = 17.422977
LONGITUDE = 78.474552

# Hussain Sagar OSM relation/way will be retrieved dynamically
# using the coordinates through Nominatim.
SEARCH_RADIUS_METERS = 5000

START_DATE = "2025-01-01"
END_DATE = "2025-12-31"

CLOUD_THRESHOLD = 40


# ============================================================
# INITIALIZE GOOGLE EARTH ENGINE
# ============================================================

print("=" * 70)
print("DYNAMIC WATERBODY SENTINEL-2 TEST")
print("=" * 70)

print("\nInitializing Google Earth Engine...")

ee.Initialize(project=PROJECT_ID)

print("Google Earth Engine initialized successfully.")


# ============================================================
# STEP 1: FIND NEARBY WATERBODY
# ============================================================

print("\nSearching for nearby waterbodies...")

overpass_url = "https://overpass-api.de/api/interpreter"

query = f"""
[out:json][timeout:60];
(
  way["natural"="water"](around:{SEARCH_RADIUS_METERS},{LATITUDE},{LONGITUDE});
  relation["natural"="water"](around:{SEARCH_RADIUS_METERS},{LATITUDE},{LONGITUDE});
  way["water"](around:{SEARCH_RADIUS_METERS},{LATITUDE},{LONGITUDE});
  relation["water"](around:{SEARCH_RADIUS_METERS},{LATITUDE},{LONGITUDE});
);
out center tags;
"""

response = requests.get(
    overpass_url,
    params={"data": query},
    headers={"User-Agent": "HAB-Detection-Project/1.0"},
    timeout=90
)

response.raise_for_status()

waterbodies = response.json()["elements"]

print(f"Waterbodies found: {len(waterbodies)}")


# ============================================================
# FIND A NAMED WATERBODY
# ============================================================

named_waterbodies = []

for element in waterbodies:

    tags = element.get("tags", {})

    name = tags.get("name")

    if name:
        named_waterbodies.append(element)


if not named_waterbodies:
    raise RuntimeError("No named waterbody found.")


# Select the nearest named waterbody
selected = min(
    named_waterbodies,
    key=lambda element: (
        (element.get("center", {}).get("lat", LATITUDE) - LATITUDE) ** 2
        +
        (element.get("center", {}).get("lon", LONGITUDE) - LONGITUDE) ** 2
    )
)

osm_type = selected["type"]
osm_id = selected["id"]
waterbody_name = selected["tags"].get("name", "Unknown")

print("\nSelected waterbody:")
print(f"Name: {waterbody_name}")
print(f"OSM type: {osm_type}")
print(f"OSM ID: {osm_id}")


# ============================================================
# STEP 2: RETRIEVE ACTUAL WATERBODY GEOMETRY
# ============================================================

print("\nRetrieving actual waterbody polygon...")

if osm_type == "way":
    osm_id_string = f"W{osm_id}"
elif osm_type == "relation":
    osm_id_string = f"R{osm_id}"
else:
    raise RuntimeError(f"Unsupported OSM type: {osm_type}")


nominatim_url = "https://nominatim.openstreetmap.org/lookup"

params = {
    "osm_ids": osm_id_string,
    "format": "geojson",
    "polygon_geojson": 1
}

response = requests.get(
    nominatim_url,
    params=params,
    headers={"User-Agent": "HAB-Detection-Project/1.0"},
    timeout=60
)

response.raise_for_status()

geojson = response.json()

features = geojson.get("features", [])

if not features:
    raise RuntimeError("Waterbody polygon could not be retrieved.")


geometry = features[0]["geometry"]

print("Waterbody polygon retrieved successfully.")
print(f"Geometry type: {geometry['type']}")


# ============================================================
# STEP 3: CONVERT GEOJSON TO EARTH ENGINE GEOMETRY
# ============================================================

print("\nConverting polygon to Earth Engine geometry...")

geometry_type = geometry["type"]
coordinates = geometry["coordinates"]


if geometry_type == "Polygon":

    aoi = ee.Geometry.Polygon(coordinates)


elif geometry_type == "MultiPolygon":

    aoi = ee.Geometry.MultiPolygon(coordinates)


else:

    raise RuntimeError(
        f"Unsupported geometry type: {geometry_type}"
    )


print("Earth Engine AOI created successfully.")


# ============================================================
# STEP 4: SENTINEL-2 COLLECTION
# ============================================================

print("\nSearching Sentinel-2 imagery...")

sentinel2 = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(aoi)
    .filterDate(START_DATE, END_DATE)
)


cloud_probability = (
    ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
    .filterBounds(aoi)
    .filterDate(START_DATE, END_DATE)
)


sentinel_count = sentinel2.size().getInfo()
cloud_count = cloud_probability.size().getInfo()


print(f"Sentinel-2 images: {sentinel_count}")
print(f"Cloud probability images: {cloud_count}")


# ============================================================
# STEP 5: MATCH SENTINEL-2 WITH CLOUD PROBABILITY
# ============================================================

print("\nMatching Sentinel-2 images with cloud probability...")

join_filter = ee.Filter.equals(
    leftField="system:index",
    rightField="system:index"
)

joined = ee.Join.saveFirst("cloud_probability").apply(
    primary=sentinel2,
    secondary=cloud_probability,
    condition=join_filter
)


joined_collection = ee.ImageCollection(joined)

joined_count = joined_collection.size().getInfo()

print(f"Successfully matched images: {joined_count}")


# ============================================================
# STEP 6: APPLY PIXEL-LEVEL CLOUD MASK
# ============================================================

def mask_clouds(image):

    cloud_probability_image = ee.Image(
        image.get("cloud_probability")
    )

    cloud_mask = cloud_probability_image.lt(CLOUD_THRESHOLD)

    return (
        image
        .updateMask(cloud_mask)
        .copyProperties(image, image.propertyNames())
    )


masked_collection = joined_collection.map(mask_clouds)

masked_count = masked_collection.size().getInfo()

print(f"Images after cloud-mask processing: {masked_count}")


# ============================================================
# STEP 7: GET AVAILABLE OBSERVATION DATES
# ============================================================

print("\nAvailable observations:")

timestamps = (
    masked_collection
    .aggregate_array("system:time_start")
    .getInfo()
)

import datetime

dates = []

for timestamp in timestamps:

    date = datetime.datetime.fromtimestamp(
        timestamp / 1000,
        tz=datetime.timezone.utc
    ).strftime("%Y-%m-%d")

    dates.append(date)


unique_dates = sorted(set(dates))


print(f"Unique observation dates: {len(unique_dates)}")

for date in unique_dates[:20]:
    print(f"  {date}")


if len(unique_dates) > 20:
    print(f"  ... and {len(unique_dates) - 20} more")


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("DYNAMIC WATERBODY SENTINEL-2 TEST COMPLETED")
print("=" * 70)

print(f"Waterbody: {waterbody_name}")
print(f"Geometry: {geometry_type}")
print(f"Sentinel-2 images: {sentinel_count}")
print(f"Cloud probability images: {cloud_count}")
print(f"Matched images: {joined_count}")
print(f"Unique usable observation dates: {len(unique_dates)}")

print("\nThe dynamic waterbody → Sentinel-2 pipeline works.")
print("=" * 70)