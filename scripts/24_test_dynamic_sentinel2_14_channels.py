from pathlib import Path
import datetime
import requests
import ee
import rasterio


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = "remote-sensing-project-507517"

LATITUDE = 17.422977
LONGITUDE = 78.474552

START_DATE = "2025-01-01"
END_DATE = "2025-12-31"

CLOUD_THRESHOLD = 40

OUTPUT_DIR = Path("data/raw/sentinel2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "Dynamic_Hussain_Sagar_14Channel_Test.tif"


# ============================================================
# INITIALIZE EARTH ENGINE
# ============================================================

print("=" * 70)
print("DYNAMIC SENTINEL-2 14-CHANNEL TEST")
print("=" * 70)

print("\nInitializing Google Earth Engine...")

ee.Initialize(project=PROJECT_ID)

print("Google Earth Engine initialized successfully.")


# ============================================================
# STEP 1: FIND HUSSAIN SAGAR THROUGH OSM
# ============================================================

print("\nSearching for Hussain Sagar...")


overpass_url = "https://overpass-api.de/api/interpreter"

query = f"""
[out:json][timeout:60];
(
  way["natural"="water"](around:5000,{LATITUDE},{LONGITUDE});
  relation["natural"="water"](around:5000,{LATITUDE},{LONGITUDE});
  way["water"](around:5000,{LATITUDE},{LONGITUDE});
  relation["water"](around:5000,{LATITUDE},{LONGITUDE});
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


# Find Hussain Sagar specifically

selected = None

for element in waterbodies:

    name = element.get("tags", {}).get("name", "")

    if name.lower() == "hussain sagar":
        selected = element
        break


if selected is None:
    raise RuntimeError("Hussain Sagar was not found.")


osm_type = selected["type"]
osm_id = selected["id"]

print(f"Waterbody found: Hussain Sagar")
print(f"OSM type: {osm_type}")
print(f"OSM ID: {osm_id}")


# ============================================================
# STEP 2: RETRIEVE ACTUAL POLYGON
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

print(f"Geometry type: {geometry['type']}")


# ============================================================
# STEP 3: CREATE EARTH ENGINE AOI
# ============================================================

coordinates = geometry["coordinates"]


if geometry["type"] == "Polygon":

    aoi = ee.Geometry.Polygon(coordinates)


elif geometry["type"] == "MultiPolygon":

    aoi = ee.Geometry.MultiPolygon(coordinates)


else:

    raise RuntimeError(
        f"Unsupported geometry type: {geometry['type']}"
    )


print("Earth Engine AOI created successfully.")


# ============================================================
# STEP 4: SENTINEL-2 COLLECTION
# ============================================================

print("\nSearching Sentinel-2 images...")


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


print(
    f"Sentinel-2 images: {sentinel2.size().getInfo()}"
)

print(
    f"Cloud probability images: {cloud_probability.size().getInfo()}"
)


# ============================================================
# STEP 5: JOIN CLOUD PROBABILITY
# ============================================================

print("\nMatching cloud probability data...")


join_filter = ee.Filter.equals(
    leftField="system:index",
    rightField="system:index"
)


joined = ee.Join.saveFirst(
    "cloud_probability"
).apply(
    primary=sentinel2,
    secondary=cloud_probability,
    condition=join_filter
)


joined_collection = ee.ImageCollection(joined)


print(
    f"Matched images: {joined_collection.size().getInfo()}"
)


# ============================================================
# STEP 6: CLOUD MASK
# ============================================================

def mask_clouds(image):

    cloud_probability_image = ee.Image(
        image.get("cloud_probability")
    )

    cloud_mask = cloud_probability_image.lt(
        CLOUD_THRESHOLD
    )

    return (
        image
        .updateMask(cloud_mask)
        .copyProperties(
            image,
            image.propertyNames()
        )
    )


masked_collection = joined_collection.map(mask_clouds)


# ============================================================
# STEP 7: SELECT A TEST IMAGE
# ============================================================

# Sort by date and select the first available observation.

image = ee.Image(
    masked_collection
    .sort("system:time_start")
    .first()
)


if image is None:
    raise RuntimeError(
        "No Sentinel-2 image is available."
    )


timestamp = image.get(
    "system:time_start"
).getInfo()


observation_date = datetime.datetime.fromtimestamp(
    timestamp / 1000,
    tz=datetime.timezone.utc
).strftime("%Y-%m-%d")


print(f"\nSelected observation date: {observation_date}")


# ============================================================
# STEP 8: SELECT 10 SENTINEL-2 BANDS
# ============================================================

print("\nPreparing Sentinel-2 bands...")


bands = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12"
]


spectral = image.select(bands)


# ============================================================
# STEP 9: CALCULATE SPECTRAL INDICES
# ============================================================

print("Calculating spectral indices...")


# NDWI
ndwi = image.normalizedDifference(
    ["B3", "B8"]
).rename("NDWI")


# MNDWI
mndwi = image.normalizedDifference(
    ["B3", "B11"]
).rename("MNDWI")


# NDCI
ndci = image.normalizedDifference(
    ["B5", "B4"]
).rename("NDCI")


# ============================================================
# FAI
# ============================================================

# Floating Algae Index
#
# Approximate wavelengths:
# B4  = 665 nm
# B8  = 842 nm
# B11 = 1610 nm

baseline = image.select("B4").add(
    image.select("B11")
    .subtract(image.select("B4"))
    .multiply(
        (842 - 665) / (1610 - 665)
    )
)


fai = image.select("B8").subtract(
    baseline
).rename("FAI")


# ============================================================
# STEP 10: CREATE 14-CHANNEL IMAGE
# ============================================================

print("\nCreating final 14-channel image...")


final_image = spectral.addBands(
    ndwi
).addBands(
    mndwi
).addBands(
    ndci
).addBands(
    fai
).toFloat()


channel_names = [
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


print("\nChannels:")
for i, channel in enumerate(channel_names, start=1):
    print(f"  {i}. {channel}")


# ============================================================
# STEP 11: DOWNLOAD IMAGE
# ============================================================

print("\nPreparing GeoTIFF download...")

download_url = final_image.getDownloadURL({
    "region": aoi,
    "scale": 10,
    "crs": "EPSG:32644",
    "format": "GEO_TIFF",
    "filePerBand": False
})


print("Download URL created successfully.")

print("\nDownloading image...")


download_response = requests.get(
    download_url,
    timeout=180
)

download_response.raise_for_status()


with open(OUTPUT_FILE, "wb") as f:
    f.write(download_response.content)


print(
    f"Downloaded to: {OUTPUT_FILE}"
)


# ============================================================
# STEP 12: VERIFY DOWNLOADED IMAGE
# ============================================================

print("\nVerifying GeoTIFF...")


with rasterio.open(OUTPUT_FILE) as src:

    print("\n" + "=" * 70)
    print("DOWNLOADED IMAGE INFORMATION")
    print("=" * 70)

    print(f"Width: {src.width}")
    print(f"Height: {src.height}")
    print(f"Bands: {src.count}")
    print(f"CRS: {src.crs}")
    print(f"Resolution: {src.res}")
    print(f"Data type: {src.dtypes[0]}")

    print("\nBand descriptions:")

    for i in range(1, src.count + 1):

        description = src.descriptions[i - 1]

        print(
            f"  Band {i}: {description}"
        )


# ============================================================
# FINAL CHECK
# ============================================================

with rasterio.open(OUTPUT_FILE) as src:

    if src.count != 14:

        raise RuntimeError(
            f"Expected 14 bands, but got {src.count}"
        )


print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print(
    "Dynamic waterbody → Sentinel-2 → "
    "cloud masking → spectral indices → "
    "14-channel GeoTIFF"
)

print("\nThe 14-channel Sentinel-2 pipeline works.")
print("=" * 70)