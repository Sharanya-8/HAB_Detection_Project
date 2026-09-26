import os
import sys
import geopandas as gpd
import ee

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from utils.gee_utils import initialize_gee


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

BOUNDARY_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "labels",
    "hab_masks",
    "training_waterbodies",
    "Shamirpet_Lake_OSM_boundary.geojson"
)

START_DATE = "2016-01-01"
END_DATE = "2026-01-01"
CLOUD_THRESHOLD = 40


# ---------------------------------------------------------
# INITIALIZE GEE
# ---------------------------------------------------------

print("=" * 70)
print("STEP 83A - SHAMIRPET SENTINEL-2 DATE CHECK")
print("=" * 70)

initialize_gee()

# ---------------------------------------------------------
# LOAD BOUNDARY
# ---------------------------------------------------------

print("\nLoading Shamirpet boundary...")

gdf = gpd.read_file(BOUNDARY_FILE)

if gdf.empty:
    raise ValueError("Shamirpet boundary is empty.")

geometry = gdf.geometry.iloc[0]

if geometry is None or geometry.is_empty:
    raise ValueError("Shamirpet geometry is empty.")

# Convert GeoJSON geometry to Earth Engine geometry
aoi = ee.Geometry(geometry.__geo_interface__)

print("Boundary loaded successfully.")

# ---------------------------------------------------------
# SENTINEL-2
# ---------------------------------------------------------

print("\nChecking Sentinel-2 availability...")
print("Period:", START_DATE, "to", END_DATE)
print("Cloud probability threshold:", CLOUD_THRESHOLD, "%")

s2 = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(aoi)
    .filterDate(START_DATE, END_DATE)
)

cloud = (
    ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
    .filterBounds(aoi)
    .filterDate(START_DATE, END_DATE)
)

print("\nTotal Sentinel-2 images:", s2.size().getInfo())
print("Total cloud-probability images:", cloud.size().getInfo())

# ---------------------------------------------------------
# JOIN CLOUD PROBABILITY
# ---------------------------------------------------------

join = ee.Join.saveFirst("cloud_probability")

filter_join = ee.Filter.equals(
    leftField="system:index",
    rightField="system:index"
)

joined = join.apply(
    primary=s2,
    secondary=cloud,
    condition=filter_join
)


# ---------------------------------------------------------
# CLOUD MASK
# ---------------------------------------------------------

def add_cloud_mask(image):

    cloud_image = ee.Image(
        image.get("cloud_probability")
    )

    probability = cloud_image.select("probability")

    clear = probability.lt(CLOUD_THRESHOLD)

    # Percentage of clear pixels inside Shamirpet
    clear_percentage = (
        clear.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=20,
            maxPixels=1e8
        )
        .get("probability")
    )

    return image.set(
        "clear_percentage",
        clear_percentage
    )


checked = joined.map(add_cloud_mask)

# Keep images with at least 60% clear pixels
usable = checked.filter(
    ee.Filter.gte("clear_percentage", 0.60)
)

# ---------------------------------------------------------
# GET DATES
# ---------------------------------------------------------

dates = (
    usable
    .aggregate_array("system:time_start")
    .getInfo()
)

dates = sorted(
    set(
        ee.Date(timestamp).format("YYYY-MM-dd").getInfo()
        for timestamp in dates
    )
)

print("\n" + "=" * 70)
print("SHAMIRPET RESULTS")
print("=" * 70)

print("Usable unique dates:", len(dates))

if dates:
    print("First usable date:", dates[0])
    print("Last usable date :", dates[-1])

    print("\nFirst 20 usable dates:")
    for date in dates[:20]:
        print(date)

    print("\nLast 20 usable dates:")
    for date in dates[-20:]:
        print(date)
else:
    print("No usable dates found.")

# ---------------------------------------------------------
# SAVE DATES
# ---------------------------------------------------------

output_file = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "shamirpet_usable_dates.txt"
)

with open(output_file, "w") as f:
    for date in dates:
        f.write(date + "\n")

print("\nSaved dates to:")
print(output_file)

print("\n" + "=" * 70)
print("STEP 83A COMPLETE")
print("=" * 70)