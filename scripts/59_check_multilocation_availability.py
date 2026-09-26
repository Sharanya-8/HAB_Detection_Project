import ee
import geopandas as gpd
from pathlib import Path
from datetime import datetime


# ============================================================
# STEP 59 - MULTI-WATERBODY SENTINEL-2 AVAILABILITY
# ============================================================

print("=" * 70)
print("STEP 59 - MULTI-WATERBODY SENTINEL-2 AVAILABILITY")
print("=" * 70)


# ------------------------------------------------------------
# Initialize Google Earth Engine
# ------------------------------------------------------------

PROJECT_ID = "remote-sensing-project-507517"

try:
    ee.Initialize(project=PROJECT_ID)
    print("Google Earth Engine initialized successfully.")
except Exception as e:
    print("ERROR initializing Google Earth Engine:")
    print(e)
    raise


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BOUNDARY_DIR = Path(
    "data/labels/hab_masks/training_waterbodies"
)


# ------------------------------------------------------------
# Training waterbodies
# ------------------------------------------------------------

waterbodies = {
    "Hussain Sagar": "Hussain_Sagar_boundary.geojson",
    "Saroor Nagar": "Saroor_Nagar_OSM_boundary.geojson",
    "Osman Sagar": "Osman_Sagar_boundary.geojson",
    "Himayat Sagar": "Himayat_Sagar_relation_boundary.geojson",
}


# ------------------------------------------------------------
# Sentinel-2 collections
# ------------------------------------------------------------

S2 = ee.ImageCollection(
    "COPERNICUS/S2_SR_HARMONIZED"
)

CLOUD_PROBABILITY = ee.ImageCollection(
    "COPERNICUS/S2_CLOUD_PROBABILITY"
)


# ------------------------------------------------------------
# Date range
# ------------------------------------------------------------

START_DATE = "2016-01-01"
END_DATE = "2026-12-31"


# ------------------------------------------------------------
# Cloud threshold
# Same threshold used in the existing project
# ------------------------------------------------------------

CLOUD_THRESHOLD = 40


# ------------------------------------------------------------
# Helper: convert GeoJSON to EE geometry
# ------------------------------------------------------------

def geojson_to_ee_geometry(filepath):

    gdf = gpd.read_file(filepath)

    if gdf.empty:
        raise ValueError(
            f"Empty boundary file: {filepath}"
        )

    geometry = gdf.geometry.iloc[0]

    if geometry is None:
        raise ValueError(
            f"No geometry found: {filepath}"
        )

    geojson = geometry.__geo_interface__

    return ee.Geometry(geojson)


# ------------------------------------------------------------
# Check each waterbody
# ------------------------------------------------------------

results = []


for name, filename in waterbodies.items():

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    filepath = BOUNDARY_DIR / filename

    print(
        "Boundary:",
        filepath
    )

    geometry = geojson_to_ee_geometry(
        filepath
    )

    # --------------------------------------------------------
    # Sentinel-2 SR images over waterbody
    # --------------------------------------------------------

    s2_collection = (
        S2
        .filterBounds(geometry)
        .filterDate(
            START_DATE,
            END_DATE
        )
    )

    total_count = (
        s2_collection.size()
        .getInfo()
    )

    print(
        "Total Sentinel-2 SR images:",
        total_count
    )

    # --------------------------------------------------------
    # Cloud probability collection
    # --------------------------------------------------------

    cloud_collection = (
        CLOUD_PROBABILITY
        .filterBounds(geometry)
        .filterDate(
            START_DATE,
            END_DATE
        )
    )

    cloud_count = (
        cloud_collection.size()
        .getInfo()
    )

    print(
        "Cloud probability images:",
        cloud_count
    )

    # --------------------------------------------------------
    # Join Sentinel-2 with cloud probability
    # --------------------------------------------------------

    joined = ee.Join.saveFirst(
        "cloud_probability"
    ).apply(
        primary=s2_collection,
        secondary=cloud_collection,
        condition=ee.Filter.equals(
            leftField="system:index",
            rightField="system:index"
        )
    )

    joined = ee.ImageCollection(joined)

    # --------------------------------------------------------
    # Filter images based on cloud probability
    # --------------------------------------------------------

    def add_cloud_percentage(image):

        cloud_image = ee.Image(
            image.get("cloud_probability")
        )

        cloud_probability = (
            cloud_image
            .select("probability")
        )

        cloudy = (
            cloud_probability
            .gte(CLOUD_THRESHOLD)
        )

        cloud_percentage = (
            cloudy
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=20,
                maxPixels=1e8
            )
            .get("probability")
        )

        return image.set(
            "cloud_percentage",
            ee.Number(cloud_percentage)
        )


    joined_with_cloud = (
        joined.map(add_cloud_percentage)
    )

    # --------------------------------------------------------
    # Keep images where less than 40% of the
    # waterbody is cloudy
    # --------------------------------------------------------

    usable = (
        joined_with_cloud
        .filter(
            ee.Filter.lt(
                "cloud_percentage",
                0.40
            )
        )
    )

    usable_count = (
        usable.size()
        .getInfo()
    )

    print(
        "Usable images (<40% cloudy):",
        usable_count
    )

    # --------------------------------------------------------
    # Get available dates
    # --------------------------------------------------------

    date_list = (
        usable
        .aggregate_array("system:time_start")
        .getInfo()
    )

    dates = []

    for timestamp in date_list:

        date = datetime.fromtimestamp(
            timestamp / 1000
        ).strftime("%Y-%m-%d")

        dates.append(date)

    dates = sorted(set(dates))

    print(
        "Unique usable dates:",
        len(dates)
    )

    if dates:

        print(
            "First usable date:",
            dates[0]
        )

        print(
            "Last usable date:",
            dates[-1]
        )

    # --------------------------------------------------------
    # Year-wise counts
    # --------------------------------------------------------

    yearly_counts = {}

    for date in dates:

        year = date[:4]

        yearly_counts[year] = (
            yearly_counts.get(year, 0) + 1
        )

    print()
    print("Year-wise usable image counts:")

    for year in sorted(yearly_counts):

        print(
            f"  {year}: {yearly_counts[year]}"
        )

    results.append({
        "waterbody": name,
        "total_images": total_count,
        "usable_images": usable_count,
        "unique_dates": len(dates),
        "first_date": dates[0] if dates else None,
        "last_date": dates[-1] if dates else None,
    })


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL AVAILABILITY SUMMARY")
print("=" * 70)

print()

for result in results:

    print(
        f"{result['waterbody']}: "
        f"{result['usable_images']} usable images, "
        f"{result['first_date']} → "
        f"{result['last_date']}"
    )

print()
print("=" * 70)
print("STEP 59 COMPLETE")
print("=" * 70)