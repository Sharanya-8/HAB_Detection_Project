from pathlib import Path

import ee


# ============================================================
# INITIALIZE GOOGLE EARTH ENGINE
# ============================================================

PROJECT_ID = "remote-sensing-project-507517"

ee.Initialize(
    project=PROJECT_ID
)


# ============================================================
# TEST WATERBODY
# Hussain Sagar coordinates
# ============================================================

latitude = 17.422977
longitude = 78.474552


# Create a small test geometry around the selected
# waterbody center. This is only for testing the
# dynamic Sentinel-2 connection.

aoi = ee.Geometry.Point(
    [
        longitude,
        latitude
    ]
).buffer(3000)


# ============================================================
# SENTINEL-2 COLLECTION
# ============================================================

sentinel2 = (
    ee.ImageCollection(
        "COPERNICUS/S2_SR_HARMONIZED"
    )
    .filterBounds(aoi)
    .filterDate(
        "2025-01-01",
        "2025-12-31"
    )
)


# ============================================================
# CLOUD PROBABILITY COLLECTION
# ============================================================

cloud_probability = (
    ee.ImageCollection(
        "COPERNICUS/S2_CLOUD_PROBABILITY"
    )
    .filterBounds(aoi)
    .filterDate(
        "2025-01-01",
        "2025-12-31"
    )
)


# ============================================================
# COUNTS
# ============================================================

sentinel_count = sentinel2.size().getInfo()

cloud_count = cloud_probability.size().getInfo()


print()
print("=" * 60)
print("DYNAMIC SENTINEL-2 ACQUISITION TEST")
print("=" * 60)

print(
    f"AOI center: "
    f"{latitude:.6f}, {longitude:.6f}"
)

print(
    f"Sentinel-2 images: "
    f"{sentinel_count}"
)

print(
    f"Cloud probability images: "
    f"{cloud_count}"
)


# ============================================================
# GET IMAGE DATES
# ============================================================

dates = (
    sentinel2
    .aggregate_array("system:time_start")
    .getInfo()
)


print()
print(
    f"Number of available observations: "
    f"{len(dates)}"
)


# ============================================================
# SHOW FIRST 10 DATES
# ============================================================

import datetime


print()
print("First 10 available Sentinel-2 observations:")
print("-" * 60)


for timestamp in dates[:10]:

    date = datetime.datetime.fromtimestamp(
        timestamp / 1000,
        tz=datetime.timezone.utc
    )

    print(
        date.strftime("%Y-%m-%d")
    )


print()
print("=" * 60)
print("TEST COMPLETED")
print("=" * 60)