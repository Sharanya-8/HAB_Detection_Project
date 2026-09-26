from pathlib import Path
import sys

import requests


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ============================================================
# IMPORT UTILITY
# ============================================================

from utils.gee_utils import (
    initialize_gee,
    get_masked_sentinel2_collection,
    get_available_dates,
    get_image_for_date,
    create_14_channel_image,
)


# ============================================================
# EARTH ENGINE
# ============================================================

import ee


print("=" * 70)
print("GEE UTILITY TEST")
print("=" * 70)


# ============================================================
# INITIALIZE
# ============================================================

print("\nInitializing GEE...")

initialize_gee()

print("GEE initialized.")


# ============================================================
# TEST AOI
# ============================================================

latitude = 17.422977
longitude = 78.474552

aoi = (
    ee.Geometry
    .Point(
        [
            longitude,
            latitude
        ]
    )
    .buffer(3000)
)


# ============================================================
# GET COLLECTION
# ============================================================

print("\nGetting Sentinel-2 collection...")

collection = (
    get_masked_sentinel2_collection(
        aoi,
        "2025-01-01",
        "2025-12-31",
        cloud_threshold=40
    )
)


count = (
    collection
    .size()
    .getInfo()
)


print(
    f"Masked Sentinel-2 images: "
    f"{count}"
)


# ============================================================
# DATES
# ============================================================

dates = get_available_dates(
    collection
)


print(
    f"Unique dates: "
    f"{len(dates)}"
)


print("\nFirst 10 dates:")

for date in dates[:10]:

    print(
        f"  {date}"
    )


# ============================================================
# SELECT IMAGE
# ============================================================

test_date = dates[0]

print(
    f"\nTesting date: "
    f"{test_date}"
)


image = get_image_for_date(
    collection,
    test_date
)


if image is None:

    raise RuntimeError(
        "No image found for test date."
    )


print(
    "Sentinel-2 image selected."
)


# ============================================================
# CREATE 14 CHANNELS
# ============================================================

print(
    "\nCreating 14-channel image..."
)


final_image = create_14_channel_image(
    image
)


band_names = (
    final_image
    .bandNames()
    .getInfo()
)


print("\nChannels:")

for index, name in enumerate(
    band_names,
    start=1
):

    print(
        f"  {index}. {name}"
    )


# ============================================================
# VERIFY
# ============================================================

if len(band_names) != 14:

    raise RuntimeError(
        f"Expected 14 channels, "
        f"found {len(band_names)}"
    )


expected = [
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


if band_names != expected:

    raise RuntimeError(
        "Channel order does not match "
        "the expected Swin input."
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print(
    "The reusable GEE utility successfully:"
)

print(
    "1. Retrieved Sentinel-2 imagery"
)

print(
    "2. Applied cloud masking"
)

print(
    "3. Selected an observation date"
)

print(
    "4. Created the required 14 channels"
)

print(
    "The utility is ready to be connected "
    "to the dashboard."
)

print("=" * 70)