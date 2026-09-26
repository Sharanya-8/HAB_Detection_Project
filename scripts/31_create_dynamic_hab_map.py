from pathlib import Path
import sys
import requests

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_FILE = Path(
    "data/processed/sentinel2/"
    "Dynamic_Hussain_Sagar_14Channel_Corrected.tif"
)

PREDICTION_DIR = Path(
    "results/maps/dynamic_corrected"
)

OUTPUT_DIR = Path(
    "results/maps"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = OUTPUT_DIR / (
    "Dynamic_Hussain_Sagar_HAB_Swin_2025_01_02.tif"
)

OSM_ID = 2833155


# ============================================================
# START
# ============================================================

print("=" * 70)
print("DYNAMIC GEOGRAPHIC HAB MAP")
print("=" * 70)


# ============================================================
# STEP 1: READ SENTINEL-2 IMAGE
# ============================================================

print("\nReading corrected Sentinel-2 image...")

with rasterio.open(IMAGE_FILE) as src:

    image = src.read()

    profile = src.profile.copy()

    transform = src.transform

    crs = src.crs

    height = src.height
    width = src.width


print(f"Image size: {width} × {height}")
print(f"Channels: {image.shape[0]}")
print(f"CRS: {crs}")
print(f"Resolution: {transform.a} × {abs(transform.e)} meters")


# ============================================================
# STEP 2: RETRIEVE OSM POLYGON
# ============================================================

print("\nRetrieving Hussain Sagar polygon...")

nominatim_url = (
    "https://nominatim.openstreetmap.org/lookup"
)

params = {
    "osm_ids": f"R{OSM_ID}",
    "format": "geojson",
    "polygon_geojson": 1
}

response = requests.get(
    nominatim_url,
    params=params,
    headers={
        "User-Agent":
        "HAB-Detection-Project/1.0"
    },
    timeout=60
)

response.raise_for_status()

geojson = response.json()

features = geojson.get(
    "features",
    []
)

if not features:

    raise RuntimeError(
        "Could not retrieve Hussain Sagar polygon."
    )

geometry = features[0]["geometry"]

print(
    f"Original polygon CRS: EPSG:4326"
)

print(
    f"Geometry type: {geometry['type']}"
)


# ============================================================
# STEP 3: REPROJECT POLYGON
# ============================================================

print("\nReprojecting waterbody polygon...")

projected_geometry = transform_geom(
    "EPSG:4326",
    crs.to_string(),
    geometry
)

print(
    f"Polygon successfully transformed "
    f"to {crs}"
)


# ============================================================
# STEP 4: CREATE WATERBODY MASK
# ============================================================

print("\nCreating waterbody mask...")

water_mask = geometry_mask(
    [projected_geometry],
    transform=transform,
    invert=True,
    out_shape=(height, width)
)

water_pixels = int(
    np.sum(water_mask)
)

print(
    f"Waterbody pixels: {water_pixels}"
)


if water_pixels == 0:

    raise RuntimeError(
        "Waterbody mask contains zero pixels. "
        "CRS alignment failed."
    )


# ============================================================
# STEP 5: LOAD SWIN PREDICTIONS
# ============================================================

print("\nLoading Swin predictions...")

prediction_files = sorted(
    PREDICTION_DIR.glob(
        "dynamic_corrected_tile_*_prediction.npy"
    )
)

if len(prediction_files) != 4:

    raise RuntimeError(
        f"Expected 4 prediction tiles, "
        f"found {len(prediction_files)}"
    )

print(
    f"Prediction tiles: "
    f"{len(prediction_files)}"
)


# ============================================================
# STEP 6: RECONSTRUCT 512 × 512
# ============================================================

print("\nReconstructing prediction image...")

full_prediction = np.zeros(
    (512, 512),
    dtype=np.uint8
)

tile_index = 0

for row in range(2):

    for col in range(2):

        prediction = np.load(
            prediction_files[tile_index]
        )

        if prediction.shape != (
            256,
            256
        ):

            raise RuntimeError(
                f"Invalid prediction shape: "
                f"{prediction.shape}"
            )

        y_start = row * 256
        x_start = col * 256

        full_prediction[
            y_start:y_start + 256,
            x_start:x_start + 256
        ] = prediction

        tile_index += 1


# ============================================================
# STEP 7: REMOVE PADDING
# ============================================================

prediction = full_prediction[
    :height,
    :width
]

print(
    f"Prediction cropped to: "
    f"{prediction.shape}"
)


# ============================================================
# STEP 8: APPLY WATERBODY MASK
# ============================================================

print(
    "\nApplying reprojected waterbody boundary..."
)

final_map = np.full(
    (height, width),
    255,
    dtype=np.uint8
)

final_map[
    water_mask
] = prediction[
    water_mask
]


# ============================================================
# STEP 9: WATERBODY-ONLY STATISTICS
# ============================================================

water_predictions = prediction[
    water_mask
]

hab_pixels = int(
    np.sum(water_predictions == 1)
)

non_hab_pixels = int(
    np.sum(water_predictions == 0)
)

valid_pixels = (
    hab_pixels +
    non_hab_pixels
)

hab_percentage = (
    hab_pixels /
    valid_pixels *
    100
)


print("\n" + "=" * 70)
print("WATERBODY-ONLY SWIN RESULT")
print("=" * 70)

print(
    f"Waterbody pixels: "
    f"{valid_pixels}"
)

print(
    f"Non-HAB pixels: "
    f"{non_hab_pixels}"
)

print(
    f"HAB pixels: "
    f"{hab_pixels}"
)

print(
    f"HAB percentage: "
    f"{hab_percentage:.2f}%"
)


# ============================================================
# STEP 10: SAVE GEOTIFF
# ============================================================

print("\nSaving geographic HAB map...")

profile.update(
    count=1,
    dtype="uint8",
    nodata=255,
    compress="lzw"
)

with rasterio.open(
    OUTPUT_FILE,
    "w",
    **profile
) as dst:

    dst.write(
        final_map,
        1
    )

    dst.set_band_description(
        1,
        "Swin_HAB_Prediction"
    )


print(
    f"Saved: {OUTPUT_FILE}"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print(
    "The OSM waterbody polygon was correctly "
    "reprojected to the Sentinel-2 CRS."
)

print(
    "Swin predictions were restricted to "
    "the actual waterbody."
)

print(
    "255 = outside waterbody / NoData"
)

print(
    "0 = predicted non-HAB"
)

print(
    "1 = predicted HAB"
)

print("=" * 70)