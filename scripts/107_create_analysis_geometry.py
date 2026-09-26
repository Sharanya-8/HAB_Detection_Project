"""
STEP 107 — CREATE ANALYSIS GEOMETRY

Converts the selected waterbody geometry into a polygon
that can be used for Sentinel-2 water masking.

For:
    Lake / Reservoir / Polygon
        -> use polygon directly

For:
    River / LineString
        -> create a configurable buffer corridor

Input:
    results/waterbody_discovery/selected_waterbody/
    selected_waterbody_geometry.geojson

Output:
    results/waterbody_discovery/selected_waterbody/
    analysis_geometry.geojson
"""

import os
import json

import geopandas as gpd
from shapely.geometry import (
    shape,
    mapping
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = os.path.join(
    "results",
    "waterbody_discovery",
    "selected_waterbody",
    "selected_waterbody_geometry.geojson"
)

OUTPUT_DIRECTORY = os.path.join(
    "results",
    "waterbody_discovery",
    "selected_waterbody"
)

os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIRECTORY,
    "analysis_geometry.geojson"
)


# ============================================================
# RIVER BUFFER
# ============================================================

# Width of analysis corridor on EACH SIDE of a river.
#
# Example:
#     100 m buffer
#     = 200 m total corridor width
#
# We keep this configurable because different rivers
# can require different analysis widths.

RIVER_BUFFER_METERS = 100


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 107 — CREATE ANALYSIS GEOMETRY")
print("=" * 70)

print()


# ============================================================
# CHECK INPUT
# ============================================================

if not os.path.exists(INPUT_PATH):

    print(
        "ERROR: Input geometry not found:"
    )

    print(
        os.path.abspath(INPUT_PATH)
    )

    raise SystemExit


# ============================================================
# READ GEOJSON
# ============================================================

print(
    "Reading selected waterbody geometry..."
)

with open(
    INPUT_PATH,
    "r",
    encoding="utf-8"
) as file:

    data = json.load(file)


features = data.get(
    "features",
    []
)


if len(features) == 0:

    print(
        "ERROR: No geometry found in GeoJSON."
    )

    raise SystemExit


feature = features[0]

properties = feature.get(
    "properties",
    {}
)

geometry_data = feature.get(
    "geometry"
)


if geometry_data is None:

    print(
        "ERROR: Geometry is missing."
    )

    raise SystemExit


# ============================================================
# WATERBODY INFORMATION
# ============================================================

name = properties.get(
    "name",
    "Unnamed waterbody"
)

water_type = properties.get(
    "water_type",
    "waterbody"
)

osm_type = properties.get(
    "osm_type"
)

osm_id = properties.get(
    "osm_id"
)


print()

print(
    f"Waterbody : {name}"
)

print(
    f"Type      : {water_type}"
)

print(
    f"OSM type  : {osm_type}"
)

print(
    f"OSM ID    : {osm_id}"
)

print()


# ============================================================
# CREATE SHAPELY GEOMETRY
# ============================================================

geometry = shape(
    geometry_data
)

print(
    f"Input geometry : {geometry.geom_type}"
)

print()


# ============================================================
# PROCESS POLYGON
# ============================================================

if geometry.geom_type in {
    "Polygon",
    "MultiPolygon"
}:

    print(
        "Polygon waterbody detected."
    )

    print(
        "Using the waterbody boundary directly."
    )

    analysis_geometry = geometry


# ============================================================
# PROCESS LINESTRING
# ============================================================

elif geometry.geom_type in {
    "LineString",
    "MultiLineString"
}:

    print(
        "River/linear waterbody detected."
    )

    print(
        f"Creating {RIVER_BUFFER_METERS} m "
        "buffer on each side..."
    )

    # --------------------------------------------------------
    # Convert geographic coordinates to a metric CRS
    # --------------------------------------------------------

    gdf = gpd.GeoDataFrame(
        [
            {
                "name": name,
                "water_type": water_type
            }
        ],
        geometry=[geometry],
        crs="EPSG:4326"
    )


    # --------------------------------------------------------
    # Use Web Mercator for the buffer operation.
    #
    # The buffer is only an analysis corridor.
    # The final Sentinel-2 mask will be created using
    # the actual image CRS.
    # --------------------------------------------------------

    gdf_metric = gdf.to_crs(
        "EPSG:3857"
    )


    # --------------------------------------------------------
    # Buffer
    # --------------------------------------------------------

    buffered = gdf_metric.geometry.buffer(
        RIVER_BUFFER_METERS
    )


    # --------------------------------------------------------
    # Convert back to geographic CRS
    # --------------------------------------------------------

    buffered_gdf = gpd.GeoDataFrame(
        gdf_metric.drop(
            columns="geometry"
        ),
        geometry=buffered,
        crs="EPSG:3857"
    )


    buffered_gdf = buffered_gdf.to_crs(
        "EPSG:4326"
    )


    analysis_geometry = (
        buffered_gdf.geometry.iloc[0]
    )


else:

    print(
        "ERROR: Unsupported geometry type:"
    )

    print(
        geometry.geom_type
    )

    raise SystemExit


# ============================================================
# VALIDATE
# ============================================================

print()

print(
    "Validating analysis geometry..."
)

if analysis_geometry.is_empty:

    print(
        "ERROR: Analysis geometry is empty."
    )

    raise SystemExit


if not analysis_geometry.is_valid:

    print(
        "Geometry is invalid."
    )

    print(
        "Attempting geometry repair..."
    )

    analysis_geometry = (
        analysis_geometry.buffer(0)
    )


if analysis_geometry.is_empty:

    print(
        "ERROR: Geometry repair failed."
    )

    raise SystemExit


# ============================================================
# CREATE OUTPUT GEOJSON
# ============================================================

output_feature = {
    "type": "Feature",
    "properties": {
        "name": name,
        "water_type": water_type,
        "osm_type": osm_type,
        "osm_id": osm_id,
        "geometry_source": "OpenStreetMap",
        "analysis_geometry": analysis_geometry.geom_type,
        "river_buffer_m": (
            RIVER_BUFFER_METERS
            if geometry.geom_type in {
                "LineString",
                "MultiLineString"
            }
            else None
        )
    },
    "geometry": mapping(
        analysis_geometry
    )
}


output_data = {
    "type": "FeatureCollection",
    "features": [
        output_feature
    ]
}


# ============================================================
# SAVE
# ============================================================

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        output_data,
        file,
        indent=4
    )


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 70)
print("ANALYSIS GEOMETRY RESULT")
print("=" * 70)

print()

print(
    f"Waterbody          : {name}"
)

print(
    f"Original geometry  : "
    f"{geometry.geom_type}"
)

print(
    f"Analysis geometry  : "
    f"{analysis_geometry.geom_type}"
)

if geometry.geom_type in {
    "LineString",
    "MultiLineString"
}:

    print(
        f"River buffer       : "
        f"{RIVER_BUFFER_METERS} m each side"
    )

print(
    f"Valid              : "
    f"{analysis_geometry.is_valid}"
)

print(
    f"Empty              : "
    f"{analysis_geometry.is_empty}"
)

print()

minx, miny, maxx, maxy = (
    analysis_geometry.bounds
)

print(
    f"Bounding box:"
)

print(
    f"  Longitude: "
    f"{minx:.6f} → {maxx:.6f}"
)

print(
    f"  Latitude : "
    f"{miny:.6f} → {maxy:.6f}"
)

print()

print(
    "Saved to:"
)

print(
    os.path.abspath(
        OUTPUT_PATH
    )
)

print()

print("=" * 70)
print("STEP 107 COMPLETE")
print("=" * 70)

print()