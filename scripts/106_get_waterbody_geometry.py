"""
STEP 106 — GET SELECTED WATERBODY GEOMETRY

Reads the waterbody selected in Step 105 and retrieves
its OpenStreetMap geometry.

Handles:
    - Polygon waterbodies
    - Multipolygon waterbodies
    - River/linear water features

Input:
    results/waterbody_discovery/selected_waterbody.json

Output:
    results/waterbody_discovery/selected_waterbody/
        selected_waterbody_geometry.geojson
"""

import os
import json
import requests

from shapely.geometry import (
    LineString,
    Polygon,
    MultiPolygon,
    mapping
)

from shapely.ops import (
    unary_union,
    polygonize
)


# ============================================================
# CONFIGURATION
# ============================================================

SELECTED_WATERBODY_PATH = os.path.join(
    "results",
    "waterbody_discovery",
    "selected_waterbody.json"
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
    "selected_waterbody_geometry.geojson"
)

OVERPASS_URL = (
    "https://overpass.kumi.systems/api/interpreter"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 106 — GET SELECTED WATERBODY GEOMETRY")
print("=" * 70)

print()


# ============================================================
# CHECK INPUT
# ============================================================

if not os.path.exists(
    SELECTED_WATERBODY_PATH
):

    print(
        "ERROR: Selected waterbody file not found:"
    )

    print(
        os.path.abspath(
            SELECTED_WATERBODY_PATH
        )
    )

    raise SystemExit


# ============================================================
# READ SELECTED WATERBODY
# ============================================================

with open(
    SELECTED_WATERBODY_PATH,
    "r",
    encoding="utf-8"
) as file:

    selected = json.load(file)


name = selected.get(
    "name",
    "Unnamed waterbody"
)

osm_type = selected.get(
    "osm_type"
)

osm_id = selected.get(
    "osm_id"
)

water_type = selected.get(
    "type",
    "waterbody"
)


print("Selected waterbody:")
print()

print(
    f"Name     : {name}"
)

print(
    f"Type     : {water_type}"
)

print(
    f"OSM type : {osm_type}"
)

print(
    f"OSM ID   : {osm_id}"
)

print()


# ============================================================
# VALIDATE OSM INFORMATION
# ============================================================

if osm_type not in {
    "way",
    "relation",
    "node"
}:

    print(
        "ERROR: Unsupported OSM element type:"
    )

    print(
        osm_type
    )

    raise SystemExit


if osm_id is None:

    print(
        "ERROR: OSM ID is missing."
    )

    raise SystemExit


# ============================================================
# BUILD OVERPASS QUERY
# ============================================================

if osm_type == "way":

    query = f"""
    [out:json][timeout:120];

    way({osm_id});

    out body geom;
    """

elif osm_type == "relation":

    query = f"""
    [out:json][timeout:120];

    relation({osm_id});

    >;

    out body geom;
    """

else:

    query = f"""
    [out:json][timeout:120];

    node({osm_id});

    out body;
    """


# ============================================================
# REQUEST OSM
# ============================================================

print(
    "Retrieving geometry from OpenStreetMap..."
)

print()

headers = {
    "User-Agent": "HAB-Detection-Project/1.0"
}

try:

    response = requests.post(
        OVERPASS_URL,
        data=query,
        headers=headers,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

except Exception as e:

    print("=" * 70)
    print("ERROR")
    print("=" * 70)

    print()

    print(
        "Could not retrieve OSM geometry."
    )

    print()

    print(
        f"Details: {e}"
    )

    raise SystemExit


elements = data.get(
    "elements",
    []
)

print(
    f"OSM elements returned: {len(elements)}"
)

print()


# ============================================================
# NODE
# ============================================================

if osm_type == "node":

    element = None

    for item in elements:

        if (
            item.get("type") == "node"
            and
            item.get("id") == osm_id
        ):

            element = item
            break


    if element is None:

        print(
            "ERROR: OSM node geometry not found."
        )

        raise SystemExit


    latitude = element.get(
        "lat"
    )

    longitude = element.get(
        "lon"
    )


    geometry = {
        "type": "Point",
        "coordinates": [
            longitude,
            latitude
        ]
    }


# ============================================================
# WAY
# ============================================================

elif osm_type == "way":

    way = None

    for item in elements:

        if (
            item.get("type") == "way"
            and
            item.get("id") == osm_id
        ):

            way = item
            break


    if way is None:

        print(
            "ERROR: OSM way not found."
        )

        raise SystemExit


    geometry_data = way.get(
        "geometry",
        []
    )


    if len(geometry_data) < 2:

        print(
            "ERROR: OSM way does not contain enough geometry points."
        )

        raise SystemExit


    coordinates = [
        (
            point["lon"],
            point["lat"]
        )
        for point in geometry_data
    ]


    # --------------------------------------------------------
    # Check whether the way is closed
    # --------------------------------------------------------

    is_closed = (
        coordinates[0]
        ==
        coordinates[-1]
    )


    if is_closed and len(coordinates) >= 4:

        geometry = {
            "type": "Polygon",
            "coordinates": [
                coordinates
            ]
        }

        print(
            "Geometry type: Polygon"
        )

    else:

        geometry = {
            "type": "LineString",
            "coordinates": coordinates
        }

        print(
            "Geometry type: LineString"
        )


# ============================================================
# RELATION
# ============================================================

else:

    # --------------------------------------------------------
    # Separate relation members
    # --------------------------------------------------------

    outer_lines = []
    inner_lines = []

    for item in elements:

        if item.get("type") != "way":

            continue


        geometry_data = item.get(
            "geometry",
            []
        )

        if len(geometry_data) < 2:

            continue


        coordinates = [
            (
                point["lon"],
                point["lat"]
            )
            for point in geometry_data
        ]


        line = LineString(
            coordinates
        )


        role = item.get(
            "tags",
            {}
        ).get(
            "role"
        )


        # Some Overpass responses don't retain
        # relation member roles directly. We therefore
        # also inspect the original relation below.
        if role == "inner":

            inner_lines.append(
                line
            )

        else:

            outer_lines.append(
                line
            )


    # --------------------------------------------------------
    # Try direct polygon construction
    # --------------------------------------------------------

    if outer_lines:

        merged_outer = unary_union(
            outer_lines
        )

        polygons = list(
            polygonize(
                merged_outer
            )
        )

    else:

        polygons = []


    # --------------------------------------------------------
    # If polygons exist
    # --------------------------------------------------------

    if polygons:

        polygons = sorted(
            polygons,
            key=lambda polygon: polygon.area,
            reverse=True
        )


        main_polygon = polygons[0]


        geometry = mapping(
            main_polygon
        )


        print(
            f"Polygons reconstructed: {len(polygons)}"
        )

        print(
            "Main geometry type: Polygon"
        )


    # --------------------------------------------------------
    # Otherwise create a line geometry
    # --------------------------------------------------------

    elif outer_lines:

        merged = unary_union(
            outer_lines
        )


        geometry = mapping(
            merged
        )


        print(
            "No closed polygon reconstructed."
        )

        print(
            "Using OSM line geometry."
        )


    else:

        print(
            "ERROR: Could not reconstruct geometry."
        )

        raise SystemExit


# ============================================================
# CREATE GEOJSON
# ============================================================

geojson = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "name": name,
                "water_type": water_type,
                "osm_type": osm_type,
                "osm_id": osm_id
            },
            "geometry": geometry
        }
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
        geojson,
        file,
        indent=4
    )


# ============================================================
# VALIDATION
# ============================================================

print()

print("=" * 70)
print("GEOMETRY RESULT")
print("=" * 70)

print()

print(
    f"Waterbody : {name}"
)

print(
    f"Type      : {water_type}"
)

print(
    f"OSM ID    : {osm_id}"
)

print(
    f"Geometry  : {geometry['type']}"
)

if geometry["type"] == "LineString":

    print(
        f"Points    : "
        f"{len(geometry['coordinates'])}"
    )

elif geometry["type"] == "Polygon":

    print(
        f"Boundary points : "
        f"{len(geometry['coordinates'][0])}"
    )


print()

print(
    "Geometry saved to:"
)

print(
    os.path.abspath(
        OUTPUT_PATH
    )
)

print()

print("=" * 70)
print("STEP 106 COMPLETE")
print("=" * 70)

print()