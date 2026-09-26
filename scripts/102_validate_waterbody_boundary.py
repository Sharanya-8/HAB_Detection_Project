"""
STEP 102 — RECONSTRUCT AND VALIDATE WATERBODY BOUNDARY

Reconstructs the complete boundary of an OSM waterbody relation
from its member ways.

The script:
1. Retrieves the OSM relation and member ways.
2. Extracts only OUTER boundary ways.
3. Combines them into linework.
4. Polygonizes the linework.
5. Selects the most suitable polygon.
6. Calculates area and bounding box.
7. Saves the validated boundary as GeoJSON.

This is designed to work with OSM relations such as:
- Hussain Sagar
- large lakes
- reservoirs
- other mapped waterbody relations
"""

import requests
import json
import os

from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.ops import unary_union, polygonize
from pyproj import Geod


# ============================================================
# SELECTED WATERBODY
# ============================================================

WATERBODY_NAME = "Hussain Sagar"

OSM_ID = 2833155

WATERBODY_TYPE = "Lake"


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIRECTORY = os.path.join(
    "results",
    "waterbody_discovery",
    "selected_waterbody"
)

os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "Hussain_Sagar_validated_boundary.geojson"
)


# ============================================================
# OVERPASS SERVERS
# ============================================================

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter"
]


# ============================================================
# OVERPASS QUERY
# ============================================================

query = f"""
[out:json][timeout:60];

relation({OSM_ID});

out body;

>;

out geom;
"""


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 102 — RECONSTRUCT AND VALIDATE WATERBODY BOUNDARY")
print("=" * 70)

print()
print(f"Waterbody : {WATERBODY_NAME}")
print(f"Type      : {WATERBODY_TYPE}")
print(f"OSM ID    : {OSM_ID}")
print()


# ============================================================
# REQUEST OSM
# ============================================================

headers = {
    "User-Agent": "HAB-Detection-Project/1.0"
}

data = None


for index, url in enumerate(
    OVERPASS_URLS,
    start=1
):

    print(
        f"Attempt {index}/{len(OVERPASS_URLS)}:"
    )

    print(
        f"  {url}"
    )

    try:

        response = requests.post(
            url,
            data=query,
            headers=headers,
            timeout=90
        )

        response.raise_for_status()

        data = response.json()

        print("  SUCCESS")
        print()

        break

    except requests.exceptions.Timeout:

        print(
            "  Timeout — trying next server..."
        )

        print()

    except requests.exceptions.RequestException as e:

        print(
            f"  Request failed: {e}"
        )

        print()


# ============================================================
# CHECK RESPONSE
# ============================================================

if data is None:

    print("=" * 70)
    print("ERROR")
    print("=" * 70)

    print()
    print(
        "All Overpass servers failed."
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
# FIND RELATION
# ============================================================

relation = None

for element in elements:

    if (
        element.get("type") == "relation"
        and element.get("id") == OSM_ID
    ):

        relation = element

        break


if relation is None:

    print(
        "ERROR: OSM relation was not found."
    )

    raise SystemExit


# ============================================================
# GET RELATION MEMBERS
# ============================================================

members = relation.get(
    "members",
    []
)


outer_way_ids = []

inner_way_ids = []


for member in members:

    if member.get("type") != "way":
        continue

    way_id = member.get(
        "ref"
    )

    role = member.get(
        "role",
        ""
    )

    if way_id is None:
        continue

    if role == "outer":

        outer_way_ids.append(
            way_id
        )

    elif role == "inner":

        inner_way_ids.append(
            way_id
        )


print(
    f"Relation members : {len(members)}"
)

print(
    f"Outer ways       : {len(outer_way_ids)}"
)

print(
    f"Inner ways       : {len(inner_way_ids)}"
)

print()


if not outer_way_ids:

    print(
        "ERROR: No OUTER member ways found."
    )

    raise SystemExit


# ============================================================
# STORE WAY GEOMETRIES
# ============================================================

way_geometries = {}


for element in elements:

    if element.get("type") != "way":
        continue

    way_id = element.get(
        "id"
    )

    geometry = element.get(
        "geometry",
        []
    )

    if way_id is None:
        continue

    if len(geometry) < 2:
        continue

    coordinates = []

    for point in geometry:

        lat = point.get("lat")
        lon = point.get("lon")

        if lat is None or lon is None:
            continue

        coordinates.append(
            (
                lon,
                lat
            )
        )

    if len(coordinates) >= 2:

        way_geometries[way_id] = coordinates


print(
    f"Usable way geometries: "
    f"{len(way_geometries)}"
)

print()


# ============================================================
# CREATE OUTER LINEWORK
# ============================================================

outer_lines = []


for way_id in outer_way_ids:

    if way_id not in way_geometries:
        continue

    coords = way_geometries[
        way_id
    ]

    if len(coords) < 2:
        continue

    try:

        line = LineString(
            coords
        )

        if line.is_valid:

            outer_lines.append(
                line
            )

    except Exception:

        continue


print(
    f"Outer boundary lines: "
    f"{len(outer_lines)}"
)

print()


if not outer_lines:

    print(
        "ERROR: Could not create outer boundary lines."
    )

    raise SystemExit


# ============================================================
# COMBINE LINEWORK
# ============================================================

print(
    "Combining outer boundary linework..."
)

combined_lines = unary_union(
    outer_lines
)


# ============================================================
# POLYGONIZE
# ============================================================

print(
    "Polygonizing boundary..."
)

polygons = list(
    polygonize(
        combined_lines
    )
)


print(
    f"Polygons reconstructed: "
    f"{len(polygons)}"
)

print()


if not polygons:

    print(
        "ERROR: Polygonization failed."
    )

    raise SystemExit


# ============================================================
# SELECT MAIN POLYGON
# ============================================================

# Calculate geodesic area for each polygon.
geod = Geod(
    ellps="WGS84"
)


def geodesic_area_hectares(
    polygon
):

    if polygon.is_empty:
        return 0.0

    lon, lat = polygon.exterior.xy

    area_m2, _ = geod.polygon_area_perimeter(
        lon,
        lat
    )

    return abs(area_m2) / 10000.0


polygon_areas = []


for polygon in polygons:

    area_ha = geodesic_area_hectares(
        polygon
    )

    polygon_areas.append(
        (
            polygon,
            area_ha
        )
    )


polygon_areas.sort(
    key=lambda x: x[1],
    reverse=True
)


main_polygon = polygon_areas[0][0]

main_area_ha = polygon_areas[0][1]


# ============================================================
# BASIC GEOMETRY VALIDATION
# ============================================================

print("=" * 70)
print("RECONSTRUCTED GEOMETRY")
print("=" * 70)

print()

print(
    f"Main polygon area : "
    f"{main_area_ha:.2f} ha"
)

print(
    f"Main polygon area : "
    f"{main_area_ha / 100:.2f} km²"
)

print(
    f"Polygon valid     : "
    f"{main_polygon.is_valid}"
)

print(
    f"Polygon empty     : "
    f"{main_polygon.is_empty}"
)

print()


# ============================================================
# BOUNDING BOX
# ============================================================

min_lon, min_lat, max_lon, max_lat = (
    main_polygon.bounds
)


print(
    f"Minimum longitude : {min_lon}"
)

print(
    f"Maximum longitude : {max_lon}"
)

print(
    f"Minimum latitude  : {min_lat}"
)

print(
    f"Maximum latitude  : {max_lat}"
)

print()


# ============================================================
# SANITY CHECK FOR HUSSAIN SAGAR
# ============================================================

if WATERBODY_NAME == "Hussain Sagar":

    expected_min_lon = 78.4623422
    expected_max_lon = 78.4867613

    expected_min_lat = 17.4085927
    expected_max_lat = 17.4373613

    expected_area_ha = 453.97

    print("=" * 70)
    print("HUSSAIN SAGAR SANITY CHECK")
    print("=" * 70)

    print()

    print(
        f"Expected approximate area : "
        f"{expected_area_ha:.2f} ha"
    )

    print(
        f"Retrieved area            : "
        f"{main_area_ha:.2f} ha"
    )

    print()

    print(
        "Expected bounding box:"
    )

    print(
        f"  Longitude: "
        f"{expected_min_lon} → "
        f"{expected_max_lon}"
    )

    print(
        f"  Latitude : "
        f"{expected_min_lat} → "
        f"{expected_max_lat}"
    )

    print()

    area_difference = abs(
        main_area_ha - expected_area_ha
    )

    area_difference_percent = (
        area_difference
        / expected_area_ha
        * 100
    )

    print(
        f"Area difference: "
        f"{area_difference:.2f} ha "
        f"({area_difference_percent:.2f}%)"
    )

    print()

    bbox_overlap = (
        min_lon <= expected_max_lon
        and max_lon >= expected_min_lon
        and min_lat <= expected_max_lat
        and max_lat >= expected_min_lat
    )

    print(
        f"Bounding-box overlap: "
        f"{bbox_overlap}"
    )

    print()


# ============================================================
# CONVERT TO GEOJSON
# ============================================================

coordinates = [
    [
        float(lon),
        float(lat)
    ]
    for lon, lat in main_polygon.exterior.coords
]


# ============================================================
# GEOJSON
# ============================================================

geojson = {

    "type": "FeatureCollection",

    "features": [

        {
            "type": "Feature",

            "properties": {

                "name":
                    WATERBODY_NAME,

                "type":
                    WATERBODY_TYPE,

                "osm_id":
                    OSM_ID,

                "area_hectares":
                    main_area_ha,

                "area_km2":
                    main_area_ha / 100.0

            },

            "geometry": {

                "type":
                    "Polygon",

                "coordinates": [
                    coordinates
                ]

            }

        }

    ]

}


# ============================================================
# SAVE
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        geojson,
        f,
        indent=2
    )


# ============================================================
# FINAL
# ============================================================

print("=" * 70)
print("STEP 102 COMPLETE")
print("=" * 70)

print()

print(
    "Validated boundary saved to:"
)

print(
    os.path.abspath(
        OUTPUT_FILE
    )
)

print()