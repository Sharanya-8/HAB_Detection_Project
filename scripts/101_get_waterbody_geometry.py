"""
STEP 101 — GET SELECTED WATERBODY GEOMETRY

Retrieves the actual OpenStreetMap geometry of a selected
waterbody and converts it into GeoJSON.

Handles both:
- OSM ways
- OSM relations

For relations, the member ways are retrieved and assembled
into a polygon.
"""

import requests
import json
import os


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
    "Hussain_Sagar_boundary.geojson"
)


# ============================================================
# OVERPASS API
# ============================================================

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter"
]


# ============================================================
# QUERY
#
# We explicitly request:
# - the selected relation
# - its member ways
# - geometry of those ways
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
print("STEP 101 — GET WATERBODY GEOMETRY")
print("=" * 70)

print()
print(f"Waterbody : {WATERBODY_NAME}")
print(f"Type      : {WATERBODY_TYPE}")
print(f"OSM ID    : {OSM_ID}")
print()

print("Querying OpenStreetMap...")
print()


# ============================================================
# REQUEST
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

        print(
            "  SUCCESS"
        )

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
# CHECK REQUEST
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
        "ERROR: Selected relation was not returned."
    )

    raise SystemExit


members = relation.get(
    "members",
    []
)


print(
    f"Relation members: {len(members)}"
)

print()


# ============================================================
# FIND MEMBER WAYS
# ============================================================

way_ids = []

for member in members:

    if member.get("type") != "way":
        continue

    role = member.get(
        "role",
        ""
    )

    way_id = member.get(
        "ref"
    )

    if way_id is None:
        continue

    way_ids.append(
        (
            way_id,
            role
        )
    )


print(
    f"Member ways found: {len(way_ids)}"
)

print()


if not way_ids:

    print(
        "ERROR: No member ways found."
    )

    raise SystemExit


# ============================================================
# EXTRACT WAY GEOMETRIES
# ============================================================

ways = {}

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
            [
                lon,
                lat
            ]
        )

    if len(coordinates) >= 2:

        ways[way_id] = coordinates


print(
    f"Way geometries retrieved: {len(ways)}"
)

print()


if not ways:

    print(
        "ERROR: No usable way geometries found."
    )

    raise SystemExit


# ============================================================
# ASSEMBLE CLOSED RINGS
# ============================================================

rings = []

for way_id, role in way_ids:

    if way_id not in ways:
        continue

    coords = ways[way_id]

    if len(coords) < 2:
        continue

    # OSM member ways may already be closed.
    if coords[0] == coords[-1]:

        rings.append(
            coords
        )


# ============================================================
# IF CLOSED RING EXISTS
# ============================================================

if rings:

    # Use the largest ring as the main waterbody boundary.
    main_ring = max(
        rings,
        key=len
    )

    print(
        f"Closed polygon rings found: {len(rings)}"
    )

    print(
        f"Selected boundary points: {len(main_ring)}"
    )

    coordinates = main_ring


# ============================================================
# OTHERWISE JOIN MEMBER WAYS
# ============================================================

else:

    print(
        "No individually closed rings found."
    )

    print(
        "Attempting to join member ways..."
    )

    unused = []

    for way_id, role in way_ids:

        if way_id in ways:

            unused.append(
                ways[way_id]
            )


    if not unused:

        print(
            "ERROR: No usable coordinates."
        )

        raise SystemExit


    # Start with first way
    assembled = unused.pop(0)

    changed = True

    while unused and changed:

        changed = False

        start = assembled[0]
        end = assembled[-1]

        for i, candidate in enumerate(
            unused
        ):

            candidate_start = candidate[0]
            candidate_end = candidate[-1]

            if end == candidate_start:

                assembled.extend(
                    candidate[1:]
                )

                unused.pop(i)

                changed = True

                break

            elif end == candidate_end:

                assembled.extend(
                    reversed(candidate[:-1])
                )

                unused.pop(i)

                changed = True

                break

            elif start == candidate_end:

                assembled = (
                    candidate[:-1]
                    + assembled
                )

                unused.pop(i)

                changed = True

                break

            elif start == candidate_start:

                assembled = (
                    list(reversed(candidate[1:]))
                    + assembled
                )

                unused.pop(i)

                changed = True

                break


    coordinates = assembled

    print(
        f"Assembled boundary points: "
        f"{len(coordinates)}"
    )


# ============================================================
# CLOSE POLYGON
# ============================================================

if len(coordinates) < 3:

    print(
        "ERROR: Boundary has fewer than 3 points."
    )

    raise SystemExit


if coordinates[0] != coordinates[-1]:

    coordinates.append(
        coordinates[0]
    )


# ============================================================
# CREATE GEOJSON
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

                "osm_element_type":
                    "relation"

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
# SAVE GEOJSON
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
# BOUNDING BOX
# ============================================================

lons = [
    point[0]
    for point in coordinates
]

lats = [
    point[1]
    for point in coordinates
]


min_lon = min(lons)
max_lon = max(lons)

min_lat = min(lats)
max_lat = max(lats)


# ============================================================
# OUTPUT
# ============================================================

print()
print("=" * 70)
print("BOUNDARY INFORMATION")
print("=" * 70)

print()

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

print(
    f"Boundary points   : {len(coordinates)}"
)

print()

print("=" * 70)
print("STEP 101 COMPLETE")
print("=" * 70)

print()

print("Saved boundary:")

print(
    os.path.abspath(OUTPUT_FILE)
)

print()