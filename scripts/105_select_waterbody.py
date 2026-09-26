"""
STEP 105 — GENERIC WATERBODY SELECTION

Searches for suitable nearby waterbodies using OpenStreetMap
and allows the user to select one.

Input:
    Latitude
    Longitude
    Search radius

Output:
    Selected waterbody information
    Saved JSON file
"""

import os
import json
import math
import requests


# ============================================================
# CONFIGURATION
# ============================================================

OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"

OUTPUT_DIRECTORY = os.path.join(
    "results",
    "waterbody_discovery"
)

os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIRECTORY,
    "selected_waterbody.json"
)


# ============================================================
# USER INPUT
# ============================================================

print("=" * 70)
print("STEP 105 — GENERIC WATERBODY SELECTION")
print("=" * 70)

print()

latitude = float(
    input("Enter latitude: ").strip()
)

longitude = float(
    input("Enter longitude: ").strip()
)

radius_km = float(
    input("Enter search radius in km: ").strip()
)

radius_m = int(
    radius_km * 1000
)

print()

print(
    f"Location : {latitude}, {longitude}"
)

print(
    f"Radius   : {radius_km} km"
)

print()


# ============================================================
# OVERPASS QUERY
# ============================================================

query = f"""
[out:json][timeout:90];

(
    way["natural"="water"](around:{radius_m},{latitude},{longitude});
    relation["natural"="water"](around:{radius_m},{latitude},{longitude});

    way["water"="lake"](around:{radius_m},{latitude},{longitude});
    relation["water"="lake"](around:{radius_m},{latitude},{longitude});

    way["water"="reservoir"](around:{radius_m},{latitude},{longitude});
    relation["water"="reservoir"](around:{radius_m},{latitude},{longitude});

    way["waterway"="river"](around:{radius_m},{latitude},{longitude});
    relation["waterway"="river"](around:{radius_m},{latitude},{longitude});
);

out center tags;
"""


# ============================================================
# SEND REQUEST
# ============================================================

print("Searching OpenStreetMap...")
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
        "Could not retrieve waterbodies from OpenStreetMap."
    )

    print()

    print(
        f"Details: {e}"
    )

    raise SystemExit


# ============================================================
# PROCESS RESULTS
# ============================================================

elements = data.get(
    "elements",
    []
)

print(
    f"Raw waterbody elements found: {len(elements)}"
)

print()


waterbodies = []

seen = set()


# ============================================================
# HELPER — DISTANCE
# ============================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):

    R = 6371.0

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    dlat = (
        lat2
        - lat1
    )

    dlon = math.radians(
        lon2
        - lon1
    )

    a = (
        math.sin(dlat / 2) ** 2
        +
        math.cos(lat1)
        *
        math.cos(lat2)
        *
        math.sin(dlon / 2) ** 2
    )

    c = (
        2
        *
        math.atan2(
            math.sqrt(a),
            math.sqrt(1 - a)
        )
    )

    return R * c


# ============================================================
# EXTRACT WATERBODIES
# ============================================================

for element in elements:

    element_id = element.get(
        "id"
    )

    element_type = element.get(
        "type"
    )

    tags = element.get(
        "tags",
        {}
    )

    name = tags.get(
        "name"
    )

    if not name:

        name = tags.get(
            "name:en"
        )

    if not name:

        name = "Unnamed waterbody"


    # --------------------------------------------------------
    # Determine type
    # --------------------------------------------------------

    water_type = (
        tags.get("water")
        or
        tags.get("waterway")
        or
        tags.get("natural")
        or
        "waterbody"
    )


    # --------------------------------------------------------
    # Keep useful waterbody types
    # --------------------------------------------------------

    suitable_types = {

        "water",
        "lake",
        "reservoir",
        "river",
        "riverbank"
    }

    if water_type not in suitable_types:

        continue


    # --------------------------------------------------------
    # Get center
    # --------------------------------------------------------

    center = element.get(
        "center"
    )

    if center:

        water_lat = center.get(
            "lat"
        )

        water_lon = center.get(
            "lon"
        )

    else:

        water_lat = element.get(
            "lat"
        )

        water_lon = element.get(
            "lon"
        )


    if water_lat is None or water_lon is None:

        continue


    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    unique_key = (
        element_type,
        element_id
    )

    if unique_key in seen:

        continue

    seen.add(
        unique_key
    )


    # --------------------------------------------------------
    # Distance
    # --------------------------------------------------------

    distance_km = haversine_distance(
        latitude,
        longitude,
        water_lat,
        water_lon
    )


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    waterbodies.append(
        {
            "name": name,
            "type": water_type,
            "osm_type": element_type,
            "osm_id": element_id,
            "latitude": water_lat,
            "longitude": water_lon,
            "distance_km": round(
                distance_km,
                3
            )
        }
    )


# ============================================================
# SORT BY DISTANCE
# ============================================================

waterbodies.sort(
    key=lambda x: x["distance_km"]
)


# ============================================================
# LIMIT DISPLAY
# ============================================================

MAX_RESULTS = 50

display_waterbodies = waterbodies[
    :MAX_RESULTS
]


# ============================================================
# NO RESULTS
# ============================================================

if len(display_waterbodies) == 0:

    print("=" * 70)
    print("NO SUITABLE WATERBODIES FOUND")
    print("=" * 70)

    print()

    print(
        "Try increasing the search radius."
    )

    raise SystemExit


# ============================================================
# DISPLAY
# ============================================================

print("=" * 70)
print("NEARBY WATERBODIES")
print("=" * 70)

print()

for index, waterbody in enumerate(
    display_waterbodies,
    start=1
):

    print(
        f"{index:2d}. "
        f"{waterbody['name']} | "
        f"{waterbody['type']} | "
        f"{waterbody['distance_km']:.2f} km"
    )


print()

print(
    f"Showing {len(display_waterbodies)} waterbodies."
)

print()


# ============================================================
# USER SELECTION
# ============================================================

while True:

    try:

        selection = int(
            input(
                "Select a waterbody number: "
            ).strip()
        )

        if (
            1
            <= selection
            <= len(display_waterbodies)
        ):

            break

        print(
            "Invalid selection. "
            "Enter one of the numbers shown above."
        )

    except ValueError:

        print(
            "Please enter a number."
        )


# ============================================================
# SELECTED WATERBODY
# ============================================================

selected_waterbody = display_waterbodies[
    selection - 1
]


# Add search location information

selected_waterbody[
    "search_latitude"
] = latitude

selected_waterbody[
    "search_longitude"
] = longitude

selected_waterbody[
    "search_radius_km"
] = radius_km


# ============================================================
# SAVE
# ============================================================

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        selected_waterbody,
        file,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()

print("=" * 70)
print("SELECTED WATERBODY")
print("=" * 70)

print()

print(
    f"Name      : {selected_waterbody['name']}"
)

print(
    f"Type      : {selected_waterbody['type']}"
)

print(
    f"OSM type  : {selected_waterbody['osm_type']}"
)

print(
    f"OSM ID    : {selected_waterbody['osm_id']}"
)

print(
    f"Latitude  : {selected_waterbody['latitude']}"
)

print(
    f"Longitude : {selected_waterbody['longitude']}"
)

print(
    f"Distance  : "
    f"{selected_waterbody['distance_km']:.3f} km"
)

print()

print(
    "Selection saved to:"
)

print(
    os.path.abspath(
        OUTPUT_PATH
    )
)

print()

print("=" * 70)
print("STEP 105 COMPLETE")
print("=" * 70)

print()