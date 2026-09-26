"""
STEP 99 — FIND NEARBY WATERBODIES

Finds lakes, reservoirs, rivers and other mapped waterbodies
near a given latitude/longitude using OpenStreetMap Overpass API.
"""

import requests
import json
import os
import math


# ============================================================
# USER INPUT
# ============================================================

LATITUDE = 17.4222
LONGITUDE = 78.4739

# Search radius in metres
RADIUS_METERS = 10000


# ============================================================
# OVERPASS API
# ============================================================

OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"


# ============================================================
# CREATE QUERY
# ============================================================

query = f"""
[out:json][timeout:60];

(
  way["natural"="water"](around:{RADIUS_METERS},{LATITUDE},{LONGITUDE});
  relation["natural"="water"](around:{RADIUS_METERS},{LATITUDE},{LONGITUDE});

  way["water"="lake"](around:{RADIUS_METERS},{LATITUDE},{LONGITUDE});
  relation["water"="lake"](around:{RADIUS_METERS},{LATITUDE},{LONGITUDE});

  way["water"="reservoir"](around:{RADIUS_METERS},{LATITUDE},{LONGITUDE});
  relation["water"="reservoir"](around:{RADIUS_METERS},{LATITUDE},{LONGITUDE});

  way["waterway"="river"](around:{RADIUS_METERS},{LATITUDE},{LONGITUDE});
  relation["waterway"="river"](around:{RADIUS_METERS},{LATITUDE},{LONGITUDE});

  way["waterway"="canal"](around:{RADIUS_METERS},{LATITUDE},{LONGITUDE});
  relation["waterway"="canal"](around:{RADIUS_METERS},{LATITUDE},{LONGITUDE});
);

out center tags;
"""


# ============================================================
# SEND REQUEST
# ============================================================

print("=" * 70)
print("STEP 99 — NEARBY WATERBODY DISCOVERY")
print("=" * 70)

print()
print(f"Latitude       : {LATITUDE}")
print(f"Longitude      : {LONGITUDE}")
print(f"Search radius  : {RADIUS_METERS / 1000:.1f} km")
print()

print("Querying OpenStreetMap...")
print()


try:

    headers = {
        "User-Agent": "HAB-Detection-Project/1.0"
    }

    response = requests.post(
        OVERPASS_URL,
        data=query,
        headers=headers,
        timeout=90
    )

    response.raise_for_status()

    data = response.json()

except Exception as e:

    print("ERROR while querying OpenStreetMap:")
    print(e)
    raise SystemExit


elements = data.get("elements", [])

print(f"Raw waterbody results: {len(elements)}")
print()


# ============================================================
# PROCESS RESULTS
# ============================================================

waterbodies = []

seen = set()

for element in elements:

    element_id = element.get("id")

    tags = element.get("tags", {})

    name = tags.get("name")

    if not name:
        name = tags.get("name:en")

    if not name:
        name = "Unnamed waterbody"

    natural_type = tags.get("natural")
    water_type = tags.get("water")
    waterway_type = tags.get("waterway")

    # Determine waterbody type
    if water_type == "lake":
        wb_type = "Lake"

    elif water_type == "reservoir":
        wb_type = "Reservoir"

    elif natural_type == "water":
        wb_type = "Waterbody"

    elif waterway_type == "river":
        wb_type = "River"

    elif waterway_type == "canal":
        wb_type = "Canal"

    else:
        wb_type = "Waterbody"

    # Coordinates
    center = element.get("center", {})

    lat = center.get("lat")
    lon = center.get("lon")

    if lat is None or lon is None:
        lat = element.get("lat")
        lon = element.get("lon")

    # Remove duplicates
    key = (
        str(name).lower().strip(),
        wb_type
    )

    if key in seen:
        continue

    seen.add(key)

    waterbodies.append(
        {
            "osm_id": element_id,
            "name": name,
            "type": wb_type,
            "latitude": lat,
            "longitude": lon
        }
    )


# ============================================================
# CALCULATE DISTANCE
# ============================================================

def distance_km(lat1, lon1, lat2, lon2):

    if lat1 is None or lon1 is None:
        return 999999

    R = 6371.0

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    dlat = lat2_rad - lat1_rad
    dlon = math.radians(lon2) - math.radians(lon1)

    a = (
        math.sin(dlat / 2) ** 2
        +
        math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return R * c


for wb in waterbodies:

    wb["distance_km"] = distance_km(
        LATITUDE,
        LONGITUDE,
        wb["latitude"],
        wb["longitude"]
    )


# ============================================================
# SORT BY DISTANCE
# ============================================================

waterbodies.sort(
    key=lambda x: x["distance_km"]
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 70)
print("NEARBY WATERBODIES")
print("=" * 70)

print()

if not waterbodies:

    print("No waterbodies found.")

else:

    for i, wb in enumerate(waterbodies, start=1):

        print(
            f"{i:02d}. "
            f"{wb['name']} | "
            f"{wb['type']} | "
            f"{wb['distance_km']:.2f} km"
        )

        print(
            f"    Coordinates: "
            f"{wb['latitude']}, "
            f"{wb['longitude']}"
        )

        print(
            f"    OSM ID: {wb['osm_id']}"
        )

        print()


# ============================================================
# SAVE RESULTS
# ============================================================

output_directory = os.path.join(
    "results",
    "waterbody_discovery"
)

os.makedirs(
    output_directory,
    exist_ok=True
)


output_file = os.path.join(
    output_directory,
    "nearby_waterbodies.json"
)


with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "search_location": {
                "latitude": LATITUDE,
                "longitude": LONGITUDE
            },
            "radius_meters": RADIUS_METERS,
            "waterbodies": waterbodies
        },
        f,
        indent=2,
        ensure_ascii=False
    )


print("=" * 70)
print("STEP 99 COMPLETE")
print("=" * 70)

print()
print("Saved:")
print(
    os.path.abspath(output_file)
)