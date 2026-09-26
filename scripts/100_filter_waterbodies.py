"""
STEP 100 — FILTER NEARBY WATERBODIES

Reads the raw OpenStreetMap waterbody discovery results
and creates a cleaned list suitable for user selection.

Raw OSM results are preserved.
"""

import json
import os


# ============================================================
# INPUT / OUTPUT
# ============================================================

INPUT_FILE = os.path.join(
    "results",
    "waterbody_discovery",
    "nearby_waterbodies.json"
)

OUTPUT_DIRECTORY = os.path.join(
    "results",
    "waterbody_discovery"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "filtered_waterbodies.json"
)


# ============================================================
# WATERBODY TYPES TO KEEP
# ============================================================

ALLOWED_TYPES = {
    "Lake",
    "Reservoir",
    "River",
    "Waterbody"
}


# ============================================================
# WATERBODY TYPES TO EXCLUDE
# ============================================================

EXCLUDED_TYPES = {
    "Canal"
}


# ============================================================
# READ RAW RESULTS
# ============================================================

print("=" * 70)
print("STEP 100 — WATERBODY FILTERING")
print("=" * 70)

print()
print("Reading:")
print(os.path.abspath(INPUT_FILE))
print()


if not os.path.exists(INPUT_FILE):

    print("ERROR: Input file does not exist.")
    raise SystemExit


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


raw_waterbodies = data.get(
    "waterbodies",
    []
)


print(
    f"Raw waterbodies: {len(raw_waterbodies)}"
)

print()


# ============================================================
# FILTER
# ============================================================

filtered_waterbodies = []

excluded_waterbodies = []

seen = set()


for wb in raw_waterbodies:

    name = wb.get(
        "name",
        "Unnamed waterbody"
    )

    wb_type = wb.get(
        "type",
        "Waterbody"
    )

    # --------------------------------------------------------
    # Exclude canals
    # --------------------------------------------------------

    if wb_type in EXCLUDED_TYPES:

        excluded_waterbodies.append(wb)

        continue

    # --------------------------------------------------------
    # Keep recognised waterbody types
    # --------------------------------------------------------

    if wb_type not in ALLOWED_TYPES:

        excluded_waterbodies.append(wb)

        continue

    # --------------------------------------------------------
    # Remove duplicate name + type combinations
    # --------------------------------------------------------

    key = (
        name.lower().strip(),
        wb_type
    )

    if key in seen:

        continue

    seen.add(key)

    # --------------------------------------------------------
    # Add selection metadata
    # --------------------------------------------------------

    cleaned = {
        "selection_id": len(filtered_waterbodies) + 1,
        "osm_id": wb.get("osm_id"),
        "name": name,
        "type": wb_type,
        "latitude": wb.get("latitude"),
        "longitude": wb.get("longitude"),
        "distance_km": wb.get("distance_km")
    }

    filtered_waterbodies.append(
        cleaned
    )


# ============================================================
# SORT BY DISTANCE
# ============================================================

filtered_waterbodies.sort(
    key=lambda x: (
        x["distance_km"]
        if x["distance_km"] is not None
        else 999999
    )
)


# Re-number after sorting

for index, wb in enumerate(
    filtered_waterbodies,
    start=1
):

    wb["selection_id"] = index


# ============================================================
# PRINT FILTERED RESULTS
# ============================================================

print("=" * 70)
print("FILTERED WATERBODIES")
print("=" * 70)

print()

for wb in filtered_waterbodies:

    print(
        f"{wb['selection_id']:02d}. "
        f"{wb['name']} | "
        f"{wb['type']} | "
        f"{wb['distance_km']:.2f} km"
    )


print()

print(
    f"Suitable waterbodies: "
    f"{len(filtered_waterbodies)}"
)

print(
    f"Excluded features: "
    f"{len(excluded_waterbodies)}"
)

print()


# ============================================================
# SAVE CLEAN RESULTS
# ============================================================

os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True
)


output_data = {

    "search_location":
        data.get(
            "search_location",
            {}
        ),

    "radius_meters":
        data.get(
            "radius_meters"
        ),

    "total_raw_results":
        len(raw_waterbodies),

    "total_filtered_results":
        len(filtered_waterbodies),

    "waterbodies":
        filtered_waterbodies
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output_data,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# COMPLETE
# ============================================================

print("=" * 70)
print("STEP 100 COMPLETE")
print("=" * 70)

print()
print("Saved:")
print(os.path.abspath(OUTPUT_FILE))
print()