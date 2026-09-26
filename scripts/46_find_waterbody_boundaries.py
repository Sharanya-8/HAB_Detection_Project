import os
import json
import requests


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "labels",
    "hab_masks",
    "waterbodies"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# WATERBODIES
# ============================================================

WATERBODIES = [
    "Osman Sagar, Hyderabad, Telangana, India",
    "Himayat Sagar, Hyderabad, Telangana, India",
    "Shamirpet Lake, Telangana, India"
]


# ============================================================
# NOMINATIM
# ============================================================

URL = "https://nominatim.openstreetmap.org/search"

HEADERS = {
    "User-Agent": "HAB-Detection-Project/1.0"
}


# ============================================================
# SEARCH
# ============================================================

def search_waterbody(query):

    print()
    print("=" * 70)
    print(query)
    print("=" * 70)

    params = {
        "q": query,
        "format": "jsonv2",
        "polygon_geojson": 1,
        "limit": 10
    }

    try:

        response = requests.get(
            URL,
            params=params,
            headers=HEADERS,
            timeout=30
        )

        response.raise_for_status()

        results = response.json()

        print(
            f"Found {len(results)} search results."
        )

        candidates = []

        for i, result in enumerate(results, 1):

            name = result.get(
                "display_name",
                "Unknown"
            )

            osm_type = result.get(
                "osm_type",
                ""
            )

            osm_id = result.get(
                "osm_id",
                ""
            )

            geometry = result.get(
                "geojson"
            )

            geometry_type = (
                geometry.get("type")
                if geometry
                else None
            )

            print()
            print(f"Result {i}")
            print(f"Name     : {name}")
            print(f"OSM type : {osm_type}")
            print(f"OSM ID   : {osm_id}")
            print(
                f"Geometry : {geometry_type}"
            )

            if geometry_type in [
                "Polygon",
                "MultiPolygon"
            ]:

                candidates.append({
                    "display_name": name,
                    "osm_type": osm_type,
                    "osm_id": osm_id,
                    "geometry": geometry
                })

        # ----------------------------------------------------
        # SAVE POLYGON CANDIDATES
        # ----------------------------------------------------

        safe_name = (
            query
            .split(",")[0]
            .strip()
            .replace(" ", "_")
        )

        output_file = os.path.join(
            OUTPUT_DIR,
            f"{safe_name}_nominatim_candidates.json"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                candidates,
                f,
                indent=2
            )

        print()
        print(
            f"Polygon candidates: "
            f"{len(candidates)}"
        )

        print(
            f"Saved: {output_file}"
        )

    except Exception as e:

        print()
        print(
            f"ERROR: {e}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "STEP 46 - FIND NAMED WATERBODY BOUNDARIES"
    )
    print("=" * 70)

    for waterbody in WATERBODIES:

        search_waterbody(waterbody)

    print()
    print("=" * 70)
    print("STEP 46 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()