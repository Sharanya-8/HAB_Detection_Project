import os
import sys
import requests
import ee


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from utils.gee_utils import initialize_gee


# ============================================================
# WATERBODIES
# ============================================================

WATERBODIES = {
    "Hussain Sagar": {
        "lat": 17.42222,
        "lon": 78.47389,
        "role": "Existing development/training waterbody"
    },

    "Saroor Nagar Lake": {
        "lat": 17.3545,
        "lon": 78.5575,
        "role": "Candidate training waterbody"
    },

    "Osman Sagar": {
        "lat": 17.3861,
        "lon": 78.2977,
        "role": "Candidate training waterbody"
    },

    "Himayat Sagar": {
        "lat": 17.3347,
        "lon": 78.3499,
        "role": "Candidate training waterbody"
    },

    "Shamirpet Lake": {
        "lat": 17.5930,
        "lon": 78.5660,
        "role": "Candidate unseen test waterbody"
    }
}


# ============================================================
# NOMINATIM
# ============================================================

HEADERS = {
    "User-Agent": "HAB-Detection-Project/1.0"
}


def get_waterbody_boundary(name, lat, lon):

    print(f"\nSearching boundary for: {name}")

    url = "https://nominatim.openstreetmap.org/reverse"

    params = {
        "lat": lat,
        "lon": lon,
        "format": "jsonv2",
        "polygon_geojson": 1,
        "zoom": 14
    }

    try:

        response = requests.get(
            url,
            params=params,
            headers=HEADERS,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        geometry = data.get("geojson")

        if geometry is None:

            print("  WARNING: No polygon returned.")
            return None

        geometry_type = geometry.get("type")

        print(f"  Geometry type: {geometry_type}")

        if geometry_type in ["Polygon", "MultiPolygon"]:

            print("  Boundary found.")

            return geometry

        print("  WARNING: Geometry is not Polygon/MultiPolygon.")

        return None

    except Exception as e:

        print(f"  ERROR: {e}")

        return None


# ============================================================
# CHECK SENTINEL-2
# ============================================================

def check_sentinel2(geometry):

    try:

        ee_geometry = ee.Geometry(geometry)

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(ee_geometry)
            .filterDate("2016-01-01", "2027-01-01")
        )

        count = collection.size().getInfo()

        return count

    except Exception as e:

        print(f"  Sentinel-2 check error: {e}")

        return 0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 45 - CHECK MULTI-WATERBODY DATA AVAILABILITY")
    print("=" * 70)

    # --------------------------------------------------------
    # INITIALIZE GEE
    # --------------------------------------------------------

    print("\nInitializing Google Earth Engine...")

    initialize_gee()

    print("GEE initialized successfully.")

    print("\nChecking waterbodies...")
    print("-" * 70)

    results = []

    for name, info in WATERBODIES.items():

        lat = info["lat"]
        lon = info["lon"]
        role = info["role"]

        print()
        print("=" * 70)
        print(name)
        print("=" * 70)

        print(f"Latitude : {lat}")
        print(f"Longitude: {lon}")
        print(f"Role     : {role}")

        # ----------------------------------------------------
        # BOUNDARY
        # ----------------------------------------------------

        geometry = get_waterbody_boundary(
            name,
            lat,
            lon
        )

        if geometry is None:

            results.append({
                "name": name,
                "boundary": "NO",
                "sentinel2_images": 0,
                "role": role
            })

            continue

        # ----------------------------------------------------
        # SENTINEL-2
        # ----------------------------------------------------

        image_count = check_sentinel2(geometry)

        print(
            f"  Sentinel-2 images available "
            f"(2016-2026): {image_count}"
        )

        results.append({
            "name": name,
            "boundary": "YES",
            "sentinel2_images": image_count,
            "role": role
        })


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 45 SUMMARY")
    print("=" * 70)

    print()

    for result in results:

        print(
            f"{result['name']:<22} | "
            f"Boundary: {result['boundary']:<3} | "
            f"S2 images: {result['sentinel2_images']:<5} | "
            f"{result['role']}"
        )

    print()
    print("=" * 70)
    print("STEP 45 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()