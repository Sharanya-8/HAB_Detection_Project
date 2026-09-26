import requests
import geopandas as gpd
from shapely.geometry import shape
from pathlib import Path


# ============================================================
# STEP 56 - DOWNLOAD ACTUAL OSM WATERBODY BOUNDARIES
# ============================================================

print("=" * 70)
print("STEP 56 - DOWNLOAD OSM WATERBODY BOUNDARIES")
print("=" * 70)


# ------------------------------------------------------------
# Output folder
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    "data/labels/hab_masks/training_waterbodies"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Exact OSM waterbody IDs
# ------------------------------------------------------------

waterbodies = {
    "Saroor_Nagar": "W27934981",
    "Himayat_Sagar": "W362966095",
    "Shamirpet_Lake": "W45056467",
}


# ------------------------------------------------------------
# Nominatim API
# ------------------------------------------------------------

URL = "https://nominatim.openstreetmap.org/lookup"

HEADERS = {
    "User-Agent": "HAB-Detection-Project/1.0"
}


# ------------------------------------------------------------
# Download each waterbody
# ------------------------------------------------------------

for name, osm_id in waterbodies.items():

    print()
    print("=" * 70)
    print(f"Downloading: {name}")
    print(f"OSM ID: {osm_id}")
    print("=" * 70)

    params = {
        "osm_ids": osm_id,
        "format": "json",
        "polygon_geojson": 1
    }

    try:

        response = requests.get(
            URL,
            params=params,
            headers=HEADERS,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        # ----------------------------------------------------
        # Check response
        # ----------------------------------------------------

        if not data:
            print("ERROR: No OSM feature returned.")
            continue

        feature = data[0]

        print(
            "OSM name:",
            feature.get("display_name")
        )

        print(
            "OSM type:",
            feature.get("type")
        )

        print(
            "OSM class:",
            feature.get("class")
        )

        # ----------------------------------------------------
        # Get GeoJSON geometry
        # ----------------------------------------------------

        geometry = feature.get("geojson")

        if geometry is None:
            print("ERROR: No GeoJSON geometry returned.")
            continue

        # ----------------------------------------------------
        # Convert GeoJSON geometry to Shapely geometry
        # ----------------------------------------------------

        shapely_geometry = shape(geometry)

        print(
            "Geometry type:",
            shapely_geometry.geom_type
        )

        # ----------------------------------------------------
        # Create GeoDataFrame
        # ----------------------------------------------------

        gdf = gpd.GeoDataFrame(
            [{
                "name": name,
                "osm_id": osm_id,
                "osm_name": feature.get(
                    "display_name"
                ),
            }],
            geometry=[shapely_geometry],
            crs="EPSG:4326"
        )

        # ----------------------------------------------------
        # Repair geometry if necessary
        # ----------------------------------------------------

        if not gdf.geometry.iloc[0].is_valid:

            print(
                "Geometry is invalid."
            )

            print(
                "Attempting geometry repair..."
            )

            gdf["geometry"] = (
                gdf.geometry.buffer(0)
            )

        # ----------------------------------------------------
        # Final validity check
        # ----------------------------------------------------

        is_valid = (
            gdf.geometry.iloc[0].is_valid
        )

        print(
            "Geometry valid:",
            is_valid
        )

        # ----------------------------------------------------
        # Calculate area
        # EPSG:32644 = UTM Zone 44N
        # Suitable for Hyderabad region
        # ----------------------------------------------------

        area_gdf = gdf.to_crs(
            "EPSG:32644"
        )

        area_m2 = (
            area_gdf.geometry.iloc[0].area
        )

        area_ha = area_m2 / 10000

        print(
            f"Area: {area_ha:.2f} hectares"
        )

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        minx, miny, maxx, maxy = (
            gdf.total_bounds
        )

        print(
            f"Longitude: {minx:.6f} → {maxx:.6f}"
        )

        print(
            f"Latitude : {miny:.6f} → {maxy:.6f}"
        )

        # ----------------------------------------------------
        # Save GeoJSON
        # ----------------------------------------------------

        output_file = (
            OUTPUT_DIR
            / f"{name}_OSM_boundary.geojson"
        )

        gdf.to_file(
            output_file,
            driver="GeoJSON"
        )

        print(
            f"Saved: {output_file}"
        )

    except requests.exceptions.RequestException as e:

        print(
            "NETWORK ERROR:",
            e
        )

    except Exception as e:

        print(
            "ERROR:",
            e
        )


# ------------------------------------------------------------
# Completion
# ------------------------------------------------------------

print()
print("=" * 70)
print("STEP 56 COMPLETE")
print("=" * 70)