import requests
import geopandas as gpd
from shapely.geometry import shape
from pathlib import Path

print("=" * 70)
print("STEP 57 - DOWNLOAD HIMAYAT SAGAR RESERVOIR RELATION")
print("=" * 70)

OUTPUT_DIR = Path(
    "data/labels/hab_masks/training_waterbodies"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Actual Himayat Sagar reservoir relation
OSM_ID = "R5411363"

URL = "https://nominatim.openstreetmap.org/lookup"

HEADERS = {
    "User-Agent": "HAB-Detection-Project/1.0"
}

params = {
    "osm_ids": OSM_ID,
    "format": "json",
    "polygon_geojson": 1
}

print(f"OSM relation: {OSM_ID}")
print("Requesting reservoir boundary...")

try:

    response = requests.get(
        URL,
        params=params,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    if not data:
        raise RuntimeError(
            "No OSM feature returned."
        )

    feature = data[0]

    print()
    print("OSM name:")
    print(feature.get("display_name"))

    print()
    print("OSM type:")
    print(feature.get("type"))

    print()
    print("OSM class:")
    print(feature.get("class"))

    geometry = feature.get("geojson")

    if geometry is None:
        raise RuntimeError(
            "No GeoJSON geometry returned."
        )

    # Convert GeoJSON to Shapely
    geom = shape(geometry)

    print()
    print("Geometry type:")
    print(geom.geom_type)

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        [{
            "name": "Himayat_Sagar",
            "osm_id": OSM_ID,
            "osm_name": feature.get(
                "display_name"
            ),
        }],
        geometry=[geom],
        crs="EPSG:4326"
    )

    # Repair if necessary
    if not gdf.geometry.iloc[0].is_valid:

        print()
        print("Geometry invalid.")
        print("Attempting repair...")

        gdf["geometry"] = (
            gdf.geometry.buffer(0)
        )

    print()
    print(
        "Geometry valid:",
        gdf.geometry.iloc[0].is_valid
    )

    # Calculate area in hectares
    area_gdf = gdf.to_crs("EPSG:32644")

    area_m2 = (
        area_gdf.geometry.iloc[0].area
    )

    area_ha = area_m2 / 10000

    print(
        f"Area: {area_ha:.2f} hectares"
    )

    # Bounding box
    minx, miny, maxx, maxy = (
        gdf.total_bounds
    )

    print(
        f"Longitude: {minx:.6f} → {maxx:.6f}"
    )

    print(
        f"Latitude : {miny:.6f} → {maxy:.6f}"
    )

    # Save
    output_file = (
        OUTPUT_DIR
        / "Himayat_Sagar_relation_boundary.geojson"
    )

    gdf.to_file(
        output_file,
        driver="GeoJSON"
    )

    print()
    print(f"Saved: {output_file}")

except requests.exceptions.RequestException as e:

    print()
    print("NETWORK ERROR:")
    print(e)

except Exception as e:

    print()
    print("ERROR:")
    print(e)

print()
print("=" * 70)
print("STEP 57 COMPLETE")
print("=" * 70)