import os
import sys
import ee
import geemap
import geopandas as gpd
from shapely.geometry import Point


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from utils.gee_utils import initialize_gee


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "labels",
    "hab_masks",
    "training_waterbodies"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# WATERBODY CENTERS
# ============================================================

WATERBODIES = {
    "Saroor_Nagar": {
        "lat": 17.3545,
        "lon": 78.5575
    },

    "Himayat_Sagar": {
        "lat": 17.3347,
        "lon": 78.3499
    },

    "Shamirpet_Lake": {
        "lat": 17.5930,
        "lon": 78.5660
    }
}


# ============================================================
# CREATE LOCAL WATER MASK
# ============================================================

def create_water_polygons(
    lat,
    lon
):

    center = ee.Geometry.Point(
        [lon, lat]
    )

    # Smaller local search area
    # to avoid selecting unrelated lakes
    aoi = center.buffer(
        3000
    )

    water_occurrence = (
        ee.Image(
            "JRC/GSW1_4/GlobalSurfaceWater"
        )
        .select("occurrence")
    )

    water_mask = (
        water_occurrence
        .gte(50)
        .selfMask()
        .clip(aoi)
    )

    vectors = water_mask.reduceToVectors(
        geometry=aoi,
        scale=30,
        geometryType="polygon",
        eightConnected=True,
        labelProperty="water",
        maxPixels=1e9
    )

    return vectors


# ============================================================
# SELECT CORRECT POLYGON
# ============================================================

def select_polygon(
    gdf,
    lat,
    lon
):

    # Repair geometries
    gdf = gdf.copy()

    gdf["geometry"] = (
        gdf.geometry.buffer(0)
    )

    gdf = gdf[
        ~gdf.geometry.is_empty
    ].copy()

    # Convert to projected CRS
    projected = gdf.to_crs(
        "EPSG:32644"
    )

    center = gpd.GeoSeries(
        [
            Point(
                lon,
                lat
            )
        ],
        crs="EPSG:4326"
    ).to_crs(
        "EPSG:32644"
    ).iloc[0]

    projected["area_ha"] = (
        projected.geometry.area
        / 10000
    )

    projected["distance_m"] = (
        projected.geometry.distance(
            center
        )
    )

    # --------------------------------------------------------
    # Prefer polygon containing center
    # --------------------------------------------------------

    containing = projected[
        projected.geometry.contains(
            center
        )
    ]

    if len(containing) > 0:

        # If multiple, choose largest
        selected = containing.nlargest(
            1,
            "area_ha"
        )

        print(
            "Selected polygon containing "
            "reference point."
        )

        return selected.to_crs(
            "EPSG:4326"
        )

    # --------------------------------------------------------
    # Otherwise choose nearest LARGE polygon
    # --------------------------------------------------------

    # Ignore tiny water patches
    large = projected[
        projected["area_ha"] >= 20
    ]

    if len(large) == 0:

        large = projected

    selected = large.sort_values(
        "distance_m"
    ).head(1)

    print(
        "No polygon contains the reference point."
    )

    print(
        "Selected nearest substantial "
        "water polygon."
    )

    print(
        f"Distance: "
        f"{selected.iloc[0]['distance_m']:.1f} m"
    )

    print(
        f"Area: "
        f"{selected.iloc[0]['area_ha']:.2f} ha"
    )

    return selected.to_crs(
        "EPSG:4326"
    )


# ============================================================
# PROCESS ONE WATERBODY
# ============================================================

def process_waterbody(
    name,
    lat,
    lon
):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Reference: {lat}, {lon}"
    )

    print(
        "Creating local JRC water polygons..."
    )

    vectors = create_water_polygons(
        lat,
        lon
    )

    temp_file = os.path.join(
        OUTPUT_DIR,
        f"{name}_temp.geojson"
    )

    print(
        "Downloading polygons..."
    )

    geemap.ee_export_vector(
        vectors,
        filename=temp_file
    )

    gdf = gpd.read_file(
        temp_file
    )

    print(
        f"Polygons generated: "
        f"{len(gdf)}"
    )

    selected = select_polygon(
        gdf,
        lat,
        lon
    )

    if selected is None or len(selected) == 0:

        print(
            "ERROR: Could not select "
            "a suitable polygon."
        )

        return

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{name}_boundary.geojson"
    )

    selected[
        ["geometry"]
    ].to_file(
        output_file,
        driver="GeoJSON"
    )

    print()
    print(
        f"FINAL BOUNDARY:"
    )

    print(
        output_file
    )

    # --------------------------------------------------------
    # Final area
    # --------------------------------------------------------

    projected = selected.to_crs(
        "EPSG:32644"
    )

    area_ha = (
        projected.geometry.area.sum()
        / 10000
    )

    print(
        f"Final area: "
        f"{area_ha:.2f} hectares"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "STEP 52 - CREATE REMAINING "
        "WATERBODY BOUNDARIES"
    )
    print("=" * 70)

    print()
    print(
        "Initializing Google Earth Engine..."
    )

    initialize_gee()

    print(
        "GEE initialized successfully."
    )

    for name, info in WATERBODIES.items():

        try:

            process_waterbody(
                name,
                info["lat"],
                info["lon"]
            )

        except Exception as e:

            print()
            print(
                f"ERROR processing {name}:"
            )

            print(e)

    print()
    print("=" * 70)
    print("STEP 52 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()