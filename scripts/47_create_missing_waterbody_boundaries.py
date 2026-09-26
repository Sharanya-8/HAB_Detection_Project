import os
import sys
import ee
import geemap
import geopandas as gpd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from utils.gee_utils import initialize_gee


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "labels",
    "hab_masks",
    "waterbodies"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# APPROXIMATE CENTERS
# ============================================================

WATERBODIES = {
    "Osman_Sagar": {
        "lat": 17.3861,
        "lon": 78.2977,
        "radius": 5000
    },

    "Himayat_Sagar": {
        "lat": 17.3347,
        "lon": 78.3499,
        "radius": 5000
    }
}


# ============================================================
# CREATE APPROXIMATE AOI
# ============================================================

def create_aoi(lat, lon, radius):

    point = ee.Geometry.Point(
        [lon, lat]
    )

    return point.buffer(radius)


# ============================================================
# GET WATER MASK
# ============================================================

def get_water_mask(aoi):

    # JRC Global Surface Water
    # Permanent water occurrence

    water_occurrence = (
        ee.Image(
            "JRC/GSW1_4/GlobalSurfaceWater"
        )
        .select("occurrence")
    )

    # Pixels with water occurrence >= 50%
    water = water_occurrence.gte(50)

    # Keep only the AOI
    water = water.clip(aoi)

    return water


# ============================================================
# VECTORIZE WATER MASK
# ============================================================

def create_boundary(name, lat, lon, radius):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Center: {lat}, {lon}"
    )

    print(
        f"Search radius: {radius} meters"
    )

    aoi = create_aoi(
        lat,
        lon,
        radius
    )

    print(
        "Creating JRC Global Surface Water mask..."
    )

    water_mask = get_water_mask(aoi)

    # --------------------------------------------------------
    # VECTORIZE
    # --------------------------------------------------------

    print(
        "Converting water mask to polygons..."
    )

    vectors = water_mask.selfMask().reduceToVectors(
        geometry=aoi,
        scale=30,
        geometryType="polygon",
        eightConnected=True,
        labelProperty="water",
        maxPixels=1e9
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    safe_name = name.replace(
        " ",
        "_"
    )

    temp_file = os.path.join(
        OUTPUT_DIR,
        f"{safe_name}_jrc_boundary.geojson"
    )

    print(
        "Downloading polygon data..."
    )

    geemap.ee_export_vector(
        vectors,
        filename=temp_file
    )

    print(
        f"Saved temporary boundary:"
    )

    print(temp_file)

    # --------------------------------------------------------
    # LOAD VECTOR
    # --------------------------------------------------------

    gdf = gpd.read_file(
        temp_file
    )

    print(
        f"Polygons found: {len(gdf)}"
    )

    if len(gdf) == 0:

        print(
            "ERROR: No water polygons found."
        )

        return

    # --------------------------------------------------------
    # FIND POLYGON NEAR CENTER
    # --------------------------------------------------------

    center_point = gpd.GeoSeries(
        gpd.points_from_xy(
            [lon],
            [lat]
        ),
        crs="EPSG:4326"
    ).iloc[0]

    # Keep polygons containing center
    containing = gdf[
        gdf.geometry.contains(
            center_point
        )
    ]

    if len(containing) > 0:

        selected = containing.copy()

        print(
            "Polygon containing center found."
        )

    else:

        # Use nearest polygon
        gdf["distance"] = gdf.geometry.distance(
            center_point
        )

        selected = gdf.nsmallest(
            1,
            "distance"
        )

        print(
            "No containing polygon found."
        )

        print(
            "Using nearest polygon."
        )

    # --------------------------------------------------------
    # SAVE FINAL BOUNDARY
    # --------------------------------------------------------

    final_file = os.path.join(
        OUTPUT_DIR,
        f"{safe_name}_boundary.geojson"
    )

    selected = selected[
        ["geometry"]
    ]

    selected.to_file(
        final_file,
        driver="GeoJSON"
    )

    print()
    print(
        "FINAL BOUNDARY SAVED:"
    )

    print(final_file)

    # --------------------------------------------------------
    # AREA
    # --------------------------------------------------------

    area_gdf = selected.to_crs(
        "EPSG:32644"
    )

    area_m2 = (
        area_gdf.geometry.area.sum()
    )

    area_ha = area_m2 / 10000

    print(
        f"Approximate boundary area: "
        f"{area_ha:.2f} hectares"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "STEP 47 - CREATE MISSING WATERBODY BOUNDARIES"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # INITIALIZE GEE
    # --------------------------------------------------------

    print()
    print(
        "Initializing Google Earth Engine..."
    )

    initialize_gee()

    print(
        "GEE initialized successfully."
    )

    # --------------------------------------------------------
    # PROCESS WATERBODIES
    # --------------------------------------------------------

    for name, info in WATERBODIES.items():

        try:

            create_boundary(
                name,
                info["lat"],
                info["lon"],
                info["radius"]
            )

        except Exception as e:

            print()
            print(
                f"ERROR processing {name}:"
            )

            print(e)

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "STEP 47 COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()