import os
import sys
import ee
import geemap
import geopandas as gpd
from shapely.geometry import Point


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
    "training_waterbodies"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CORRECTED WATERBODY CENTERS
# ============================================================

WATERBODIES = {

    # Saroornagar Lake
    "Saroor_Nagar": {
        "lat": 17.3539,
        "lon": 78.5270
    },

    # Himayat Sagar
    "Himayat_Sagar": {
        "lat": 17.28233,
        "lon": 78.36119
    },

    # Shamirpet Lake
    "Shamirpet_Lake": {
        "lat": 17.60967,
        "lon": 78.55663
    }
}


# ============================================================
# CREATE JRC WATER POLYGONS
# ============================================================

def get_water_polygons(lat, lon):

    point = ee.Geometry.Point(
        [lon, lat]
    )

    # 5 km local search
    aoi = point.buffer(5000)

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

    polygons = water_mask.reduceToVectors(
        geometry=aoi,
        scale=30,
        geometryType="polygon",
        eightConnected=True,
        labelProperty="water",
        maxPixels=1e9
    )

    return polygons


# ============================================================
# SELECT CORRECT POLYGON
# ============================================================

def select_correct_polygon(
    gdf,
    lat,
    lon
):

    # --------------------------------------------------------
    # Repair geometry
    # --------------------------------------------------------

    gdf = gdf.copy()

    gdf["geometry"] = (
        gdf.geometry.buffer(0)
    )

    gdf = gdf[
        ~gdf.geometry.is_empty
    ].copy()

    # --------------------------------------------------------
    # Project to UTM
    # --------------------------------------------------------

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
    # FIRST: polygon containing center
    # --------------------------------------------------------

    containing = projected[
        projected.geometry.contains(
            center
        )
    ]

    if len(containing) > 0:

        selected = containing.nlargest(
            1,
            "area_ha"
        )

        print(
            "Center is inside a water polygon."
        )

        print(
            f"Selected area: "
            f"{selected.iloc[0]['area_ha']:.2f} ha"
        )

        return selected.to_crs(
            "EPSG:4326"
        )

    # --------------------------------------------------------
    # SECOND: nearest substantial polygon
    # --------------------------------------------------------

    print(
        "Center is not inside a polygon."
    )

    # Ignore tiny ponds
    substantial = projected[
        projected["area_ha"] >= 10
    ]

    if len(substantial) == 0:

        substantial = projected

    selected = substantial.sort_values(
        "distance_m"
    ).head(1)

    print(
        f"Nearest substantial polygon: "
        f"{selected.iloc[0]['distance_m']:.1f} m"
    )

    print(
        f"Selected area: "
        f"{selected.iloc[0]['area_ha']:.2f} ha"
    )

    return selected.to_crs(
        "EPSG:4326"
    )


# ============================================================
# PROCESS
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
        f"Corrected center:"
    )

    print(
        f"Latitude : {lat}"
    )

    print(
        f"Longitude: {lon}"
    )

    # --------------------------------------------------------
    # GET WATER POLYGONS
    # --------------------------------------------------------

    print()
    print(
        "Creating JRC water polygons..."
    )

    polygons = get_water_polygons(
        lat,
        lon
    )

    # --------------------------------------------------------
    # TEMP FILE
    # --------------------------------------------------------

    temp_file = os.path.join(
        OUTPUT_DIR,
        f"{name}_corrected_temp.geojson"
    )

    print(
        "Downloading polygons..."
    )

    geemap.ee_export_vector(
        polygons,
        filename=temp_file
    )

    # --------------------------------------------------------
    # READ
    # --------------------------------------------------------

    gdf = gpd.read_file(
        temp_file
    )

    print(
        f"Generated polygons: "
        f"{len(gdf)}"
    )

    # --------------------------------------------------------
    # SELECT
    # --------------------------------------------------------

    selected = select_correct_polygon(
        gdf,
        lat,
        lon
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # FINAL AREA
    # --------------------------------------------------------

    projected = selected.to_crs(
        "EPSG:32644"
    )

    area_ha = (
        projected.geometry.area.sum()
        / 10000
    )

    print()
    print(
        "FINAL BOUNDARY:"
    )

    print(
        output_file
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
        "STEP 53 - CORRECT WATERBODY BOUNDARIES"
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
    print("STEP 53 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()