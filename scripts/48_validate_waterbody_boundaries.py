import os
import geopandas as gpd
from shapely.geometry import Point


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


BOUNDARY_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "labels",
    "hab_masks",
    "waterbodies"
)


# ============================================================
# WATERBODY CENTERS
# ============================================================

WATERBODIES = {
    "Hussain_Sagar": (17.42222, 78.47389),
    "Saroor_Nagar": (17.3545, 78.5575),
    "Osman_Sagar": (17.3861, 78.2977),
    "Himayat_Sagar": (17.3347, 78.3499),
    "Shamirpet_Lake": (17.5930, 78.5660),
}


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 48 - VALIDATE WATERBODY BOUNDARIES")
    print("=" * 70)

    all_valid = True

    for name, (lat, lon) in WATERBODIES.items():

        print()
        print("=" * 70)
        print(name)
        print("=" * 70)

        boundary_file = os.path.join(
            BOUNDARY_DIR,
            f"{name}_boundary.geojson"
        )

        if not os.path.exists(boundary_file):

            print("ERROR: Boundary file not found.")
            print(boundary_file)

            all_valid = False
            continue

        # ----------------------------------------------------
        # READ
        # ----------------------------------------------------

        gdf = gpd.read_file(
            boundary_file
        )

        print(
            f"Geometry count: {len(gdf)}"
        )

        print(
            f"CRS: {gdf.crs}"
        )

        # ----------------------------------------------------
        # GEOMETRY TYPE
        # ----------------------------------------------------

        geometry_types = (
            gdf.geometry.geom_type.unique()
        )

        print(
            f"Geometry type(s): "
            f"{list(geometry_types)}"
        )

        # ----------------------------------------------------
        # VALIDITY
        # ----------------------------------------------------

        valid = gdf.geometry.is_valid.all()

        print(
            f"Geometry valid: {valid}"
        )

        if not valid:
            all_valid = False

        # ----------------------------------------------------
        # AREA
        # ----------------------------------------------------

        projected = gdf.to_crs(
            "EPSG:32644"
        )

        area_m2 = (
            projected.geometry.area.sum()
        )

        area_ha = area_m2 / 10000

        print(
            f"Area: {area_ha:.2f} hectares"
        )

        # ----------------------------------------------------
        # CENTER CHECK
        # ----------------------------------------------------

        point = gpd.GeoSeries(
            [
                Point(
                    lon,
                    lat
                )
            ],
            crs="EPSG:4326"
        )

        boundary_wgs84 = gdf.to_crs(
            "EPSG:4326"
        )

        inside = boundary_wgs84.geometry.contains(
            point.iloc[0]
        ).any()

        print(
            f"Reference center inside polygon: "
            f"{inside}"
        )

        if not inside:

            print(
                "WARNING: Reference center is "
                "outside the selected boundary."
            )

        # ----------------------------------------------------
        # BOUNDS
        # ----------------------------------------------------

        minx, miny, maxx, maxy = (
            boundary_wgs84.total_bounds
        )

        print(
            f"Bounding box:"
        )

        print(
            f"  Longitude: {minx:.6f} → {maxx:.6f}"
        )

        print(
            f"  Latitude : {miny:.6f} → {maxy:.6f}"
        )

        print(
            "Status: CHECK COMPLETE"
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)

    if all_valid:

        print(
            "STEP 48 COMPLETE - "
            "BOUNDARY FILES ARE VALID"
        )

    else:

        print(
            "STEP 48 COMPLETE - "
            "SOME BOUNDARIES NEED ATTENTION"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()