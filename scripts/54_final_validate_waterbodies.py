import os
import geopandas as gpd
from shapely.geometry import Point


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

BOUNDARY_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "labels",
    "hab_masks",
    "training_waterbodies"
)


WATERBODIES = {
    "Hussain_Sagar": (17.42222, 78.47389),
    "Saroor_Nagar": (17.3539, 78.5270),
    "Osman_Sagar": (17.3861, 78.2977),
    "Himayat_Sagar": (17.28233, 78.36119),
    "Shamirpet_Lake": (17.60967, 78.55663),
}


def main():

    print("=" * 70)
    print("STEP 54 - FINAL WATERBODY BOUNDARY VALIDATION")
    print("=" * 70)

    for name, (lat, lon) in WATERBODIES.items():

        print()
        print("=" * 70)
        print(name)
        print("=" * 70)

        path = os.path.join(
            BOUNDARY_DIR,
            f"{name}_boundary.geojson"
        )

        if not os.path.exists(path):

            print("ERROR: Boundary not found.")
            continue

        gdf = gpd.read_file(path)

        print(f"Geometry count: {len(gdf)}")
        print(f"CRS: {gdf.crs}")
        print(
            f"Geometry type: "
            f"{list(gdf.geometry.geom_type.unique())}"
        )

        # Repair geometry for validation
        gdf["geometry"] = gdf.geometry.buffer(0)

        valid = gdf.geometry.is_valid.all()

        print(f"Geometry valid: {valid}")

        # Project to UTM for area/distance
        projected = gdf.to_crs("EPSG:32644")

        area_ha = (
            projected.geometry.area.sum()
            / 10000
        )

        print(
            f"Area: {area_ha:.2f} hectares"
        )

        # Reference point
        point = gpd.GeoSeries(
            [
                Point(lon, lat)
            ],
            crs="EPSG:4326"
        )

        point_projected = point.to_crs(
            "EPSG:32644"
        ).iloc[0]

        # Distance
        distance = (
            projected.geometry
            .distance(point_projected)
            .min()
        )

        print(
            f"Distance from reference point: "
            f"{distance:.1f} meters"
        )

        # Contains
        contains = (
            gdf.geometry
            .contains(
                point.iloc[0]
            )
            .any()
        )

        print(
            f"Reference point inside: "
            f"{contains}"
        )

        # Bounds
        bounds = gdf.to_crs(
            "EPSG:4326"
        ).total_bounds

        print(
            f"Longitude: "
            f"{bounds[0]:.6f} → {bounds[2]:.6f}"
        )

        print(
            f"Latitude : "
            f"{bounds[1]:.6f} → {bounds[3]:.6f}"
        )

        print("Validation complete.")

    print()
    print("=" * 70)
    print("STEP 54 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()