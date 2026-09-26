import geopandas as gpd
from pathlib import Path


print("=" * 70)
print("STEP 58 - FINAL MULTI-WATERBODY BOUNDARY VALIDATION")
print("=" * 70)


folder = Path(
    "data/labels/hab_masks/training_waterbodies"
)


waterbodies = {
    "Hussain Sagar": "Hussain_Sagar_boundary.geojson",
    "Saroor Nagar": "Saroor_Nagar_OSM_boundary.geojson",
    "Osman Sagar": "Osman_Sagar_boundary.geojson",
    "Himayat Sagar": "Himayat_Sagar_relation_boundary.geojson",
    "Shamirpet Lake": "Shamirpet_Lake_OSM_boundary.geojson",
}


for name, filename in waterbodies.items():

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    filepath = folder / filename

    if not filepath.exists():

        print("ERROR: File not found:")
        print(filepath)
        continue

    gdf = gpd.read_file(filepath)

    print(
        "File:",
        filename
    )

    print(
        "Geometry count:",
        len(gdf)
    )

    print(
        "CRS:",
        gdf.crs
    )

    print(
        "Geometry type:",
        gdf.geometry.geom_type.tolist()
    )

    # Repair if necessary
    if not gdf.geometry.is_valid.all():

        print(
            "Geometry invalid - repairing..."
        )

        gdf["geometry"] = (
            gdf.geometry.buffer(0)
        )

    print(
        "Geometry valid:",
        gdf.geometry.is_valid.all()
    )

    # Area in UTM 44N
    area_gdf = gdf.to_crs(
        "EPSG:32644"
    )

    area_ha = (
        area_gdf.geometry.area.sum()
        / 10000
    )

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


print()
print("=" * 70)
print("STEP 58 COMPLETE")
print("=" * 70)