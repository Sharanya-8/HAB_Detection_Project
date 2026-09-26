import os
import json
import geopandas as gpd


print("=" * 70)
print("STEP 113 — PREPARE SECOND WATERBODY")
print("=" * 70)


# ------------------------------------------------------------
# EXISTING VALIDATED OSMAN SAGAR BOUNDARY
# ------------------------------------------------------------

SOURCE_PATH = (
    "data/labels/hab_masks/training_waterbodies/"
    "Osman_Sagar_boundary.geojson"
)

OUTPUT_DIR = (
    "results/waterbody_discovery/"
    "second_waterbody"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "selected_waterbody_geometry.geojson"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# LOAD BOUNDARY
# ------------------------------------------------------------

print("\nLoading validated Osman Sagar boundary...")

gdf = gpd.read_file(SOURCE_PATH)

print(f"Geometry type : {gdf.geometry.iloc[0].geom_type}")
print(f"CRS           : {gdf.crs}")
print(f"Valid         : {gdf.geometry.iloc[0].is_valid}")


# ------------------------------------------------------------
# SAVE COPY FOR GENERIC PIPELINE
# ------------------------------------------------------------

gdf.to_file(
    OUTPUT_PATH,
    driver="GeoJSON"
)


# ------------------------------------------------------------
# CALCULATE AREA
# ------------------------------------------------------------

area_gdf = gdf.to_crs(epsg=32643)

area_m2 = area_gdf.geometry.area.sum()
area_ha = area_m2 / 10000


# ------------------------------------------------------------
# CREATE WATERBODY INFORMATION
# ------------------------------------------------------------

result = {
    "name": "Osman Sagar",
    "geometry_file": OUTPUT_PATH,
    "geometry_type": gdf.geometry.iloc[0].geom_type,
    "crs": str(gdf.crs),
    "valid": bool(gdf.geometry.iloc[0].is_valid),
    "area_ha": float(area_ha)
}

JSON_PATH = os.path.join(
    OUTPUT_DIR,
    "selected_waterbody.json"
)

with open(
    JSON_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        result,
        f,
        indent=4
    )


# ------------------------------------------------------------
# RESULT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 113 COMPLETE")
print("=" * 70)

print(f"\nWaterbody : Osman Sagar")
print(f"Geometry  : {gdf.geometry.iloc[0].geom_type}")
print(f"Valid     : {gdf.geometry.iloc[0].is_valid}")
print(f"Area      : {area_ha:.2f} ha")

print("\nGeometry saved to:")
print(os.path.abspath(OUTPUT_PATH))

print("\nInformation saved to:")
print(os.path.abspath(JSON_PATH))

print("=" * 70)