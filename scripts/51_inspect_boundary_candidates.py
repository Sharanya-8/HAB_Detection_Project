import os
import json
import geopandas as gpd
from shapely.geometry import Point


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

BASE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "labels"
)

HAB_DIR = os.path.join(
    BASE_DIR,
    "hab_masks"
)


# ============================================================
# REFERENCE CENTERS
# ============================================================

CENTERS = {
    "Saroor_Nagar": (17.3545, 78.5575),
    "Himayat_Sagar": (17.3347, 78.3499),
    "Shamirpet_Lake": (17.5930, 78.5660),
}


# ============================================================
# SEARCH ALL PROJECT FILES
# ============================================================

def find_all_files():

    results = []

    for root, dirs, files in os.walk(PROJECT_ROOT):

        # Ignore virtual environment
        dirs[:] = [
            d for d in dirs
            if d != "venv"
        ]

        for file in files:

            if file.lower().endswith(
                (".geojson", ".json", ".shp")
            ):

                results.append(
                    os.path.join(
                        root,
                        file
                    )
                )

    return results


# ============================================================
# SAROOR NAGAR SEARCH
# ============================================================

def inspect_saroor(files):

    print()
    print("=" * 70)
    print("1. SEARCHING FOR SAROOR NAGAR")
    print("=" * 70)

    matches = []

    for path in files:

        filename = os.path.basename(
            path
        ).lower()

        if (
            "saroor" in filename
            or "saroornagar" in filename
        ):

            matches.append(path)

    if not matches:

        print(
            "No Saroor Nagar boundary file "
            "found by filename."
        )

        return

    for path in matches:

        print()
        print(
            "Candidate:"
        )

        print(
            os.path.relpath(
                path,
                PROJECT_ROOT
            )
        )

        try:

            gdf = gpd.read_file(
                path
            )

            print(
                f"  Geometry count: {len(gdf)}"
            )

            print(
                f"  Geometry types: "
                f"{list(gdf.geometry.geom_type.unique())}"
            )

            print(
                f"  CRS: {gdf.crs}"
            )

        except Exception as e:

            print(
                f"  Could not read: {e}"
            )


# ============================================================
# HIMAYAT SAGAR INSPECTION
# ============================================================

def inspect_himayat():

    print()
    print("=" * 70)
    print("2. INSPECTING HIMAYAT SAGAR JRC POLYGONS")
    print("=" * 70)

    path = os.path.join(
        HAB_DIR,
        "waterbodies",
        "Himayat_Sagar_jrc_boundary.geojson"
    )

    if not os.path.exists(path):

        print(
            "Himayat JRC file not found."
        )

        return

    gdf = gpd.read_file(
        path
    )

    print(
        f"Total polygons: {len(gdf)}"
    )

    # Repair only for inspection
    gdf = gdf.copy()

    gdf["geometry"] = (
        gdf.geometry.buffer(0)
    )

    gdf = gdf[
        ~gdf.geometry.is_empty
    ].copy()

    # Project to UTM for correct distance/area
    projected = gdf.to_crs(
        "EPSG:32644"
    )

    center = gpd.GeoSeries(
        [
            Point(
                CENTERS["Himayat_Sagar"][1],
                CENTERS["Himayat_Sagar"][0]
            )
        ],
        crs="EPSG:4326"
    ).to_crs(
        "EPSG:32644"
    ).iloc[0]

    projected["area_ha"] = (
        projected.geometry.area / 10000
    )

    projected["distance_m"] = (
        projected.geometry.distance(
            center
        )
    )

    # Sort by distance
    projected = projected.sort_values(
        "distance_m"
    )

    print()
    print(
        "Closest polygons to the reference point:"
    )

    for index, row in projected.head(10).iterrows():

        print(
            f"Polygon {index}: "
            f"distance={row['distance_m']:.1f} m, "
            f"area={row['area_ha']:.2f} ha"
        )


# ============================================================
# SHAMIRPET INSPECTION
# ============================================================

def inspect_shamirpet():

    print()
    print("=" * 70)
    print("3. INSPECTING SHAMIRPET POLYGON")
    print("=" * 70)

    path = os.path.join(
        HAB_DIR,
        "waterbodies",
        "Shamirpet_Lake_nominatim_candidates.json"
    )

    if not os.path.exists(path):

        print(
            "Shamirpet candidate file not found."
        )

        return

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        candidates = json.load(f)

    print(
        f"Candidates: {len(candidates)}"
    )

    if not candidates:
        return

    from shapely.geometry import shape

    for i, candidate in enumerate(
        candidates,
        1
    ):

        geometry = shape(
            candidate["geometry"]
        )

        area = gpd.GeoSeries(
            [geometry],
            crs="EPSG:4326"
        ).to_crs(
            "EPSG:32644"
        ).area.iloc[0] / 10000

        center = Point(
            CENTERS["Shamirpet_Lake"][1],
            CENTERS["Shamirpet_Lake"][0]
        )

        contains = geometry.contains(
            center
        )

        distance = gpd.GeoSeries(
            [geometry],
            crs="EPSG:4326"
        ).to_crs(
            "EPSG:32644"
        ).geometry.distance(
            gpd.GeoSeries(
                [center],
                crs="EPSG:4326"
            ).to_crs(
                "EPSG:32644"
            ).iloc[0]
        ).iloc[0]

        print()
        print(
            f"Candidate {i}:"
        )

        print(
            f"  Name: "
            f"{candidate.get('display_name')}"
        )

        print(
            f"  OSM ID: "
            f"{candidate.get('osm_id')}"
        )

        print(
            f"  Geometry: "
            f"{geometry.geom_type}"
        )

        print(
            f"  Area: "
            f"{area:.2f} ha"
        )

        print(
            f"  Center inside: "
            f"{contains}"
        )

        print(
            f"  Distance to center: "
            f"{distance:.1f} m"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 51 - INSPECT BOUNDARY CANDIDATES")
    print("=" * 70)

    files = find_all_files()

    print()
    print(
        f"Project vector files found: "
        f"{len(files)}"
    )

    inspect_saroor(
        files
    )

    inspect_himayat()

    inspect_shamirpet()

    print()
    print("=" * 70)
    print("STEP 51 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()