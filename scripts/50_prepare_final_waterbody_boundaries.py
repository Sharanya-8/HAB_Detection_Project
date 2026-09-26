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
    "labels",
    "hab_masks"
)


OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "training_waterbodies"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# REFERENCE CENTERS
# ============================================================

CENTERS = {
    "Hussain_Sagar": (17.42222, 78.47389),
    "Saroor_Nagar": (17.3545, 78.5575),
    "Osman_Sagar": (17.3861, 78.2977),
    "Himayat_Sagar": (17.3347, 78.3499),
    "Shamirpet_Lake": (17.5930, 78.5660)
}


# ============================================================
# EXISTING BOUNDARIES
# ============================================================

EXISTING = {
    "Hussain_Sagar": os.path.join(
        BASE_DIR,
        "Hussain_Sagar_boundary.geojson"
    ),

    "Osman_Sagar": os.path.join(
        BASE_DIR,
        "waterbodies",
        "Osman_Sagar_jrc_boundary.geojson"
    ),

    "Himayat_Sagar": os.path.join(
        BASE_DIR,
        "waterbodies",
        "Himayat_Sagar_jrc_boundary.geojson"
    ),

    "Shamirpet_Lake": os.path.join(
        BASE_DIR,
        "waterbodies",
        "Shamirpet_Lake_nominatim_candidates.json"
    )
}


# ============================================================
# SAROOR NAGAR
# ============================================================

# We don't know its exact file location from the current
# search, so search automatically.

def find_file(name):

    for root, dirs, files in os.walk(BASE_DIR):

        for file in files:

            if file.lower() == name.lower():

                return os.path.join(
                    root,
                    file
                )

    return None


# ============================================================
# REPAIR GEOMETRY
# ============================================================

def repair_geometry(gdf):

    gdf = gdf.copy()

    # Remove empty geometries
    gdf = gdf[
        ~gdf.geometry.is_empty
    ].copy()

    # Repair invalid geometries
    gdf["geometry"] = (
        gdf.geometry
        .buffer(0)
    )

    # Remove empty results
    gdf = gdf[
        ~gdf.geometry.is_empty
    ].copy()

    return gdf


# ============================================================
# SELECT POLYGON CONTAINING CENTER
# ============================================================

def select_center_polygon(
    gdf,
    lat,
    lon
):

    center = Point(
        lon,
        lat
    )

    gdf = gdf.to_crs(
        "EPSG:4326"
    )

    # First try polygons containing center
    containing = gdf[
        gdf.geometry.contains(
            center
        )
    ]

    if len(containing) > 0:

        return containing.copy()

    return None


# ============================================================
# SAVE
# ============================================================

def save_boundary(
    gdf,
    name
):

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{name}_boundary.geojson"
    )

    gdf = gdf[
        ["geometry"]
    ]

    gdf.to_file(
        output_file,
        driver="GeoJSON"
    )

    print(
        f"Saved: {output_file}"
    )

    return output_file


# ============================================================
# PROCESS NORMAL GEOJSON
# ============================================================

def process_geojson(
    name,
    source_file
):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Source: {source_file}"
    )

    gdf = gpd.read_file(
        source_file
    )

    print(
        f"Source polygons: {len(gdf)}"
    )

    gdf = repair_geometry(
        gdf
    )

    print(
        f"After geometry repair: "
        f"{len(gdf)}"
    )

    lat, lon = CENTERS[name]

    selected = select_center_polygon(
        gdf,
        lat,
        lon
    )

    if selected is None:

        print(
            "ERROR: No polygon contains "
            "the reference center."
        )

        return False

    print(
        "Reference center is inside "
        "the selected polygon."
    )

    save_boundary(
        selected,
        name
    )

    return True


# ============================================================
# PROCESS SHAMIRPET JSON
# ============================================================

def process_shamirpet(
    source_file
):

    name = "Shamirpet_Lake"

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    with open(
        source_file,
        "r",
        encoding="utf-8"
    ) as f:

        candidates = json.load(f)

    print(
        f"Polygon candidates: "
        f"{len(candidates)}"
    )

    if len(candidates) == 0:

        print(
            "ERROR: No polygon candidate."
        )

        return False

    # Use first valid polygon candidate
    geometry = candidates[0]["geometry"]

    gdf = gpd.GeoDataFrame(
        {
            "name": [name]
        },
        geometry=[
            gpd.GeoSeries.from_geojson
            if False else None
        ],
        crs="EPSG:4326"
    )

    # Create GeoDataFrame correctly
    from shapely.geometry import shape

    gdf = gpd.GeoDataFrame(
        {
            "name": [name]
        },
        geometry=[
            shape(geometry)
        ],
        crs="EPSG:4326"
    )

    gdf = repair_geometry(
        gdf
    )

    lat, lon = CENTERS[name]

    selected = select_center_polygon(
        gdf,
        lat,
        lon
    )

    if selected is None:

        print(
            "WARNING: Center not inside "
            "polygon."
        )

        print(
            "Using available polygon."
        )

        selected = gdf

    save_boundary(
        selected,
        name
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "STEP 50 - PREPARE FINAL "
        "TRAINING WATERBODY BOUNDARIES"
    )
    print("=" * 70)

    success = []

    # --------------------------------------------------------
    # HUSSAIN SAGAR
    # --------------------------------------------------------

    source = EXISTING[
        "Hussain_Sagar"
    ]

    if os.path.exists(source):

        success.append(
            process_geojson(
                "Hussain_Sagar",
                source
            )
        )

    else:

        print(
            "Hussain Sagar source not found."
        )

        success.append(False)

    # --------------------------------------------------------
    # SAROOR NAGAR
    # --------------------------------------------------------

    saroor_source = find_file(
        "Saroor_Nagar_boundary.geojson"
    )

    if saroor_source:

        success.append(
            process_geojson(
                "Saroor_Nagar",
                saroor_source
            )
        )

    else:

        print(
            "Saroor Nagar boundary not found."
        )

        success.append(False)

    # --------------------------------------------------------
    # OSMAN SAGAR
    # --------------------------------------------------------

    success.append(
        process_geojson(
            "Osman_Sagar",
            EXISTING["Osman_Sagar"]
        )
    )

    # --------------------------------------------------------
    # HIMAYAT SAGAR
    # --------------------------------------------------------

    success.append(
        process_geojson(
            "Himayat_Sagar",
            EXISTING["Himayat_Sagar"]
        )
    )

    # --------------------------------------------------------
    # SHAMIRPET
    # --------------------------------------------------------

    success.append(
        process_shamirpet(
            EXISTING["Shamirpet_Lake"]
        )
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 50 SUMMARY")
    print("=" * 70)

    files = sorted(
        os.listdir(
            OUTPUT_DIR
        )
    )

    print()

    for file in files:

        if file.endswith(
            ".geojson"
        ):

            print(
                f"[OK] {file}"
            )

    print()
    print("=" * 70)
    print("STEP 50 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()