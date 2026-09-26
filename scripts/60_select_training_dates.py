import ee
import geopandas as gpd
import pandas as pd
from pathlib import Path
from datetime import datetime


# ============================================================
# STEP 60 - SELECT BALANCED TRAINING DATES
# ============================================================

print("=" * 70)
print("STEP 60 - SELECT BALANCED MULTI-WATERBODY TRAINING DATES")
print("=" * 70)


# ------------------------------------------------------------
# Initialize Google Earth Engine
# ------------------------------------------------------------

PROJECT_ID = "remote-sensing-project-507517"

ee.Initialize(project=PROJECT_ID)

print("Google Earth Engine initialized successfully.")


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BOUNDARY_DIR = Path(
    "data/labels/hab_masks/training_waterbodies"
)

OUTPUT_DIR = Path(
    "data/processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


OUTPUT_FILE = (
    OUTPUT_DIR / "selected_training_dates.csv"
)


# ------------------------------------------------------------
# Training waterbodies
# ------------------------------------------------------------

waterbodies = {
    "Hussain Sagar": "Hussain_Sagar_boundary.geojson",
    "Saroor Nagar": "Saroor_Nagar_OSM_boundary.geojson",
    "Osman Sagar": "Osman_Sagar_boundary.geojson",
    "Himayat Sagar": "Himayat_Sagar_relation_boundary.geojson",
}


# ------------------------------------------------------------
# Sentinel-2 collections
# ------------------------------------------------------------

S2 = ee.ImageCollection(
    "COPERNICUS/S2_SR_HARMONIZED"
)

CLOUD_PROBABILITY = ee.ImageCollection(
    "COPERNICUS/S2_CLOUD_PROBABILITY"
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

START_YEAR = 2016
END_YEAR = 2025

CLOUD_THRESHOLD = 40

DATES_PER_YEAR = 10


# ------------------------------------------------------------
# Convert GeoJSON to Earth Engine geometry
# ------------------------------------------------------------

def geojson_to_ee_geometry(filepath):

    gdf = gpd.read_file(filepath)

    if gdf.empty:
        raise ValueError(
            f"Empty boundary file: {filepath}"
        )

    geometry = gdf.geometry.unary_union

    return ee.Geometry(
        geometry.__geo_interface__
    )


# ------------------------------------------------------------
# Get usable dates for one waterbody/year
# ------------------------------------------------------------

def get_usable_dates(
    geometry,
    year
):

    start_date = f"{year}-01-01"
    end_date = f"{year + 1}-01-01"

    s2_collection = (
        S2
        .filterBounds(geometry)
        .filterDate(
            start_date,
            end_date
        )
    )

    cloud_collection = (
        CLOUD_PROBABILITY
        .filterBounds(geometry)
        .filterDate(
            start_date,
            end_date
        )
    )

    # --------------------------------------------------------
    # Join Sentinel-2 with cloud probability
    # --------------------------------------------------------

    joined = ee.Join.saveFirst(
        "cloud_probability"
    ).apply(
        primary=s2_collection,
        secondary=cloud_collection,
        condition=ee.Filter.equals(
            leftField="system:index",
            rightField="system:index"
        )
    )

    joined = ee.ImageCollection(joined)

    # --------------------------------------------------------
    # Calculate cloud percentage over waterbody
    # --------------------------------------------------------

    def add_cloud_percentage(image):

        cloud_image = ee.Image(
            image.get("cloud_probability")
        )

        cloud_probability = (
            cloud_image
            .select("probability")
        )

        cloudy = (
            cloud_probability
            .gte(CLOUD_THRESHOLD)
        )

        cloud_percentage = (
            cloudy
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=20,
                maxPixels=1e8
            )
            .get("probability")
        )

        return image.set(
            "cloud_percentage",
            cloud_percentage
        )

    joined = joined.map(
        add_cloud_percentage
    )

    # --------------------------------------------------------
    # Keep images with <40% cloud coverage
    # --------------------------------------------------------

    usable = (
        joined
        .filter(
            ee.Filter.lt(
                "cloud_percentage",
                0.40
            )
        )
    )

    timestamps = (
        usable
        .aggregate_array(
            "system:time_start"
        )
        .getInfo()
    )

    dates = []

    for timestamp in timestamps:

        date = datetime.fromtimestamp(
            timestamp / 1000
        ).strftime("%Y-%m-%d")

        dates.append(date)

    return sorted(set(dates))


# ------------------------------------------------------------
# Select approximately evenly distributed dates
# ------------------------------------------------------------

def select_dates_evenly(
    dates,
    number
):

    if not dates:
        return []

    if len(dates) <= number:
        return dates

    # Convert to datetime
    date_objects = [
        datetime.strptime(
            d,
            "%Y-%m-%d"
        )
        for d in dates
    ]

    # Target positions distributed through the year
    selected = []

    for i in range(number):

        position = (
            i * (len(date_objects) - 1)
            / (number - 1)
        )

        index = round(position)

        selected.append(
            dates[index]
        )

    # Remove accidental duplicates
    return sorted(set(selected))


# ============================================================
# MAIN PROCESS
# ============================================================

all_rows = []


for waterbody, filename in waterbodies.items():

    print()
    print("=" * 70)
    print(waterbody)
    print("=" * 70)

    filepath = (
        BOUNDARY_DIR / filename
    )

    geometry = geojson_to_ee_geometry(
        filepath
    )

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        print(
            f"\n{year}: checking usable dates..."
        )

        usable_dates = get_usable_dates(
            geometry,
            year
        )

        selected_dates = select_dates_evenly(
            usable_dates,
            DATES_PER_YEAR
        )

        print(
            f"Usable dates: {len(usable_dates)}"
        )

        print(
            f"Selected dates: {len(selected_dates)}"
        )

        print(
            selected_dates
        )

        for date in selected_dates:

            all_rows.append({
                "waterbody": waterbody,
                "year": year,
                "date": date
            })


# ============================================================
# SAVE CSV
# ============================================================

df = pd.DataFrame(
    all_rows
)

df = df.sort_values(
    [
        "waterbody",
        "year",
        "date"
    ]
).reset_index(drop=True)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL DATASET SELECTION SUMMARY")
print("=" * 70)

print()

print(
    df.groupby("waterbody")
    .size()
    .to_string()
)

print()

print(
    "Total selected images:",
    len(df)
)

print()

print(
    "Expected maximum:",
    len(waterbodies)
    * (END_YEAR - START_YEAR + 1)
    * DATES_PER_YEAR
)

print()

print(
    f"Saved: {OUTPUT_FILE}"
)

print()
print("=" * 70)
print("STEP 60 COMPLETE")
print("=" * 70)