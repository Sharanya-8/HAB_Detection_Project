from pathlib import Path
import sys
import numpy as np
import pandas as pd
import rasterio
import importlib.util

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add project root so utils can be imported
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# GEE UTILITIES
# ============================================================

from utils.gee_utils import (
    initialize_gee,
    get_masked_sentinel2_collection,
    get_available_dates,
    get_image_for_date,
    create_14_channel_image,
    download_14_channel_image,
)

import ee


# ============================================================
# LOAD EXISTING WATER-MASK FUNCTION
# ============================================================

MASK_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "94_create_aligned_water_mask.py"
)

spec = importlib.util.spec_from_file_location(
    "aligned_water_mask",
    MASK_SCRIPT
)

mask_module = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    mask_module
)

create_aligned_water_mask = (
    mask_module.create_aligned_water_mask
)


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dynamic_temporal"
)

IMAGE_ROOT = (
    OUTPUT_ROOT
    / "images"
)

FEATURE_OUTPUT = (
    OUTPUT_ROOT
    / "dynamic_temporal_features.csv"
)

SKIPPED_OUTPUT = (
    OUTPUT_ROOT
    / "dynamic_temporal_skipped.csv"
)

DEFAULT_START_DATE = "2018-01-01"

DEFAULT_END_DATE = "2026-09-25"

CLOUD_THRESHOLD = 40

# Approximately one observation every 30 days.
TARGET_INTERVAL_DAYS = 30


# ============================================================
# FEATURE CALCULATION
# ============================================================

def calculate_features(
    image_path,
    water_mask_path
):
    """
    Calculate temporal spectral features only from
    pixels inside the selected waterbody.

    Parameters
    ----------
    image_path:
        14-band Sentinel-2 GeoTIFF.

    water_mask_path:
        Raster water mask aligned with the Sentinel-2 image.

    Channels
    --------
    0  B2
    1  B3
    2  B4
    3  B5
    4  B6
    5  B7
    6  B8
    7  B8A
    8  B11
    9  B12
    10 NDWI
    11 MNDWI
    12 NDCI
    13 FAI
    """

    # --------------------------------------------------------
    # READ SENTINEL-2 IMAGE
    # --------------------------------------------------------

    with rasterio.open(
        image_path
    ) as src:

        image = src.read().astype(
            np.float32
        )

        image_shape = (
            src.height,
            src.width
        )

    # --------------------------------------------------------
    # CHECK BAND COUNT
    # --------------------------------------------------------

    if image.shape[0] != 14:

        raise ValueError(
            f"Expected 14 bands, "
            f"found {image.shape[0]}"
        )

    # --------------------------------------------------------
    # READ WATER MASK
    # --------------------------------------------------------

    with rasterio.open(
        water_mask_path
    ) as mask_src:

        water_mask = mask_src.read(1)

        mask_shape = (
            mask_src.height,
            mask_src.width
        )

    # --------------------------------------------------------
    # CHECK IMAGE / MASK ALIGNMENT
    # --------------------------------------------------------

    if mask_shape != image_shape:

        raise ValueError(
            "Image and water mask shape mismatch: "
            f"image={image_shape}, "
            f"mask={mask_shape}"
        )

    # --------------------------------------------------------
    # WATERBODY PIXELS ONLY
    # --------------------------------------------------------

    valid = (
        water_mask == 1
    )

    # --------------------------------------------------------
    # REMOVE INVALID PIXELS
    # --------------------------------------------------------

    for band_index in range(14):

        valid &= np.isfinite(
            image[band_index]
        )

    # Remove completely zero pixels.
    valid &= (
        np.abs(image[:10]).sum(axis=0) > 0
    )

    if not valid.any():

        raise ValueError(
            "No valid waterbody pixels found."
        )

    # --------------------------------------------------------
    # EXTRACT INDICES
    # --------------------------------------------------------

    ndwi = image[10][valid]

    mndwi = image[11][valid]

    ndci = image[12][valid]

    fai = image[13][valid]

    # --------------------------------------------------------
    # CALCULATE FEATURES
    # --------------------------------------------------------

    return {

        "valid_water_pixels":
            int(valid.sum()),

        "ndci_mean":
            float(np.mean(ndci)),

        "ndci_median":
            float(np.median(ndci)),

        "ndci_max":
            float(np.max(ndci)),

        "ndwi_mean":
            float(np.mean(ndwi)),

        "ndwi_median":
            float(np.median(ndwi)),

        "mndwi_mean":
            float(np.mean(mndwi)),

        "mndwi_median":
            float(np.median(mndwi)),

        "fai_mean":
            float(np.mean(fai)),

        "fai_median":
            float(np.median(fai)),

        "fai_max":
            float(np.max(fai)),
    }


# ============================================================
# SELECT APPROXIMATELY MONTHLY DATES
# ============================================================

def select_temporal_dates(
    available_dates,
    target_interval_days=TARGET_INTERVAL_DAYS
):
    """
    Select observations approximately every 30 days.

    Sentinel-2 observations are irregular, so the function
    selects the first available observation on or after
    the desired interval.
    """

    if not available_dates:

        return []

    dates = sorted(
        pd.to_datetime(
            available_dates
        )
    )

    selected = [
        dates[0]
    ]

    last_selected = dates[0]

    for date in dates[1:]:

        difference = (
            date - last_selected
        ).days

        if difference >= target_interval_days:

            selected.append(
                date
            )

            last_selected = date

    return selected


# ============================================================
# PROCESS WATERBODY
# ============================================================

def process_waterbody(
    waterbody_name,
    geometry,
    boundary_path,
    start_date=DEFAULT_START_DATE,
    end_date=DEFAULT_END_DATE,
):
    """
    Build a dynamic temporal dataset for one waterbody.

    Parameters
    ----------
    waterbody_name:
        Name used for the output folder.

    geometry:
        GeoJSON-style geometry dictionary.

    boundary_path:
        Waterbody boundary GeoJSON used to create the
        aligned water mask.

    start_date:
        Historical start date.

    end_date:
        Historical end date.
    """

    print()

    print(
        "=" * 70
    )

    print(
        f"DYNAMIC TEMPORAL DATASET: "
        f"{waterbody_name}"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    output_dir = (
        IMAGE_ROOT
        / waterbody_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # CONVERT GEOMETRY
    # --------------------------------------------------------

    aoi = ee.Geometry(
        geometry
    )

    # --------------------------------------------------------
    # GET CLOUD-MASKED SENTINEL-2 COLLECTION
    # --------------------------------------------------------

    print(
        "Getting Sentinel-2 collection..."
    )

    collection = (
        get_masked_sentinel2_collection(
            aoi,
            start_date,
            end_date,
            CLOUD_THRESHOLD
        )
    )

    # --------------------------------------------------------
    # GET AVAILABLE DATES
    # --------------------------------------------------------

    print(
        "Finding available dates..."
    )

    available_dates = (
        get_available_dates(
            collection
        )
    )

    print(
        f"Available observations: "
        f"{len(available_dates)}"
    )

    if not available_dates:

        raise RuntimeError(
            "No usable Sentinel-2 "
            "observations found."
        )

    # --------------------------------------------------------
    # SELECT APPROXIMATELY MONTHLY DATES
    # --------------------------------------------------------

    selected_dates = (
        select_temporal_dates(
            available_dates
        )
    )

    print(
        f"Selected observations: "
        f"{len(selected_dates)}"
    )

    rows = []

    skipped = []

    # --------------------------------------------------------
    # PROCESS EACH DATE
    # --------------------------------------------------------

    for index, date in enumerate(
        selected_dates,
        start=1
    ):

        date_string = (
            pd.Timestamp(date)
            .strftime(
                "%Y-%m-%d"
            )
        )

        # ----------------------------------------------------
        # IMAGE PATH
        # ----------------------------------------------------

        output_path = (
            output_dir
            / f"{date_string}_14Channel.tif"
        )

        # ----------------------------------------------------
        # WATER MASK PATH
        # ----------------------------------------------------

        water_mask_path = (
            output_dir
            / f"{date_string}_water_mask.tif"
        )

        print()

        print(
            f"[{index}/{len(selected_dates)}] "
            f"{date_string}"
        )

        try:

            # ------------------------------------------------
            # GET / REUSE SENTINEL-2 IMAGE
            # ------------------------------------------------

            if output_path.exists():

                print(
                    "  Using existing image."
                )

            else:

                print(
                    "  Downloading image..."
                )

                image = (
                    get_image_for_date(
                        collection,
                        date_string
                    )
                )

                image_14 = (
                    create_14_channel_image(
                        image
                    )
                )

                download_14_channel_image(
                    image_14,
                    aoi,
                    output_path,
                    scale=10
                )

            # ------------------------------------------------
            # CREATE / REUSE ALIGNED WATER MASK
            # ------------------------------------------------

            if water_mask_path.exists():

                print(
                    "  Using existing water mask."
                )

            else:

                print(
                    "  Creating water mask..."
                )

                create_aligned_water_mask(
                    boundary_path=boundary_path,
                    image_path=output_path,
                    output_mask_path=water_mask_path
                )

            # ------------------------------------------------
            # CALCULATE TEMPORAL FEATURES
            # ------------------------------------------------

            features = (
                calculate_features(
                    output_path,
                    water_mask_path
                )
            )

            # ------------------------------------------------
            # STORE RECORD
            # ------------------------------------------------

            rows.append({

                "waterbody":
                    waterbody_name,

                "date":
                    pd.Timestamp(date),

                **features,

                "image_path":
                    str(
                        output_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),

                "water_mask_path":
                    str(
                        water_mask_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
            })

        except Exception as e:

            skipped.append({

                "waterbody":
                    waterbody_name,

                "date":
                    date_string,

                "reason":
                    str(e),

            })

            print(
                f"  SKIPPED: {e}"
            )

    return rows, skipped


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(
    rows,
    skipped
):

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    if not rows:

        raise RuntimeError(
            "No temporal records were created."
        )

    df = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "waterbody",
                "date"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # NEXT OBSERVATION TARGET
    # --------------------------------------------------------

    df["target_next_ndci"] = (
        df.groupby(
            "waterbody"
        )[
            "ndci_mean"
        ].shift(-1)
    )

    df["next_date"] = (
        df.groupby(
            "waterbody"
        )[
            "date"
        ].shift(-1)
    )

    df[
        "days_to_next_observation"
    ] = (
        df["next_date"]
        - df["date"]
    ).dt.days

    # --------------------------------------------------------
    # SAVE DATASET
    # --------------------------------------------------------

    df.to_csv(
        FEATURE_OUTPUT,
        index=False
    )

    pd.DataFrame(
        skipped
    ).to_csv(
        SKIPPED_OUTPUT,
        index=False
    )

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "DYNAMIC TEMPORAL DATASET COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Records created: "
        f"{len(df)}"
    )

    print(
        f"Skipped observations: "
        f"{len(skipped)}"
    )

    print()

    print(
        "Records by waterbody:"
    )

    print(
        df.groupby(
            "waterbody"
        ).size().to_string()
    )

    print()

    print(
        "Date ranges:"
    )

    print(
        df.groupby(
            "waterbody"
        )["date"]
        .agg(
            [
                "min",
                "max",
                "count"
            ]
        )
        .to_string()
    )

    print()

    print(
        "Saved:"
    )

    print(
        FEATURE_OUTPUT
    )

    print(
        SKIPPED_OUTPUT
    )

    print(
        "=" * 70
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    initialize_gee()

    print()

    print("=" * 70)

    print(
        "STEP 118 - "
        "DYNAMIC TEMPORAL DATASET"
    )

    print("=" * 70)

    print()

    # --------------------------------------------------------
    # HUSSAIN SAGAR TEST WATERBODY
    # --------------------------------------------------------

    waterbody_name = "Hussain_Sagar"

    boundary_path = (
        PROJECT_ROOT
        / "data"
        / "labels"
        / "hab_masks"
        / "training_waterbodies"
        / "Hussain_Sagar_boundary.geojson"
    )

    start_date = "2018-01-01"

    end_date = "2026-09-25"

    # --------------------------------------------------------
    # LOAD WATERBODY GEOMETRY
    # --------------------------------------------------------

    print(
        f"Waterbody: {waterbody_name}"
    )

    print(
        f"Boundary: {boundary_path}"
    )

    print(
        f"Date range: "
        f"{start_date} to {end_date}"
    )

    print()

    if not boundary_path.exists():

        raise FileNotFoundError(
            f"Boundary file not found: "
            f"{boundary_path}"
        )

    # --------------------------------------------------------
    # READ GEOJSON
    # --------------------------------------------------------

    import json

    with open(
        boundary_path,
        "r",
        encoding="utf-8"
    ) as f:

        boundary_data = json.load(f)

    # --------------------------------------------------------
    # GET GEOMETRY
    # --------------------------------------------------------

    if (
        "features" not in boundary_data
        or not boundary_data["features"]
    ):

        raise ValueError(
            "No features found in "
            "the boundary GeoJSON."
        )

    geometry = (
        boundary_data["features"][0]["geometry"]
    )

    # --------------------------------------------------------
    # PROCESS WATERBODY
    # --------------------------------------------------------

    rows, skipped = process_waterbody(
        waterbody_name=waterbody_name,
        geometry=geometry,
        boundary_path=boundary_path,
        start_date=start_date,
        end_date=end_date,
    )

    # --------------------------------------------------------
    # SAVE DATASET
    # --------------------------------------------------------

    save_dataset(
        rows,
        skipped
    )

    print()

    print(
        "Step 118 finished successfully."
    )