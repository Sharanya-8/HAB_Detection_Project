from pathlib import Path
import sys

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from pyproj import Transformer


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# INPUT FILES
# ============================================================

SENTINEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sentinel2"
    / "Dynamic_Hussain_Sagar_2025-01-02_14Channel.tif"
)

PREDICTION_FILE = (
    PROJECT_ROOT
    / "results"
    / "swin"
    / "dynamic"
    / "dynamic_swin_prediction_512x512.npy"
)

BOUNDARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "hab_masks"
    / "Hussain_Sagar_boundary.geojson"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "maps"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "Dynamic_Hussain_Sagar_HAB_Swin_2025-01-02.tif"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DYNAMIC HAB MAP CREATION")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    for file_path in [
        SENTINEL_FILE,
        PREDICTION_FILE,
        BOUNDARY_FILE
    ]:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{file_path}"
            )

    # --------------------------------------------------------
    # READ ORIGINAL SENTINEL-2 RASTER
    # --------------------------------------------------------

    print()
    print("Reading Sentinel-2 raster...")

    with rasterio.open(
        SENTINEL_FILE
    ) as src:

        width = src.width
        height = src.height
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()

    print(
        f"  Width:  {width}"
    )

    print(
        f"  Height: {height}"
    )

    print(
        f"  CRS:    {crs}"
    )

    # --------------------------------------------------------
    # READ SWIN PREDICTION
    # --------------------------------------------------------

    print()
    print("Reading Swin prediction...")

    prediction = np.load(
        PREDICTION_FILE
    )

    print(
        f"  Prediction shape: "
        f"{prediction.shape}"
    )

    # --------------------------------------------------------
    # CROP PREDICTION BACK TO ORIGINAL RASTER SIZE
    # --------------------------------------------------------

    if (
        prediction.shape[0] < height
        or prediction.shape[1] < width
    ):

        raise ValueError(
            "Prediction is smaller than "
            "the original Sentinel-2 raster."
        )

    prediction_cropped = prediction[
        :height,
        :width
    ]

    print()
    print(
        f"  Cropped prediction: "
        f"{prediction_cropped.shape}"
    )

    # --------------------------------------------------------
    # READ WATERBODY GEOJSON
    # --------------------------------------------------------

    print()
    print("Reading waterbody boundary...")

    import json

    with open(
        BOUNDARY_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        boundary_data = json.load(f)

    geometry = (
        boundary_data[
            "features"
        ][0][
            "geometry"
        ]
    )

    # --------------------------------------------------------
    # CREATE WATERBODY MASK
    # --------------------------------------------------------

    print()
    print("Creating geographic waterbody mask...")

    # The GeoTIFF is currently EPSG:4326.
    # Therefore the GeoJSON geometry is already
    # in the same coordinate system.

    if crs is None:

        raise ValueError(
            "Sentinel-2 raster has no CRS."
        )

    if str(crs) != "EPSG:4326":

        print(
            f"  Reprojecting boundary from EPSG:4326 "
            f"to {crs}..."
        )

        transformer = Transformer.from_crs(
            "EPSG:4326",
            crs,
            always_xy=True
        )

        def transform_coordinates(
            coords
        ):

            if isinstance(
                coords[0],
                (float, int)
            ):

                x, y = transformer.transform(
                    coords[0],
                    coords[1]
                )

                return [
                    x,
                    y
                ]

            return [
                transform_coordinates(
                    item
                )
                for item in coords
            ]

        transformed_geometry = {
            "type": geometry["type"],
            "coordinates":
                transform_coordinates(
                    geometry["coordinates"]
                )
        }

    else:

        transformed_geometry = geometry

    # --------------------------------------------------------
    # WATER MASK
    # --------------------------------------------------------

    water_mask = geometry_mask(
        [
            transformed_geometry
        ],
        out_shape=(
            height,
            width
        ),
        transform=transform,
        invert=True
    )

    water_pixels = int(
        np.sum(
            water_mask
        )
    )

    print(
        f"  Waterbody pixels: "
        f"{water_pixels}"
    )

    if water_pixels == 0:

        raise ValueError(
            "Waterbody mask contains zero pixels. "
            "Check CRS and geometry."
        )

    # --------------------------------------------------------
    # SELECT ONLY WATERBODY PREDICTIONS
    # --------------------------------------------------------

    water_predictions = (
        prediction_cropped[
            water_mask
        ]
    )

    # --------------------------------------------------------
    # HAB / NON-HAB COUNTS
    # --------------------------------------------------------

    hab_pixels = int(
        np.sum(
            water_predictions == 1
        )
    )

    non_hab_pixels = int(
        np.sum(
            water_predictions == 0
        )
    )

    total_water_pixels = (
        hab_pixels
        + non_hab_pixels
    )

    if total_water_pixels == 0:

        raise ValueError(
            "No valid waterbody prediction "
            "pixels were found."
        )

    hab_percentage = (
        hab_pixels
        / total_water_pixels
        * 100.0
    )

    # --------------------------------------------------------
    # CREATE FINAL MAP
    # --------------------------------------------------------

    final_map = np.zeros(
        (
            height,
            width
        ),
        dtype=np.uint8
    )

    # 0 = outside waterbody
    # 1 = Non-HAB
    # 2 = HAB

    final_map[
        water_mask
        & (prediction_cropped == 0)
    ] = 1

    final_map[
        water_mask
        & (prediction_cropped == 1)
    ] = 2

    # --------------------------------------------------------
    # SAVE GEOTIFF
    # --------------------------------------------------------

    output_profile = profile.copy()

    output_profile.update(
        {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "uint8",
            "compress": "lzw",
            "nodata": 0
        }
    )

    print()
    print("Saving final HAB map...")

    with rasterio.open(
        OUTPUT_FILE,
        "w",
        **output_profile
    ) as dst:

        dst.write(
            final_map,
            1
        )

        dst.set_band_description(
            1,
            "HAB classification"
        )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("SUCCESS")
    print("=" * 70)

    print(
        f"Waterbody pixels:     "
        f"{total_water_pixels}"
    )

    print(
        f"Non-HAB pixels:       "
        f"{non_hab_pixels}"
    )

    print(
        f"HAB pixels:           "
        f"{hab_pixels}"
    )

    print(
        f"FINAL HAB percentage: "
        f"{hab_percentage:.2f}%"
    )

    print()
    print(
        "Classification:"
    )

    print(
        "  0 = Outside waterbody"
    )

    print(
        "  1 = Non-HAB"
    )

    print(
        "  2 = HAB"
    )

    print()
    print(
        f"Final map saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This is a model-based HAB indicator "
        "generated by the Swin Transformer."
    )

    print(
        "The model was trained using "
        "spectral pseudo-labels and has not "
        "been field validated."
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()