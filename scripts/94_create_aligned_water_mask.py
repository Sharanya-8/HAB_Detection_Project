"""
STEP 94
Reusable waterbody mask generator.

Takes:
    - a waterbody boundary GeoJSON
    - any Sentinel-2 raster

Creates:
    - a raster water mask aligned exactly
      with that Sentinel-2 raster.
"""

from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask


# ================================================================
# GENERIC FUNCTION
# ================================================================

def create_aligned_water_mask(
    boundary_path,
    image_path,
    output_mask_path
):
    """
    Create a raster water mask aligned to a Sentinel-2 image.

    Parameters
    ----------
    boundary_path:
        Waterbody boundary GeoJSON.

    image_path:
        14-band Sentinel-2 image.

    output_mask_path:
        Output raster mask path.
    """

    boundary_path = Path(boundary_path)
    image_path = Path(image_path)
    output_mask_path = Path(output_mask_path)

    print("=" * 70)
    print("CREATING ALIGNED WATER MASK")
    print("=" * 70)

    print("\nBoundary:")
    print(boundary_path)

    print("\nSentinel-2 image:")
    print(image_path)

    print("\nOutput mask:")
    print(output_mask_path)

    # ------------------------------------------------------------
    # CHECK INPUTS
    # ------------------------------------------------------------

    if not boundary_path.exists():
        raise FileNotFoundError(
            f"Boundary not found:\n{boundary_path}"
        )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Sentinel-2 image not found:\n{image_path}"
        )

    # ------------------------------------------------------------
    # READ BOUNDARY
    # ------------------------------------------------------------

    print("\nReading waterbody boundary...")

    gdf = gpd.read_file(
        boundary_path
    )

    if gdf.empty:
        raise ValueError(
            "Waterbody boundary contains no geometry."
        )

    print(
        "Boundary CRS:",
        gdf.crs
    )

    print(
        "Geometry:",
        gdf.geometry.iloc[0].geom_type
    )

    print(
        "Valid:",
        gdf.geometry.iloc[0].is_valid
    )

    # ------------------------------------------------------------
    # READ SENTINEL-2 REFERENCE RASTER
    # ------------------------------------------------------------

    print("\nReading Sentinel-2 raster...")

    with rasterio.open(
        image_path
    ) as src:

        height = src.height
        width = src.width
        transform = src.transform
        image_crs = src.crs

    print(
        "Image size:",
        width,
        "x",
        height
    )

    print(
        "Image CRS:",
        image_crs
    )

    # ------------------------------------------------------------
    # REPROJECT BOUNDARY
    # ------------------------------------------------------------

    print(
        "\nReprojecting boundary "
        "to image CRS..."
    )

    if gdf.crs is None:
        raise ValueError(
            "Boundary has no CRS."
        )

    gdf = gdf.to_crs(
        image_crs
    )

    geometries = list(
        gdf.geometry
    )

    # ------------------------------------------------------------
    # CREATE RASTER MASK
    # ------------------------------------------------------------

    print(
        "\nRasterizing waterbody boundary..."
    )

    mask = geometry_mask(
        geometries,
        out_shape=(
            height,
            width
        ),
        transform=transform,
        invert=True
    )

    mask = mask.astype(
        "uint8"
    )

    water_pixels = int(
        mask.sum()
    )

    total_pixels = (
        height *
        width
    )

    water_percentage = (
        water_pixels /
        total_pixels
    ) * 100.0

    print(
        "Water pixels:",
        f"{water_pixels:,}"
    )

    print(
        "Water percentage:",
        f"{water_percentage:.2f}%"
    )

    # ------------------------------------------------------------
    # SAVE MASK
    # ------------------------------------------------------------

    output_mask_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with rasterio.open(
        output_mask_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs=image_crs,
        transform=transform,
        nodata=0
    ) as dst:

        dst.write(
            mask,
            1
        )

    print(
        "\nMask saved successfully:"
    )

    print(
        output_mask_path
    )

    print("=" * 70)

    return output_mask_path


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":

    PROJECT_ROOT = (
        Path(__file__).resolve().parents[1]
    )

    # ------------------------------------------------------------
    # TEST WATERBODY
    # ------------------------------------------------------------

    boundary = (
        PROJECT_ROOT
        / "data"
        / "labels"
        / "hab_masks"
        / "training_waterbodies"
        / "Shamirpet_Lake_OSM_boundary.geojson"
    )

    # ------------------------------------------------------------
    # TEST IMAGE
    # ------------------------------------------------------------

    image = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "sentinel2"
        / "shamirpet_test"
        / "2016-01-30.tif"
    )

    # ------------------------------------------------------------
    # TEST OUTPUT
    # ------------------------------------------------------------

    output = (
        PROJECT_ROOT
        / "results"
        / "water_masks"
        / "Shamirpet"
        / "2016-01-30_water_mask.tif"
    )

    create_aligned_water_mask(
        boundary_path=boundary,
        image_path=image,
        output_mask_path=output
    )

    print()
    print(
        "STEP 94 COMPLETE"
    )