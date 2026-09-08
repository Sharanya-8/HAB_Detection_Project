# ============================================================
# GOOGLE EARTH ENGINE UTILITIES
# ============================================================

import ee


def initialize_gee(project_id):
    """
    Initialize Google Earth Engine.

    Parameters
    ----------
    project_id : str
        Google Cloud project ID.
    """

    try:
        ee.Initialize(project=project_id)
        print("Google Earth Engine initialized successfully.")

    except Exception as error:
        print("Google Earth Engine initialization failed.")
        print(error)
        raise


def create_aoi(
    min_lon,
    min_lat,
    max_lon,
    max_lat
):
    """
    Create a rectangular Area of Interest (AOI).

    Returns
    -------
    ee.Geometry
        Rectangular Earth Engine geometry.
    """

    return ee.Geometry.Rectangle(
        [
            min_lon,
            min_lat,
            max_lon,
            max_lat
        ]
    )


def get_sentinel2_collection(
    aoi,
    start_date,
    end_date,
    max_cloud
):
    """
    Retrieve Sentinel-2 Surface Reflectance imagery
    for the specified AOI and date range.
    """

    collection = (
        ee.ImageCollection(
            "COPERNICUS/S2_SR_HARMONIZED"
        )
        .filterBounds(aoi)
        .filterDate(
            start_date,
            end_date
        )
        .filter(
            ee.Filter.lte(
                "CLOUDY_PIXEL_PERCENTAGE",
                max_cloud
            )
        )
    )

    return collection


def calculate_ndwi(image):
    """
    Calculate Normalized Difference Water Index.
    """

    return image.normalizedDifference(
        ["B3", "B8"]
    ).rename("NDWI")


def calculate_mndwi(image):
    """
    Calculate Modified Normalized Difference
    Water Index.
    """

    return image.normalizedDifference(
        ["B3", "B11"]
    ).rename("MNDWI")


def calculate_ndci(image):
    """
    Calculate Normalized Difference Chlorophyll Index.
    """

    return image.normalizedDifference(
        ["B5", "B4"]
    ).rename("NDCI")


def add_spectral_indices(image):
    """
    Add NDWI, MNDWI and NDCI to a Sentinel-2 image.
    """

    ndwi = calculate_ndwi(image)
    mndwi = calculate_mndwi(image)
    ndci = calculate_ndci(image)

    return image.addBands(
        [
            ndwi,
            mndwi,
            ndci
        ]
    )