# ============================================================
# GEE UTILITIES
# AI-POWERED HARMFUL ALGAL BLOOM DETECTION
# ============================================================

from pathlib import Path

import ee
import geemap


# ============================================================
# GOOGLE EARTH ENGINE PROJECT
# ============================================================

PROJECT_ID = "remote-sensing-project-507517"


# ============================================================
# SENTINEL-2 CONFIGURATION
# ============================================================

SENTINEL2_COLLECTION = (
    "COPERNICUS/S2_SR_HARMONIZED"
)

CLOUD_PROBABILITY_COLLECTION = (
    "COPERNICUS/S2_CLOUD_PROBABILITY"
)

DEFAULT_CLOUD_THRESHOLD = 40


# ============================================================
# INITIALIZE GOOGLE EARTH ENGINE
# ============================================================

def initialize_gee():
    """
    Initialize Google Earth Engine using the
    configured Google Cloud project.
    """

    try:

        ee.Initialize(
            project=PROJECT_ID
        )

    except Exception:

        ee.Authenticate()

        ee.Initialize(
            project=PROJECT_ID
        )


# ============================================================
# CONVERT GEOJSON TO EARTH ENGINE GEOMETRY
# ============================================================

def geojson_to_ee_geometry(geometry):
    """
    Convert a GeoJSON geometry dictionary into
    an Earth Engine Geometry object.
    """

    if geometry is None:

        raise ValueError(
            "Geometry cannot be None."
        )

    return ee.Geometry(
        geometry
    )


# ============================================================
# GET SENTINEL-2 COLLECTION
# ============================================================

def get_sentinel2_collection(
    aoi,
    start_date,
    end_date
):
    """
    Retrieve Sentinel-2 Surface Reflectance imagery
    for the supplied AOI and date range.
    """

    collection = (
        ee.ImageCollection(
            SENTINEL2_COLLECTION
        )
        .filterBounds(aoi)
        .filterDate(
            start_date,
            end_date
        )
    )

    return collection


# ============================================================
# GET CLOUD PROBABILITY COLLECTION
# ============================================================

def get_cloud_probability_collection(
    aoi,
    start_date,
    end_date
):
    """
    Retrieve Sentinel-2 cloud probability imagery.
    """

    collection = (
        ee.ImageCollection(
            CLOUD_PROBABILITY_COLLECTION
        )
        .filterBounds(aoi)
        .filterDate(
            start_date,
            end_date
        )
    )

    return collection


# ============================================================
# JOIN SENTINEL-2 WITH CLOUD PROBABILITY
# ============================================================

def join_cloud_probability(
    sentinel_collection,
    cloud_collection
):
    """
    Match Sentinel-2 images with their corresponding
    cloud probability images using system:index.
    """

    join = ee.Join.saveFirst(
        "cloud_probability"
    )

    condition = ee.Filter.equals(
        leftField="system:index",
        rightField="system:index"
    )

    joined = join.apply(
        sentinel_collection,
        cloud_collection,
        condition
    )

    return ee.ImageCollection(
        joined
    )


# ============================================================
# CLOUD MASK
# ============================================================

def mask_sentinel2_clouds(
    image,
    threshold=DEFAULT_CLOUD_THRESHOLD
):
    """
    Apply a cloud probability mask to a Sentinel-2 image.

    Pixels with cloud probability below the threshold
    are retained.
    """

    cloud_probability = ee.Image(
        image.get(
            "cloud_probability"
        )
    )

    cloud_mask = (
        cloud_probability
        .select("probability")
        .lt(threshold)
    )

    return (
        image
        .updateMask(cloud_mask)
        .copyProperties(
            image,
            image.propertyNames()
        )
    )


# ============================================================
# GET CLOUD-MASKED SENTINEL-2 COLLECTION
# ============================================================

def get_masked_sentinel2_collection(
    aoi,
    start_date,
    end_date,
    cloud_threshold=DEFAULT_CLOUD_THRESHOLD
):
    """
    Retrieve Sentinel-2 imagery, join cloud probability
    data and apply the cloud mask.
    """

    sentinel_collection = (
        get_sentinel2_collection(
            aoi,
            start_date,
            end_date
        )
    )

    cloud_collection = (
        get_cloud_probability_collection(
            aoi,
            start_date,
            end_date
        )
    )

    joined_collection = (
        join_cloud_probability(
            sentinel_collection,
            cloud_collection
        )
    )

    masked_collection = (
        joined_collection.map(
            lambda image:
            mask_sentinel2_clouds(
                image,
                cloud_threshold
            )
        )
    )

    return masked_collection


# ============================================================
# GET AVAILABLE DATES
# ============================================================

def get_available_dates(
    collection
):
    """
    Return unique Sentinel-2 observation dates
    from an Earth Engine ImageCollection.
    """

    date_list = (
        collection
        .aggregate_array(
            "system:time_start"
        )
        .getInfo()
    )

    dates = []

    for timestamp in date_list:

        date = (
            ee.Date(
                timestamp
            )
            .format(
                "YYYY-MM-dd"
            )
            .getInfo()
        )

        dates.append(
            date
        )

    # Remove duplicates

    dates = sorted(
        list(
            set(dates)
        )
    )

    return dates


# ============================================================
# GET IMAGE FOR A SPECIFIC DATE
# ============================================================

def get_image_for_date(
    collection,
    date
):
    """
    Retrieve the Sentinel-2 image corresponding
    to a specific observation date.

    If multiple Sentinel-2 tiles exist on the same
    date, they are mosaicked together.
    """

    start = ee.Date(
        date
    )

    end = start.advance(
        1,
        "day"
    )

    daily_collection = (
        collection
        .filterDate(
            start,
            end
        )
    )

    image_count = (
        daily_collection
        .size()
        .getInfo()
    )

    if image_count == 0:

        raise ValueError(
            f"No Sentinel-2 image found for {date}."
        )

    # Mosaic multiple tiles from the same date.
    # This is more suitable for arbitrary waterbodies
    # than selecting only the first tile.

    image = (
        daily_collection
        .mosaic()
        .set(
            "system:time_start",
            start.millis()
        )
        .set(
            "selected_date",
            date
        )
    )

    return image


# ============================================================
# CREATE 14-CHANNEL SENTINEL-2 IMAGE
# ============================================================

def create_14_channel_image(
    image
):
    """
    Create the 14-channel Sentinel-2 input used
    by the HAB detection model.

    Channels:

    1.  B2
    2.  B3
    3.  B4
    4.  B5
    5.  B6
    6.  B7
    7.  B8
    8.  B8A
    9.  B11
    10. B12
    11. NDWI
    12. MNDWI
    13. NDCI
    14. FAI

    Sentinel-2 Surface Reflectance bands are stored
    using a scale factor of 10000. Therefore the
    spectral bands are multiplied by 0.0001 before
    calculating the spectral indices.
    """

    # --------------------------------------------------------
    # Sentinel-2 spectral bands
    # --------------------------------------------------------

    spectral_bands = [
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B11",
        "B12"
    ]

    # --------------------------------------------------------
    # Convert Sentinel-2 scaled integer values
    # to surface reflectance values.
    #
    # Example:
    #
    # 535 -> 0.0535
    # 1042 -> 0.1042
    # --------------------------------------------------------

    reflectance = (
        image
        .select(
            spectral_bands
        )
        .multiply(
            0.0001
        )
    )

    # --------------------------------------------------------
    # Individual bands
    # --------------------------------------------------------

    B2 = reflectance.select(
        "B2"
    )

    B3 = reflectance.select(
        "B3"
    )

    B4 = reflectance.select(
        "B4"
    )

    B5 = reflectance.select(
        "B5"
    )

    B6 = reflectance.select(
        "B6"
    )

    B7 = reflectance.select(
        "B7"
    )

    B8 = reflectance.select(
        "B8"
    )

    B8A = reflectance.select(
        "B8A"
    )

    B11 = reflectance.select(
        "B11"
    )

    B12 = reflectance.select(
        "B12"
    )

    # ========================================================
    # NDWI
    # ========================================================
    #
    # NDWI = (Green - NIR) / (Green + NIR)
    #
    # Green = B3
    # NIR   = B8
    #
    # ========================================================

    ndwi = (
        B3.subtract(B8)
        .divide(
            B3.add(B8)
        )
        .rename("NDWI")
    )

    # ========================================================
    # MNDWI
    # ========================================================
    #
    # MNDWI = (Green - SWIR1) / (Green + SWIR1)
    #
    # Green = B3
    # SWIR1 = B11
    #
    # ========================================================

    mndwi = (
        B3.subtract(B11)
        .divide(
            B3.add(B11)
        )
        .rename("MNDWI")
    )

    # ========================================================
    # NDCI
    # ========================================================
    #
    # NDCI = (Red Edge - Red) /
    #        (Red Edge + Red)
    #
    # Red Edge = B5
    # Red      = B4
    #
    # ========================================================

    ndci = (
        B5.subtract(B4)
        .divide(
            B5.add(B4)
        )
        .rename("NDCI")
    )

    # ========================================================
    # FAI
    # ========================================================
    #
    # Floating Algae Index
    #
    # Red  = B4  = approximately 665 nm
    # NIR  = B8  = approximately 842 nm
    # SWIR = B11 = approximately 1610 nm
    #
    # ========================================================

    wavelength_red = 665.0

    wavelength_nir = 842.0

    wavelength_swir = 1610.0

    baseline = (
        B4.add(
            B11.subtract(B4)
            .multiply(
                (
                    wavelength_nir
                    - wavelength_red
                )
                /
                (
                    wavelength_swir
                    - wavelength_red
                )
            )
        )
    )

    fai = (
        B8.subtract(
            baseline
        )
        .rename(
            "FAI"
        )
    )

    # ========================================================
    # COMBINE 14 CHANNELS
    # ========================================================

    combined = (
        B2
        .addBands(B3)
        .addBands(B4)
        .addBands(B5)
        .addBands(B6)
        .addBands(B7)
        .addBands(B8)
        .addBands(B8A)
        .addBands(B11)
        .addBands(B12)
        .addBands(ndwi)
        .addBands(mndwi)
        .addBands(ndci)
        .addBands(fai)
    )

    return combined


# ============================================================
# DOWNLOAD 14-CHANNEL IMAGE
# ============================================================

def download_14_channel_image(
    image,
    geometry,
    output_path,
    scale=10
):
    """
    Download the 14-channel image from Google Earth Engine
    as a GeoTIFF.

    The selected waterbody geometry is used as the
    download region.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Convert GeoJSON geometry to EE geometry if necessary
    # --------------------------------------------------------

    if isinstance(
        geometry,
        dict
    ):

        region = geojson_to_ee_geometry(
            geometry
        )

    else:

        region = geometry

    # --------------------------------------------------------
    # Clip image to waterbody
    # --------------------------------------------------------

    clipped_image = (
        image
        .clip(region)
    )

    # --------------------------------------------------------
    # Download using geemap
    # --------------------------------------------------------

    geemap.ee_export_image(
        clipped_image,
        filename=str(
            output_path
        ),
        scale=scale,
        region=region,
        file_per_band=False
    )

    return output_path