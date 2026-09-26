from pathlib import Path

import numpy as np
import rasterio


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/raw/sentinel2/"
    "Dynamic_Hussain_Sagar_14Channel_Test.tif"
)

OUTPUT_FILE = Path(
    "data/processed/sentinel2/"
    "Dynamic_Hussain_Sagar_14Channel_Corrected.tif"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHANNEL ORDER
# ============================================================

CHANNEL_NAMES = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
    "NDWI",
    "MNDWI",
    "NDCI",
    "FAI"
]


# ============================================================
# START
# ============================================================

print("=" * 70)
print("CORRECT DYNAMIC SENTINEL-2 SCALING")
print("=" * 70)


# ============================================================
# READ IMAGE
# ============================================================

print("\nReading dynamic Sentinel-2 image...")

with rasterio.open(INPUT_FILE) as src:

    image = src.read().astype(np.float32)

    profile = src.profile.copy()


print(f"Original shape: {image.shape}")


if image.shape[0] != 14:

    raise RuntimeError(
        f"Expected 14 channels, found {image.shape[0]}"
    )


# ============================================================
# STEP 1: SEPARATE SPECTRAL BANDS
# ============================================================

print("\nConverting Sentinel-2 reflectance bands...")


# First 10 channels are Sentinel-2 reflectance bands.

spectral = image[:10]


# Sentinel-2 surface reflectance is scaled by 10000.

spectral = spectral / 10000.0


# Handle invalid values.

spectral = np.nan_to_num(
    spectral,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


# ============================================================
# STEP 2: RE-CALCULATE INDICES FROM REFLECTANCE
# ============================================================

print("Recalculating spectral indices...")


B2 = spectral[0]
B3 = spectral[1]
B4 = spectral[2]
B5 = spectral[3]
B6 = spectral[4]
B7 = spectral[5]
B8 = spectral[6]
B8A = spectral[7]
B11 = spectral[8]
B12 = spectral[9]


# ------------------------------------------------------------
# NDWI
# ------------------------------------------------------------

ndwi_denominator = B3 + B8

ndwi = np.divide(
    B3 - B8,
    ndwi_denominator,
    out=np.zeros_like(B3),
    where=ndwi_denominator != 0
)


# ------------------------------------------------------------
# MNDWI
# ------------------------------------------------------------

mndwi_denominator = B3 + B11

mndwi = np.divide(
    B3 - B11,
    mndwi_denominator,
    out=np.zeros_like(B3),
    where=mndwi_denominator != 0
)


# ------------------------------------------------------------
# NDCI
# ------------------------------------------------------------

ndci_denominator = B5 + B4

ndci = np.divide(
    B5 - B4,
    ndci_denominator,
    out=np.zeros_like(B5),
    where=ndci_denominator != 0
)


# ------------------------------------------------------------
# FAI
# ------------------------------------------------------------

# Sentinel-2 wavelengths:
#
# B4  = 665 nm
# B8  = 842 nm
# B11 = 1610 nm

baseline = B4 + (
    (B11 - B4)
    *
    ((842.0 - 665.0) / (1610.0 - 665.0))
)


fai = B8 - baseline


# ============================================================
# STEP 3: CREATE CORRECTED 14-CHANNEL IMAGE
# ============================================================

corrected = np.stack(
    [
        B2,
        B3,
        B4,
        B5,
        B6,
        B7,
        B8,
        B8A,
        B11,
        B12,
        ndwi,
        mndwi,
        ndci,
        fai
    ],
    axis=0
).astype(np.float32)


# ============================================================
# STEP 4: HANDLE INVALID VALUES
# ============================================================

corrected = np.nan_to_num(
    corrected,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


# ============================================================
# STEP 5: WRITE GEOTIFF
# ============================================================

print("\nSaving corrected GeoTIFF...")


profile.update(
    dtype="float32",
    count=14,
    compress="lzw"
)


with rasterio.open(
    OUTPUT_FILE,
    "w",
    **profile
) as dst:

    dst.write(corrected)

    for i, name in enumerate(
        CHANNEL_NAMES,
        start=1
    ):

        dst.set_band_description(
            i,
            name
        )


print(
    f"Saved: {OUTPUT_FILE}"
)


# ============================================================
# STEP 6: PRINT STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("CORRECTED IMAGE STATISTICS")
print("=" * 70)


for i, name in enumerate(
    CHANNEL_NAMES
):

    band = corrected[i]

    print(
        f"Channel {i+1:2d} {name:6s}: "
        f"min={band.min():.6f}, "
        f"max={band.max():.6f}, "
        f"mean={band.mean():.6f}, "
        f"median={np.median(band):.6f}"
    )


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print(
    "Dynamic Sentinel-2 data has been converted "
    "to the same general reflectance scale used "
    "by the Swin training dataset."
)

print("=" * 70)