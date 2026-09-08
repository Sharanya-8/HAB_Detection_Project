from pathlib import Path
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "sentinel2"

WATER_MASK_PATH = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "hab_masks"
    / "Hussain_Sagar_water_mask.tif"
)

V4_DATES = [
    "2023_01_08",
    "2023_02_12",
    "2023_03_24",
    "2023_05_28",
    "2023_10_05",
    "2024_01_28",
    "2024_03_08",
    "2024_04_22",
    "2024_06_26",
    "2025_01_22",
    "2025_02_26",
    "2025_04_04",
    "2025_06_01",
    "2025_11_13",
    "2025_12_28",
]

print("=" * 80)
print("HUSSAIN SAGAR MULTI-DATE PERCENTILE ANALYSIS")
print("=" * 80)

# Load original water mask
with rasterio.open(WATER_MASK_PATH) as water_src:
    original_water = water_src.read(1)
    water_transform = water_src.transform
    water_crs = water_src.crs

water_pixels_original = original_water == 1

for date in V4_DATES:

    file = PROCESSED_DIR / f"Hussain_Sagar_{date}.tif"

    print()
    print("-" * 80)
    print("DATE:", date)
    print("-" * 80)

    with rasterio.open(file) as src:

        ndci = src.read(13)
        fai = src.read(14)

        aligned_water = np.zeros(
            (src.height, src.width),
            dtype=np.uint8
        )

        reproject(
            source=original_water,
            destination=aligned_water,
            src_transform=water_transform,
            src_crs=water_crs,
            dst_transform=src.transform,
            dst_crs=src.crs,
            resampling=Resampling.nearest
        )

    water_pixels = aligned_water == 1

    valid = (
        water_pixels
        & np.isfinite(ndci)
        & np.isfinite(fai)
        & (ndci != 0)
        & (fai != 0)
    )

    ndci_values = ndci[valid]
    fai_values = fai[valid]

    if len(ndci_values) == 0:
        print("No valid water pixels.")
        continue

    print("Valid water pixels:", len(ndci_values))

    # Percentile thresholds
    ndci_p90 = np.percentile(ndci_values, 90)
    ndci_p95 = np.percentile(ndci_values, 95)
    ndci_p975 = np.percentile(ndci_values, 97.5)

    fai_p90 = np.percentile(fai_values, 90)
    fai_p95 = np.percentile(fai_values, 95)
    fai_p975 = np.percentile(fai_values, 97.5)

    print()
    print("NDCI thresholds:")
    print(f"  90th percentile : {ndci_p90:.6f}")
    print(f"  95th percentile : {ndci_p95:.6f}")
    print(f"  97.5th percentile: {ndci_p975:.6f}")

    print()
    print("FAI thresholds:")
    print(f"  90th percentile : {fai_p90:.6f}")
    print(f"  95th percentile : {fai_p95:.6f}")
    print(f"  97.5th percentile: {fai_p975:.6f}")

    # NDCI-only candidate percentages
    for name, threshold in [
        ("NDCI 90%", ndci_p90),
        ("NDCI 95%", ndci_p95),
        ("NDCI 97.5%", ndci_p975),
    ]:

        candidates = valid & (ndci > threshold)

        percentage = (
            np.count_nonzero(candidates)
            / len(ndci_values)
        ) * 100

        print(
            f"{name:12} candidates: "
            f"{np.count_nonzero(candidates):6} "
            f"({percentage:5.2f}%)"
        )

    # Combined NDCI + FAI anomaly rule
    for percentile_name, ndci_threshold, fai_threshold in [
        ("90%", ndci_p90, fai_p90),
        ("95%", ndci_p95, fai_p95),
        ("97.5%", ndci_p975, fai_p975),
    ]:

        candidates = (
            valid
            & (
                (ndci > ndci_threshold)
                | (fai > fai_threshold)
            )
        )

        percentage = (
            np.count_nonzero(candidates)
            / len(ndci_values)
        ) * 100

        print(
            f"Combined {percentile_name:5} : "
            f"{np.count_nonzero(candidates):6} "
            f"({percentage:5.2f}%)"
        )

print()
print("=" * 80)
print("PERCENTILE ANALYSIS COMPLETE")
print("=" * 80)