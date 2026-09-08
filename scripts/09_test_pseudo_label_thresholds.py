import rasterio
import numpy as np

file_path = "data/raw/sentinel2/HAB_Sentinel2_2023_2025.tif"

with rasterio.open(file_path) as src:

    ndwi = src.read(11)
    mndwi = src.read(12)
    ndci = src.read(13)
    fai = src.read(14)

# Remove invalid values
valid = (
    np.isfinite(ndwi) &
    np.isfinite(mndwi) &
    np.isfinite(ndci) &
    np.isfinite(fai)
)

# Convert to 1D arrays for analysis
ndwi_v = ndwi[valid]
mndwi_v = mndwi[valid]
ndci_v = ndci[valid]
fai_v = fai[valid]

print("PSEUDO-LABEL THRESHOLD TEST")
print("============================")
print("Total valid pixels:", len(ndci_v))

# Candidate water masks
water_masks = {
    "NDWI > 0.00": ndwi_v > 0.00,
    "NDWI > 0.05": ndwi_v > 0.05,
    "MNDWI > 0.00": mndwi_v > 0.00,
    "MNDWI > 0.05": mndwi_v > 0.05,
    "MNDWI > 0.10": mndwi_v > 0.10,
}

print("\nWATER PIXEL TEST")
print("----------------")

for name, mask in water_masks.items():

    count = np.sum(mask)
    percentage = (count / len(ndci_v)) * 100

    print(
        f"{name}: "
        f"{count} pixels "
        f"({percentage:.2f}%)"
    )


# Test high NDCI / FAI thresholds
ndci_thresholds = [0.05, 0.10, 0.15, 0.20]
fai_thresholds = [0.02, 0.05, 0.08, 0.10]

print("\nNDCI THRESHOLD TEST")
print("------------------")

for threshold in ndci_thresholds:

    mask = ndci_v > threshold

    count = np.sum(mask)
    percentage = (count / len(ndci_v)) * 100

    print(
        f"NDCI > {threshold}: "
        f"{count} pixels "
        f"({percentage:.2f}%)"
    )


print("\nFAI THRESHOLD TEST")
print("------------------")

for threshold in fai_thresholds:

    mask = fai_v > threshold

    count = np.sum(mask)
    percentage = (count / len(fai_v)) * 100

    print(
        f"FAI > {threshold}: "
        f"{count} pixels "
        f"({percentage:.2f}%)"
    )


# Combined HAB candidate test
print("\nCOMBINED HAB CANDIDATE TEST")
print("---------------------------")

for water_name, water_mask in water_masks.items():

    for ndci_threshold in [0.05, 0.10, 0.15]:

        for fai_threshold in [0.02, 0.05]:

            hab_mask = (
                water_mask &
                (
                    (ndci_v > ndci_threshold) |
                    (fai_v > fai_threshold)
                )
            )

            count = np.sum(hab_mask)
            percentage = (count / len(ndci_v)) * 100

            print(
                f"{water_name}, "
                f"NDCI > {ndci_threshold}, "
                f"FAI > {fai_threshold}: "
                f"{count} pixels "
                f"({percentage:.2f}%)"
            )