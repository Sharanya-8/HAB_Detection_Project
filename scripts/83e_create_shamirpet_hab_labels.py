import os
import glob
import numpy as np
import rasterio
import pandas as pd

# ============================================================
# STEP 83E - CREATE SHAMIRPET HAB PSEUDO-LABELS
# ============================================================

INPUT_DIR = r"data\raw\sentinel2\shamirpet_test"
OUTPUT_DIR = r"data\labels\hab_masks\shamirpet_test"
STATS_FILE = r"data\processed\shamirpet_test_label_statistics.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("STEP 83E - CREATE SHAMIRPET HAB PSEUDO-LABELS")
print("=" * 70)

files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.tif")))

print(f"\nInput images found: {len(files)}")

if len(files) == 0:
    raise RuntimeError("No Shamirpet TIFF files found.")

records = []
success = 0
failed = 0

for i, tif_path in enumerate(files, start=1):

    filename = os.path.basename(tif_path)
    date = filename.replace(".tif", "")

    try:
        with rasterio.open(tif_path) as src:

            # 14-band structure:
            # 1-10 = Sentinel-2 bands
            # 11 = NDWI
            # 12 = MNDWI
            # 13 = NDCI
            # 14 = FAI

            ndci = src.read(13).astype(np.float32)
            fai = src.read(14).astype(np.float32)

            profile = src.profile.copy()

        # Valid pixels
        valid = (
            np.isfinite(ndci)
            & np.isfinite(fai)
            & ((ndci != 0) | (fai != 0))
        )

        valid_count = int(np.sum(valid))

        if valid_count == 0:
            print(f"FAILED: {filename} - no valid NDCI/FAI pixels")
            failed += 1
            continue

        # ----------------------------------------------------
        # Same methodology used for training labels:
        # 95th percentile thresholds
        # ----------------------------------------------------

        ndci_valid = ndci[valid]
        fai_valid = fai[valid]

        ndci_threshold = float(np.percentile(ndci_valid, 95))
        fai_threshold = float(np.percentile(fai_valid, 95))

        # HAB = NDCI above threshold OR FAI above threshold
        hab = (
            valid
            & (
                (ndci > ndci_threshold)
                | (fai > fai_threshold)
            )
        )

        # ----------------------------------------------------
        # Create mask
        # 0   = Non-HAB
        # 1   = HAB
        # 255 = invalid
        # ----------------------------------------------------

        mask = np.full(ndci.shape, 255, dtype=np.uint8)

        mask[valid] = 0
        mask[hab] = 1

        # Output profile
        mask_profile = profile.copy()
        mask_profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress="lzw",
            nodata=255
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{date}_HAB.tif"
        )

        with rasterio.open(output_path, "w", **mask_profile) as dst:
            dst.write(mask, 1)

        hab_pixels = int(np.sum(hab))
        non_hab_pixels = int(np.sum(valid) - hab_pixels)

        hab_percent = (
            hab_pixels / valid_count * 100
            if valid_count > 0
            else 0
        )

        # Approximate area at 10 m resolution
        pixel_area_ha = 0.01
        hab_area_ha = hab_pixels * pixel_area_ha

        records.append({
            "date": date,
            "image": filename,
            "ndci_threshold": ndci_threshold,
            "fai_threshold": fai_threshold,
            "valid_pixels": valid_count,
            "hab_pixels": hab_pixels,
            "non_hab_pixels": non_hab_pixels,
            "hab_percent": hab_percent,
            "hab_area_ha": hab_area_ha
        })

        success += 1

        if i % 10 == 0 or i == len(files):
            print(f"Processed: {i}/{len(files)}")

    except Exception as e:
        print(f"FAILED: {filename}")
        print(f"       {e}")
        failed += 1


# ============================================================
# SAVE STATISTICS
# ============================================================

if records:
    df = pd.DataFrame(records)

    os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)
    df.to_csv(STATS_FILE, index=False)

else:
    df = pd.DataFrame()


print("\n" + "=" * 70)
print("STEP 83E SUMMARY")
print("=" * 70)

print(f"Input images : {len(files)}")
print(f"Labels created: {success}")
print(f"Failed        : {failed}")

if len(df) > 0:
    print(f"\nTotal valid pixels : {df['valid_pixels'].sum():,}")
    print(f"Total HAB pixels   : {df['hab_pixels'].sum():,}")

    total_valid = df["valid_pixels"].sum()
    total_hab = df["hab_pixels"].sum()

    overall_hab_percent = (
        total_hab / total_valid * 100
        if total_valid > 0
        else 0
    )

    print(f"Overall HAB %      : {overall_hab_percent:.2f}%")
    print(f"Mean image HAB %   : {df['hab_percent'].mean():.2f}%")
    print(f"Median image HAB % : {df['hab_percent'].median():.2f}%")

    print(f"\nStatistics saved to:")
    print(STATS_FILE)

print("\n" + "=" * 70)
print("STEP 83E COMPLETE")
print("=" * 70)