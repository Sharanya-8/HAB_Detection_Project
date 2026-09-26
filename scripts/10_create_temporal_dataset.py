from pathlib import Path
import numpy as np
import pandas as pd
import rasterio


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]

IMAGE_DIR = PROJECT_DIR / "data" / "processed" / "sentinel2"
LABEL_DIR = PROJECT_DIR / "data" / "labels" / "hab_masks"

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "Hussain_Sagar_temporal_features.csv"


# ---------------------------------------------------------
# FIND OUR 15 DATES
# ---------------------------------------------------------
image_files = sorted(IMAGE_DIR.glob("Hussain_Sagar_Large_*.tif"))

print("=" * 60)
print("OBJECTIVE 3 - TEMPORAL DATASET CREATION")
print("=" * 60)

print(f"Images found: {len(image_files)}")

if len(image_files) == 0:
    raise FileNotFoundError("No Hussain Sagar processed images found.")

rows = []


# ---------------------------------------------------------
# PROCESS EACH DATE
# ---------------------------------------------------------
for image_file in image_files:

    # Extract date from filename
    # Example:
    # Hussain_Sagar_Large_2023_01_08.tif
    #                         2023_01_08
    date_text = image_file.stem.replace(
        "Hussain_Sagar_Large_", ""
    )

    label_file = LABEL_DIR / f"Hussain_Sagar_HAB_{date_text}.tif"

    if not label_file.exists():
        print(f"WARNING: Label missing for {date_text}")
        continue

    print(f"\nProcessing: {date_text}")

    # -----------------------------------------------------
    # READ IMAGE
    # -----------------------------------------------------
    with rasterio.open(image_file) as src:

        image = src.read().astype(np.float32)

        # Band numbers in our 14-band dataset:
        #
        # 1-10 = Sentinel-2 bands
        # 11 = NDWI
        # 12 = MNDWI
        # 13 = NDCI
        # 14 = FAI

        ndwi = image[10]
        mndwi = image[11]
        ndci = image[12]
        fai = image[13]


    # -----------------------------------------------------
    # READ HAB LABEL
    # -----------------------------------------------------
    with rasterio.open(label_file) as src:
        mask = src.read(1)


    # -----------------------------------------------------
    # VALID WATER PIXELS
    # -----------------------------------------------------
    valid = (
        (mask != 255)
        & np.isfinite(ndwi)
        & np.isfinite(mndwi)
        & np.isfinite(ndci)
        & np.isfinite(fai)
    )

    valid_pixels = np.sum(valid)

    if valid_pixels == 0:
        print("WARNING: No valid pixels")
        continue


    # -----------------------------------------------------
    # EXTRACT VALID VALUES
    # -----------------------------------------------------
    ndwi_values = ndwi[valid]
    mndwi_values = mndwi[valid]
    ndci_values = ndci[valid]
    fai_values = fai[valid]


    # -----------------------------------------------------
    # HAB PIXELS
    # -----------------------------------------------------
    hab_pixels = np.sum(mask[valid] == 1)

    hab_percentage = (
        hab_pixels / valid_pixels
    ) * 100


    # -----------------------------------------------------
    # CREATE ONE ROW
    # -----------------------------------------------------
    row = {
        "date": date_text,

        "valid_water_pixels": int(valid_pixels),
        "hab_pixels": int(hab_pixels),
        "hab_percentage": float(hab_percentage),

        "ndwi_mean": float(np.mean(ndwi_values)),
        "ndwi_median": float(np.median(ndwi_values)),

        "mndwi_mean": float(np.mean(mndwi_values)),
        "mndwi_median": float(np.median(mndwi_values)),

        "ndci_mean": float(np.mean(ndci_values)),
        "ndci_median": float(np.median(ndci_values)),
        "ndci_max": float(np.max(ndci_values)),

        "fai_mean": float(np.mean(fai_values)),
        "fai_median": float(np.median(fai_values)),
        "fai_max": float(np.max(fai_values)),
    }

    rows.append(row)

    print(f"Valid water pixels : {valid_pixels}")
    print(f"HAB pixels         : {hab_pixels}")
    print(f"HAB percentage     : {hab_percentage:.2f}%")


# ---------------------------------------------------------
# CREATE DATAFRAME
# ---------------------------------------------------------
if len(rows) == 0:
    raise RuntimeError("No temporal observations were created.")

df = pd.DataFrame(rows)

# Convert date to proper datetime
df["date"] = pd.to_datetime(
    df["date"],
    format="%Y_%m_%d"
)

# Sort chronologically
df = df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------
df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("TEMPORAL DATASET CREATED")
print("=" * 60)

print(f"Observations: {len(df)}")
print(f"Output file : {OUTPUT_FILE}")

print("\nDataset:")
print(df.to_string(index=False))

print("\nSaved successfully.")