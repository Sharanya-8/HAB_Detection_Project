from pathlib import Path
import rasterio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sentinel2"
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

files = [
    PROCESSED_DIR / f"Hussain_Sagar_{date}.tif"
    for date in V4_DATES
]

print("=" * 80)
print("HUSSAIN SAGAR V4 MULTI-DATE QUALITY CHECK")
print("=" * 80)

print(f"Expected files: {len(files)}")
print()

missing_files = []

for file in files:

    if not file.exists():
        missing_files.append(file.name)
        print(f"{file.name:45} MISSING")
        continue

    with rasterio.open(file) as src:

        data = src.read()

        valid_pixels = np.count_nonzero(
            np.isfinite(data[0]) & (data[0] != 0)
        )

        total_pixels = data.shape[1] * data.shape[2]

        valid_percentage = (
            valid_pixels / total_pixels
        ) * 100

        print(
            f"{file.name:45} "
            f"Valid: {valid_pixels:6} / {total_pixels:6} "
            f"({valid_percentage:6.2f}%)"
        )

print()
print("=" * 80)

if missing_files:
    print("MISSING FILES:")
    for name in missing_files:
        print(" -", name)
    print("STATUS: CHECK FILES")
else:
    print("All 15 V4 files found.")
    print("STATUS: SUCCESS")

print("=" * 80)