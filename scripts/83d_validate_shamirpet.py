import os
import glob
import numpy as np
import rasterio

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "sentinel2",
    "shamirpet_test"
)

print("=" * 70)
print("STEP 83D - VALIDATE SHAMIRPET TEST IMAGES")
print("=" * 70)

files = sorted(
    glob.glob(
        os.path.join(INPUT_DIR, "*.tif")
    )
)

print("\nTIFF files found:", len(files))

valid_files = 0
invalid_files = 0

for index, file_path in enumerate(files, start=1):

    filename = os.path.basename(file_path)

    try:

        with rasterio.open(file_path) as src:

            data = src.read()

            # Check bands
            if src.count != 14:
                print(
                    f"INVALID: {filename} "
                    f"- Bands: {src.count}"
                )
                invalid_files += 1
                continue

            # Check dimensions
            if src.width == 0 or src.height == 0:
                print(
                    f"INVALID: {filename} "
                    f"- Invalid dimensions"
                )
                invalid_files += 1
                continue

            # Check finite values
            finite = np.isfinite(data)

            if not finite.all():
                print(
                    f"INVALID: {filename} "
                    f"- NaN/Inf found"
                )
                invalid_files += 1
                continue

            # Check whether image has usable values
            if np.all(data == 0):
                print(
                    f"INVALID: {filename} "
                    f"- All values are zero"
                )
                invalid_files += 1
                continue

            valid_files += 1

        if index % 10 == 0:
            print(
                f"Validated: {index}/{len(files)}"
            )

    except Exception as e:

        print(
            f"INVALID: {filename} - {e}"
        )

        invalid_files += 1


print("\n" + "=" * 70)
print("STEP 83D SUMMARY")
print("=" * 70)

print("Total TIFF files :", len(files))
print("Valid files      :", valid_files)
print("Invalid files    :", invalid_files)

print("\n" + "=" * 70)

if invalid_files == 0:
    print("ALL SHAMIRPET IMAGES ARE VALID")
else:
    print("SOME SHAMIRPET IMAGES ARE INVALID")

print("=" * 70)