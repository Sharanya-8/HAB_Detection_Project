from pathlib import Path
import numpy as np
import rasterio


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "sentinel2"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "sentinel2"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EXPECTED BAND NAMES
# ============================================================

EXPECTED_BANDS = [
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
# PREPROCESS ONE IMAGE
# ============================================================

def preprocess_image(input_path, output_path):

    print()
    print("-" * 60)
    print("Processing:", input_path.name)
    print("-" * 60)

    with rasterio.open(input_path) as src:

        data = src.read()

        profile = src.profile.copy()

        descriptions = src.descriptions

        print("Input shape:", data.shape)
        print("Input dtype:", data.dtype)
        print("CRS:", src.crs)
        print("Resolution:", src.res)

        # ----------------------------------------------------
        # Verify number of bands
        # ----------------------------------------------------

        if src.count != 14:
            raise ValueError(
                f"{input_path.name}: Expected 14 bands, "
                f"found {src.count}"
            )

        # ----------------------------------------------------
        # Verify band names
        # ----------------------------------------------------

        if descriptions != tuple(EXPECTED_BANDS):

            print("Warning: Band descriptions are:")
            print(descriptions)

        # ----------------------------------------------------
        # Convert to Float32
        # ----------------------------------------------------

        data = data.astype(np.float32)

        # ----------------------------------------------------
        # Replace invalid numerical values
        # ----------------------------------------------------

        invalid_before = np.sum(
            ~np.isfinite(data)
        )

        data = np.nan_to_num(
            data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        invalid_after = np.sum(
            ~np.isfinite(data)
        )

        print(
            "Invalid values before:",
            invalid_before
        )

        print(
            "Invalid values after:",
            invalid_after
        )

        # ----------------------------------------------------
        # Output profile
        # ----------------------------------------------------

        profile.update(
            dtype="float32",
            count=14,
            compress="lzw"
        )

        # ----------------------------------------------------
        # Write processed image
        # ----------------------------------------------------

        with rasterio.open(
            output_path,
            "w",
            **profile
        ) as dst:

            dst.write(data)

            for band_number, band_name in enumerate(
                EXPECTED_BANDS,
                start=1
            ):

                dst.set_band_description(
                    band_number,
                    band_name
                )

    print(
        "Saved:",
        output_path.name
    )


# ============================================================
# PROCESS ALL 15 LARGE-AOI HUSSAIN SAGAR IMAGES
# ============================================================

def main():

    print("=" * 60)
    print("HUSSAIN SAGAR LARGE-AOI PREPROCESSING")
    print("=" * 60)

    input_files = sorted(
        RAW_DIR.glob(
            "Hussain_Sagar_Large_*.tif"
        )
    )

    print(
        "Input files found:",
        len(input_files)
    )

    if len(input_files) != 15:

        raise ValueError(
            f"Expected 15 large-AOI images, "
            f"found {len(input_files)}"
        )

    # --------------------------------------------------------
    # Process each date
    # --------------------------------------------------------

    for input_path in input_files:

        output_name = input_path.name.replace(
            "Hussain_Sagar_Large_",
            "Hussain_Sagar_Large_"
        )

        output_path = (
            PROCESSED_DIR /
            output_name
        )

        preprocess_image(
            input_path,
            output_path
        )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    output_files = sorted(
        PROCESSED_DIR.glob(
            "Hussain_Sagar_Large_*.tif"
        )
    )

    print()
    print("=" * 60)
    print("LARGE-AOI PREPROCESSING COMPLETE")
    print("=" * 60)

    print(
        "Processed large-AOI files:",
        len(output_files)
    )

    if len(output_files) == 15:

        print(
            "STATUS: SUCCESS"
        )

    else:

        print(
            "STATUS: CHECK OUTPUT"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()