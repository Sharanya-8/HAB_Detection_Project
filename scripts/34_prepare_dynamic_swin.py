from pathlib import Path
import sys

import numpy as np
import rasterio


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# INPUT / OUTPUT
# ============================================================

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sentinel2"
    / "Dynamic_Hussain_Sagar_2025-01-02_14Channel.tif"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "tiles"
    / "dynamic_swin"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

TILE_SIZE = 256


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DYNAMIC SWIN INPUT PREPARATION")
    print("=" * 70)

    print()
    print("Input:")
    print(INPUT_FILE)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    with rasterio.open(
        INPUT_FILE
    ) as src:

        image = src.read()

        height = src.height
        width = src.width
        bands = src.count

        transform = src.transform
        crs = src.crs

    print()
    print("Image information:")
    print(f"  Bands:  {bands}")
    print(f"  Width:  {width}")
    print(f"  Height: {height}")
    print(f"  CRS:    {crs}")

    # --------------------------------------------------------
    # VERIFY 14 CHANNELS
    # --------------------------------------------------------

    if bands != 14:

        raise ValueError(
            f"Expected 14 bands, but found {bands}."
        )

    # --------------------------------------------------------
    # CONVERT TO FLOAT32
    # --------------------------------------------------------

    image = image.astype(
        np.float32
    )

    # --------------------------------------------------------
    # HANDLE INVALID VALUES
    # --------------------------------------------------------

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # --------------------------------------------------------
    # CREATE 256 × 256 TILES
    # --------------------------------------------------------

    tile_number = 0

    for row in range(
        0,
        height,
        TILE_SIZE
    ):

        for col in range(
            0,
            width,
            TILE_SIZE
        ):

            row_end = min(
                row + TILE_SIZE,
                height
            )

            col_end = min(
                col + TILE_SIZE,
                width
            )

            tile = image[
                :,
                row:row_end,
                col:col_end
            ]

            actual_height = (
                row_end - row
            )

            actual_width = (
                col_end - col
            )

            # ------------------------------------------------
            # PAD SMALL EDGE TILES
            # ------------------------------------------------

            padded_tile = np.zeros(
                (
                    bands,
                    TILE_SIZE,
                    TILE_SIZE
                ),
                dtype=np.float32
            )

            padded_tile[
                :,
                :actual_height,
                :actual_width
            ] = tile

            # ------------------------------------------------
            # SAVE TILE
            # ------------------------------------------------

            tile_number += 1

            tile_path = (
                OUTPUT_DIR
                / (
                    f"dynamic_tile_"
                    f"{tile_number:03d}.npy"
                )
            )

            np.save(
                tile_path,
                padded_tile
            )

            print(
                f"  Tile {tile_number:03d}: "
                f"source={actual_height}x{actual_width} "
                f"-> saved=256x256"
            )

    # --------------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------------

    metadata_path = (
        OUTPUT_DIR
        / "metadata.txt"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "Dynamic Swin Input Metadata\n"
        )

        f.write(
            "============================\n"
        )

        f.write(
            f"Input file: {INPUT_FILE.name}\n"
        )

        f.write(
            f"Bands: {bands}\n"
        )

        f.write(
            f"Original width: {width}\n"
        )

        f.write(
            f"Original height: {height}\n"
        )

        f.write(
            f"CRS: {crs}\n"
        )

        f.write(
            f"Tile size: {TILE_SIZE}\n"
        )

        f.write(
            f"Transform: {transform}\n"
        )

    print()
    print("=" * 70)
    print("SUCCESS")
    print("=" * 70)

    print(
        f"Created {tile_number} dynamic Swin tiles."
    )

    print(
        f"Output directory:\n{OUTPUT_DIR}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()