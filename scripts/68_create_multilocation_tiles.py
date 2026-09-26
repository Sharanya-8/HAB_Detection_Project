from pathlib import Path
import rasterio
import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

IMAGE_ROOT = Path("data/raw/sentinel2/multi_waterbody")
LABEL_ROOT = Path("data/labels/hab_masks/multi_waterbody")

IMAGE_OUTPUT = Path("data/tiles/images")
MASK_OUTPUT = Path("data/tiles/masks")

REPORT_OUTPUT = Path(
    "data/processed/multilocation_tile_statistics.csv"
)

IMAGE_OUTPUT.mkdir(parents=True, exist_ok=True)
MASK_OUTPUT.mkdir(parents=True, exist_ok=True)


# =========================================================
# CONFIGURATION
# =========================================================

TILE_SIZE = 256

# Minimum percentage of real image pixels required.
MIN_VALID_PERCENT = 20.0

# Tile is considered HAB-containing if at least 1%
# of its valid pixels are HAB.
MIN_HAB_PERCENT = 1.0


# =========================================================
# CLEAR OLD TILES
# =========================================================

print("=" * 70)
print("STEP 69 - CORRECTED MULTI-WATERBODY TILE CREATION")
print("=" * 70)

print("Removing previously generated tiles...")

for f in IMAGE_OUTPUT.glob("*.npy"):
    f.unlink()

for f in MASK_OUTPUT.glob("*.npy"):
    f.unlink()

print("Old tiles removed.")
print()


# =========================================================
# FIND INPUT IMAGES
# =========================================================

image_files = sorted(
    IMAGE_ROOT.rglob("*.tif")
)

print(f"Input images found: {len(image_files)}")
print()


# =========================================================
# STORAGE
# =========================================================

records = []

total_tiles = 0
hab_tiles = 0
non_hab_tiles = 0
skipped_images = 0


# =========================================================
# PROCESS EACH IMAGE
# =========================================================

for image_file in image_files:

    waterbody = image_file.parent.name
    date = image_file.stem

    label_file = (
        LABEL_ROOT
        / waterbody
        / f"{date}_HAB.tif"
    )

    # -----------------------------------------------------
    # Missing label
    # -----------------------------------------------------

    if not label_file.exists():

        print(
            f"SKIP - no label: "
            f"{waterbody}/{date}"
        )

        skipped_images += 1
        continue

    try:

        # -------------------------------------------------
        # READ IMAGE
        # -------------------------------------------------

        with rasterio.open(image_file) as src:
            image = src.read()

        # Expected:
        # (14, height, width)

        # -------------------------------------------------
        # READ MASK
        # -------------------------------------------------

        with rasterio.open(label_file) as src:
            mask = src.read(1)

        # -------------------------------------------------
        # CHECK SHAPE
        # -------------------------------------------------

        if image.shape[1:] != mask.shape:

            print(
                f"SKIP - shape mismatch: "
                f"{waterbody}/{date}"
            )

            skipped_images += 1
            continue

        height = image.shape[1]
        width = image.shape[2]

        tile_number = 0

        # -------------------------------------------------
        # TILE GRID
        # -------------------------------------------------

        for y in range(0, height, TILE_SIZE):

            for x in range(0, width, TILE_SIZE):

                y_end = min(
                    y + TILE_SIZE,
                    height
                )

                x_end = min(
                    x + TILE_SIZE,
                    width
                )

                actual_height = y_end - y
                actual_width = x_end - x

                # -----------------------------------------
                # Extract actual region
                # -----------------------------------------

                image_region = image[
                    :,
                    y:y_end,
                    x:x_end
                ]

                mask_region = mask[
                    y:y_end,
                    x:x_end
                ]

                # -----------------------------------------
                # Calculate valid pixels BEFORE padding
                # -----------------------------------------

                valid_region = (
                    mask_region != 255
                )

                valid_count = np.sum(
                    valid_region
                )

                if valid_count == 0:
                    continue

                valid_percent = (
                    valid_count /
                    (
                        actual_height *
                        actual_width
                    )
                ) * 100

                if valid_percent < MIN_VALID_PERCENT:
                    continue

                # -----------------------------------------
                # Statistics BEFORE padding
                # -----------------------------------------

                hab_count = np.sum(
                    mask_region[
                        valid_region
                    ] == 1
                )

                non_hab_count = np.sum(
                    mask_region[
                        valid_region
                    ] == 0
                )

                hab_percent = (
                    hab_count /
                    valid_count
                ) * 100

                non_hab_percent = (
                    non_hab_count /
                    valid_count
                ) * 100

                # -----------------------------------------
                # PAD IMAGE TO 256 x 256
                # -----------------------------------------

                padded_image = np.zeros(
                    (
                        image.shape[0],
                        TILE_SIZE,
                        TILE_SIZE
                    ),
                    dtype=np.float32
                )

                padded_mask = np.full(
                    (
                        TILE_SIZE,
                        TILE_SIZE
                    ),
                    255,
                    dtype=np.uint8
                )

                padded_image[
                    :,
                    :actual_height,
                    :actual_width
                ] = image_region.astype(
                    np.float32
                )

                padded_mask[
                    :actual_height,
                    :actual_width
                ] = mask_region.astype(
                    np.uint8
                )

                # -----------------------------------------
                # TILE ID
                # -----------------------------------------

                tile_number += 1

                tile_id = (
                    f"{waterbody}_{date}"
                    f"_tile_{tile_number:03d}"
                )

                image_output_file = (
                    IMAGE_OUTPUT /
                    f"{tile_id}.npy"
                )

                mask_output_file = (
                    MASK_OUTPUT /
                    f"{tile_id}.npy"
                )

                # -----------------------------------------
                # SAVE
                # -----------------------------------------

                np.save(
                    image_output_file,
                    padded_image
                )

                np.save(
                    mask_output_file,
                    padded_mask
                )

                # -----------------------------------------
                # HAB FLAG
                # -----------------------------------------

                contains_hab = (
                    hab_percent >= MIN_HAB_PERCENT
                )

                records.append({

                    "waterbody": waterbody,

                    "date": date,

                    "tile_id": tile_id,

                    "actual_height": actual_height,

                    "actual_width": actual_width,

                    "valid_pixels": int(
                        valid_count
                    ),

                    "hab_pixels": int(
                        hab_count
                    ),

                    "non_hab_pixels": int(
                        non_hab_count
                    ),

                    "hab_percent": float(
                        hab_percent
                    ),

                    "non_hab_percent": float(
                        non_hab_percent
                    ),

                    "contains_hab": bool(
                        contains_hab
                    )
                })

                total_tiles += 1

                if contains_hab:
                    hab_tiles += 1
                else:
                    non_hab_tiles += 1

        print(
            f"{waterbody:20s} "
            f"{date} -> "
            f"{tile_number} tiles"
        )

    except Exception as e:

        print(
            f"ERROR: "
            f"{waterbody}/{date}"
        )

        print(e)


# =========================================================
# CREATE DATAFRAME
# =========================================================

df = pd.DataFrame(records)

if df.empty:

    print()
    print("ERROR: No tiles were created.")
    raise SystemExit(1)


# =========================================================
# SAVE REPORT
# =========================================================

df.to_csv(
    REPORT_OUTPUT,
    index=False
)


# =========================================================
# OVERALL SUMMARY
# =========================================================

print()
print("=" * 70)
print("STEP 69 COMPLETE")
print("=" * 70)

print(
    f"Total tiles created       : "
    f"{total_tiles}"
)

print(
    f"HAB-containing tiles      : "
    f"{hab_tiles}"
)

print(
    f"Non-HAB-only tiles        : "
    f"{non_hab_tiles}"
)

print(
    f"Images without labels     : "
    f"{skipped_images}"
)

print()
print(
    f"Tile statistics report:"
)

print(REPORT_OUTPUT)


# =========================================================
# WATERBODY SUMMARY
# =========================================================

print()
print("=" * 70)
print("WATERBODY-WISE TILE COUNTS")
print("=" * 70)

summary = (
    df.groupby("waterbody")
    .agg(
        total_tiles=("tile_id", "count"),
        hab_tiles=("contains_hab", "sum")
    )
    .reset_index()
)

summary["non_hab_only_tiles"] = (
    summary["total_tiles"]
    - summary["hab_tiles"]
)

print(
    summary.to_string(index=False)
)


# =========================================================
# PIXEL DISTRIBUTION
# =========================================================

print()
print("=" * 70)
print("HAB PIXEL DISTRIBUTION INSIDE TILES")
print("=" * 70)

print(
    f"Minimum HAB % : "
    f"{df['hab_percent'].min():.2f}%"
)

print(
    f"Maximum HAB % : "
    f"{df['hab_percent'].max():.2f}%"
)

print(
    f"Mean HAB %    : "
    f"{df['hab_percent'].mean():.2f}%"
)

print(
    f"Median HAB %  : "
    f"{df['hab_percent'].median():.2f}%"
)


# =========================================================
# CHECK SAROOR NAGAR
# =========================================================

print()
print("=" * 70)
print("SAROOR NAGAR CHECK")
print("=" * 70)

saroor = df[
    df["waterbody"] == "Saroor_Nagar"
]

print(
    f"Saroor Nagar tiles: "
    f"{len(saroor)}"
)

if len(saroor) == 0:
    print(
        "WARNING: Saroor Nagar still has "
        "no usable tiles."
    )
else:
    print(
        "Saroor Nagar successfully "
        "included in the dataset."
    )

print()