from pathlib import Path
import numpy as np
import pandas as pd


TILES_DIR = Path("data/tiles")

IMAGE_DIR = TILES_DIR / "images"
MASK_DIR = TILES_DIR / "masks"

SPLIT_FILE = Path(
    "data/processed/train_val_split/multilocation_train_val_split.csv"
)


print("=" * 70)
print("STEP 73 - VALIDATE TRAINING TILES")
print("=" * 70)


# ---------------------------------------------------------
# LOAD SPLIT INFORMATION
# ---------------------------------------------------------

df = pd.read_csv(SPLIT_FILE)

print(f"Tiles in split report: {len(df)}")
print()


# ---------------------------------------------------------
# FIND IMAGE/MASK FILES
# ---------------------------------------------------------

image_files = list(IMAGE_DIR.rglob("*.npy"))
mask_files = list(MASK_DIR.rglob("*.npy"))

print(f"Image files found: {len(image_files)}")
print(f"Mask files found : {len(mask_files)}")
print()


# ---------------------------------------------------------
# CREATE LOOKUP
# ---------------------------------------------------------

image_lookup = {
    f.stem: f
    for f in image_files
}

mask_lookup = {
    f.stem: f
    for f in mask_files
}


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

valid = 0
missing_images = []
missing_masks = []
wrong_shape = []
wrong_channels = []
nan_inf_images = []
invalid_masks = []

train_count = 0
val_count = 0


for _, row in df.iterrows():

    tile_id = str(row["tile_id"])
    split = row["split"]

    # -----------------------------------------------------
    # FIND IMAGE
    # -----------------------------------------------------

    image_file = image_lookup.get(tile_id)

    if image_file is None:
        missing_images.append(tile_id)
        continue

    # -----------------------------------------------------
    # FIND MASK
    # -----------------------------------------------------

    mask_file = mask_lookup.get(tile_id)

    if mask_file is None:
        missing_masks.append(tile_id)
        continue

    # -----------------------------------------------------
    # LOAD
    # -----------------------------------------------------

    try:

        image = np.load(image_file)
        mask = np.load(mask_file)

    except Exception as e:

        wrong_shape.append(
            f"{tile_id}: load error - {e}"
        )

        continue


    # -----------------------------------------------------
    # IMAGE SHAPE
    # -----------------------------------------------------

    if image.shape != (14, 256, 256):

        wrong_shape.append(
            f"{tile_id}: image shape={image.shape}"
        )

        continue


    # -----------------------------------------------------
    # MASK SHAPE
    # -----------------------------------------------------

    if mask.shape != (256, 256):

        wrong_shape.append(
            f"{tile_id}: mask shape={mask.shape}"
        )

        continue


    # -----------------------------------------------------
    # CHANNEL COUNT
    # -----------------------------------------------------

    if image.shape[0] != 14:

        wrong_channels.append(
            f"{tile_id}: channels={image.shape[0]}"
        )

        continue


    # -----------------------------------------------------
    # NAN / INF
    # -----------------------------------------------------

    if not np.isfinite(image).all():

        nan_inf_images.append(tile_id)

        continue


    # -----------------------------------------------------
    # MASK VALUES
    # -----------------------------------------------------

    unique_values = set(
        np.unique(mask).tolist()
    )

    allowed_values = {0, 1, 255}

    if not unique_values.issubset(
        allowed_values
    ):

        invalid_masks.append(
            f"{tile_id}: values={sorted(unique_values)}"
        )

        continue


    # -----------------------------------------------------
    # VALID TILE
    # -----------------------------------------------------

    valid += 1

    if split == "train":
        train_count += 1

    elif split == "val":
        val_count += 1


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

print("=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

print(f"Valid tiles      : {valid}")
print(f"Training tiles   : {train_count}")
print(f"Validation tiles : {val_count}")

print()

print(f"Missing images   : {len(missing_images)}")
print(f"Missing masks    : {len(missing_masks)}")
print(f"Wrong shapes     : {len(wrong_shape)}")
print(f"Wrong channels   : {len(wrong_channels)}")
print(f"NaN/Inf images   : {len(nan_inf_images)}")
print(f"Invalid masks    : {len(invalid_masks)}")

print()


# ---------------------------------------------------------
# SHOW PROBLEMS
# ---------------------------------------------------------

if missing_images:

    print("MISSING IMAGES:")
    for x in missing_images[:20]:
        print(x)
    print()


if missing_masks:

    print("MISSING MASKS:")
    for x in missing_masks[:20]:
        print(x)
    print()


if wrong_shape:

    print("WRONG SHAPES:")
    for x in wrong_shape[:20]:
        print(x)
    print()


if wrong_channels:

    print("WRONG CHANNELS:")
    for x in wrong_channels[:20]:
        print(x)
    print()


if nan_inf_images:

    print("NAN/INF IMAGES:")
    for x in nan_inf_images[:20]:
        print(x)
    print()


if invalid_masks:

    print("INVALID MASKS:")
    for x in invalid_masks[:20]:
        print(x)
    print()


# ---------------------------------------------------------
# FINAL STATUS
# ---------------------------------------------------------

if (
    valid == len(df)
    and
    len(missing_images) == 0
    and
    len(missing_masks) == 0
    and
    len(wrong_shape) == 0
    and
    len(wrong_channels) == 0
    and
    len(nan_inf_images) == 0
    and
    len(invalid_masks) == 0
):

    print("=" * 70)
    print("ALL TRAINING TILES ARE VALID")
    print("=" * 70)

else:

    print("=" * 70)
    print("SOME TILES NEED ATTENTION")
    print("=" * 70)


# ---------------------------------------------------------
# SAVE REPORT
# ---------------------------------------------------------

report = pd.DataFrame({
    "total_tiles": [len(df)],
    "valid_tiles": [valid],
    "missing_images": [len(missing_images)],
    "missing_masks": [len(missing_masks)],
    "wrong_shapes": [len(wrong_shape)],
    "wrong_channels": [len(wrong_channels)],
    "nan_inf_images": [len(nan_inf_images)],
    "invalid_masks": [len(invalid_masks)]
})

output = Path(
    "data/processed/train_val_split/"
    "tile_validation_summary.csv"
)

report.to_csv(
    output,
    index=False
)

print()
print(f"Validation report saved to:")
print(output)

print()
print("STEP 73 COMPLETE")