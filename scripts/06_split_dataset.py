# ============================================================
# STEP 6: DATE-BASED TRAIN / VALIDATION / TEST SPLIT
# HUSSAIN SAGAR MULTI-DATE DATASET
# ============================================================

from pathlib import Path
import shutil


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "tiles"
    / "images"
)

SOURCE_MASK_DIR = (
    PROJECT_ROOT
    / "data"
    / "tiles"
    / "masks"
)

TILES_DIR = (
    PROJECT_ROOT
    / "data"
    / "tiles"
)


# ============================================================
# DATE SPLIT
#
# 10 dates -> TRAIN
# 3 dates  -> VALIDATION
# 2 dates  -> TEST
#
# Each date has 2 tiles.
# Therefore:
# TRAIN = 20 tiles
# VAL   = 6 tiles
# TEST  = 4 tiles
# ============================================================

TRAIN_DATES = [
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
]

VAL_DATES = [
    "2025_02_26",
    "2025_04_04",
    "2025_06_01",
]

TEST_DATES = [
    "2025_11_13",
    "2025_12_28",
]


# ============================================================
# PREPARE SPLIT DIRECTORIES
# ============================================================

def prepare_directories():

    for split in ["train", "val", "test"]:

        image_dir = TILES_DIR / split / "images"
        mask_dir = TILES_DIR / split / "masks"

        image_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        mask_dir.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# CLEAR OLD SPLIT DATA
# ============================================================

def clear_split_directories():

    print()
    print("Removing old split files...")

    for split in ["train", "val", "test"]:

        image_dir = TILES_DIR / split / "images"
        mask_dir = TILES_DIR / split / "masks"

        for file in image_dir.glob("*.npy"):
            file.unlink()

        for file in mask_dir.glob("*.npy"):
            file.unlink()

    print("Old split files removed.")


# ============================================================
# COPY FILES FOR A DATE SPLIT
# ============================================================

def copy_date_files(dates, split):

    source_images = sorted(
        SOURCE_IMAGE_DIR.glob("*.npy")
    )

    source_masks = sorted(
        SOURCE_MASK_DIR.glob("*.npy")
    )

    destination_images = (
        TILES_DIR
        / split
        / "images"
    )

    destination_masks = (
        TILES_DIR
        / split
        / "masks"
    )

    copied = 0

    for date in dates:

        date_files = [
            file
            for file in source_images
            if f"Hussain_Sagar_{date}_" in file.name
        ]

        for image_file in date_files:

            mask_file = (
                SOURCE_MASK_DIR
                / image_file.name
            )

            if not mask_file.exists():

                print(
                    f"WARNING: Matching mask missing: "
                    f"{image_file.name}"
                )

                continue

            shutil.copy2(
                image_file,
                destination_images
                / image_file.name
            )

            shutil.copy2(
                mask_file,
                destination_masks
                / mask_file.name
            )

            copied += 1

    return copied


# ============================================================
# VERIFY SPLIT
# ============================================================

def verify_split(split, expected_dates):

    image_dir = (
        TILES_DIR
        / split
        / "images"
    )

    mask_dir = (
        TILES_DIR
        / split
        / "masks"
    )

    image_files = sorted(
        image_dir.glob("*.npy")
    )

    mask_files = sorted(
        mask_dir.glob("*.npy")
    )

    print()
    print(f"{split.upper()} SET")
    print("-" * 40)

    print(
        f"Image tiles: {len(image_files)}"
    )

    print(
        f"Mask tiles : {len(mask_files)}"
    )

    # Check image/mask count
    if len(image_files) != len(mask_files):

        print(
            "STATUS: FAILED - counts do not match"
        )

        return False

    # Check filenames
    image_names = [
        file.name
        for file in image_files
    ]

    mask_names = [
        file.name
        for file in mask_files
    ]

    if image_names != mask_names:

        print(
            "STATUS: FAILED - filenames do not match"
        )

        return False

    # Check dates
    found_dates = set()

    for file in image_files:

        for date in expected_dates:

            if f"Hussain_Sagar_{date}_" in file.name:
                found_dates.add(date)

    print(
        f"Dates present: "
        f"{sorted(found_dates)}"
    )

    if set(expected_dates) != found_dates:

        print(
            "STATUS: FAILED - expected dates missing"
        )

        return False

    print("STATUS: PASSED")

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("HUSSAIN SAGAR DATE-BASED DATASET SPLIT")
    print("=" * 80)

    prepare_directories()
    clear_split_directories()

    # ========================================================
    # TRAIN
    # ========================================================

    train_count = copy_date_files(
        TRAIN_DATES,
        "train"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    val_count = copy_date_files(
        VAL_DATES,
        "val"
    )

    # ========================================================
    # TEST
    # ========================================================

    test_count = copy_date_files(
        TEST_DATES,
        "test"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("SPLIT COMPLETE")
    print("=" * 80)

    print(
        f"Training dates     : {len(TRAIN_DATES)}"
    )

    print(
        f"Validation dates   : {len(VAL_DATES)}"
    )

    print(
        f"Test dates         : {len(TEST_DATES)}"
    )

    print()

    print(
        f"Training tiles     : {train_count}"
    )

    print(
        f"Validation tiles   : {val_count}"
    )

    print(
        f"Test tiles         : {test_count}"
    )

    # ========================================================
    # VERIFY
    # ========================================================

    train_ok = verify_split(
        "train",
        TRAIN_DATES
    )

    val_ok = verify_split(
        "val",
        VAL_DATES
    )

    test_ok = verify_split(
        "test",
        TEST_DATES
    )

    print()
    print("=" * 80)

    if train_ok and val_ok and test_ok:

        print(
            "STATUS: SUCCESS"
        )

    else:

        print(
            "STATUS: FAILED"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()