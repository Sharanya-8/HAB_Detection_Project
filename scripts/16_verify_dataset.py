# ============================================================
# STEP 7: VERIFY FINAL HUSSAIN SAGAR DATASET
# ============================================================

from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TILES_DIR = (
    PROJECT_ROOT
    / "data"
    / "tiles"
)


SPLITS = ["train", "val", "test"]


def verify_split(split):

    image_dir = TILES_DIR / split / "images"
    mask_dir = TILES_DIR / split / "masks"

    image_files = sorted(image_dir.glob("*.npy"))
    mask_files = sorted(mask_dir.glob("*.npy"))

    print()
    print("=" * 70)
    print(f"{split.upper()} DATASET")
    print("=" * 70)

    print(f"Image tiles: {len(image_files)}")
    print(f"Mask tiles : {len(mask_files)}")

    if len(image_files) != len(mask_files):
        print("STATUS: FAILED - image/mask counts differ")
        return False

    if len(image_files) == 0:
        print("STATUS: FAILED - no tiles found")
        return False

    image_names = [f.name for f in image_files]
    mask_names = [f.name for f in mask_files]

    if image_names != mask_names:
        print("STATUS: FAILED - filenames do not match")
        return False

    total_hab = 0
    total_non_hab = 0
    total_ignore = 0

    expected_image_shape = (14, 256, 256)
    expected_mask_shape = (256, 256)

    for image_file, mask_file in zip(
        image_files,
        mask_files
    ):

        image = np.load(image_file)
        mask = np.load(mask_file)

        # ----------------------------------------------------
        # Shape checks
        # ----------------------------------------------------

        if image.shape != expected_image_shape:

            print(
                f"FAILED image shape: "
                f"{image_file.name} -> {image.shape}"
            )

            return False

        if mask.shape != expected_mask_shape:

            print(
                f"FAILED mask shape: "
                f"{mask_file.name} -> {mask.shape}"
            )

            return False

        # ----------------------------------------------------
        # Image dtype
        # ----------------------------------------------------

        if image.dtype != np.float32:

            print(
                f"WARNING: {image_file.name} "
                f"dtype is {image.dtype}"
            )

        # ----------------------------------------------------
        # Check NaN / Inf
        # ----------------------------------------------------

        if not np.isfinite(image).all():

            print(
                f"FAILED: invalid values in "
                f"{image_file.name}"
            )

            return False

        # ----------------------------------------------------
        # Mask values
        # ----------------------------------------------------

        unique_values = set(
            np.unique(mask).tolist()
        )

        allowed_values = {0, 1, 255}

        if not unique_values.issubset(
            allowed_values
        ):

            print(
                f"FAILED mask values: "
                f"{mask_file.name} -> "
                f"{sorted(unique_values)}"
            )

            return False

        # ----------------------------------------------------
        # Pixel statistics
        # ----------------------------------------------------

        total_hab += np.count_nonzero(
            mask == 1
        )

        total_non_hab += np.count_nonzero(
            mask == 0
        )

        total_ignore += np.count_nonzero(
            mask == 255
        )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    total_valid = (
        total_hab
        + total_non_hab
    )

    if total_valid > 0:

        hab_percentage = (
            total_hab
            / total_valid
            * 100
        )

    else:

        hab_percentage = 0

    print()
    print("PIXEL STATISTICS")
    print("-" * 40)

    print(
        f"HAB pixels       : {total_hab:,}"
    )

    print(
        f"Non-HAB pixels   : {total_non_hab:,}"
    )

    print(
        f"Ignore pixels    : {total_ignore:,}"
    )

    print(
        f"Valid pixels     : {total_valid:,}"
    )

    print(
        f"HAB percentage   : "
        f"{hab_percentage:.2f}%"
    )

    print()
    print("STATUS: PASSED")

    return True


def main():

    print("=" * 70)
    print("FINAL HUSSAIN SAGAR DATASET VERIFICATION")
    print("=" * 70)

    results = []

    for split in SPLITS:

        results.append(
            verify_split(split)
        )

    print()
    print("=" * 70)
    print("FINAL VERIFICATION")
    print("=" * 70)

    if all(results):

        print("TRAIN: PASSED")
        print("VAL  : PASSED")
        print("TEST : PASSED")
        print()
        print("STATUS: SUCCESS")

    else:

        print("STATUS: FAILED")


if __name__ == "__main__":
    main()