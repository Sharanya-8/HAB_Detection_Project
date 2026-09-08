import rasterio
import numpy as np
from pathlib import Path


# --------------------------------------------------
# INPUT FILES
# --------------------------------------------------

image_file = "data/raw/sentinel2/HAB_Sentinel2_2023_2025.tif"

label_file = "data/labels/HAB_clean_labels.tif"


# --------------------------------------------------
# OUTPUT FOLDER
# --------------------------------------------------

output_dir = Path("data/labels/patches")
output_dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# PATCH SIZE
# --------------------------------------------------

PATCH_SIZE = 256


# --------------------------------------------------
# OPEN IMAGE AND LABEL
# --------------------------------------------------

with rasterio.open(image_file) as image_src:

    image_height = image_src.height
    image_width = image_src.width

    print("Image width:", image_width)
    print("Image height:", image_height)
    print("Patch size:", PATCH_SIZE)


    with rasterio.open(label_file) as label_src:

        # Check dimensions match
        if (
            image_width != label_src.width
            or image_height != label_src.height
        ):
            raise ValueError(
                "Image and label dimensions do not match!"
            )


        # Read all image bands
        image = image_src.read()

        # Read label
        label = label_src.read(1)


        patch_count = 0


        # --------------------------------------------------
        # SAME PATCH LOOP AS 05_create_patches.py
        # --------------------------------------------------

        for row in range(
            0,
            image_height - PATCH_SIZE + 1,
            PATCH_SIZE
        ):

            for col in range(
                0,
                image_width - PATCH_SIZE + 1,
                PATCH_SIZE
            ):

                # Extract image patch
                image_patch = image[
                    :,
                    row:row + PATCH_SIZE,
                    col:col + PATCH_SIZE
                ]


                # --------------------------------------------------
                # SAME VALIDITY TEST AS ORIGINAL PATCH SCRIPT
                # --------------------------------------------------

                valid_pixels = np.isfinite(image_patch).sum()

                total_pixels = image_patch.size

                valid_ratio = valid_pixels / total_pixels


                # Skip exactly the same patches
                if valid_ratio < 0.5:
                    continue


                # --------------------------------------------------
                # EXTRACT MATCHING LABEL PATCH
                # --------------------------------------------------

                label_patch = label[
                    row:row + PATCH_SIZE,
                    col:col + PATCH_SIZE
                ]


                patch_count += 1


                # --------------------------------------------------
                # SAVE WITH SAME PATCH NUMBER
                # --------------------------------------------------

                output_file = (
                    output_dir /
                    f"patch_{patch_count:04d}.npy"
                )


                np.save(
                    output_file,
                    label_patch
                )


                # --------------------------------------------------
                # STATISTICS FOR THIS PATCH
                # --------------------------------------------------

                hab_pixels = np.sum(label_patch == 1)

                valid_label_pixels = np.sum(
                    label_patch != 255
                )


                print(
                    f"Saved: {output_file} | "
                    f"HAB pixels: {hab_pixels} | "
                    f"Valid label pixels: {valid_label_pixels}"
                )


print()
print("Label patch creation completed.")
print("Total label patches:", patch_count)
print()
print("Expected image patches: 62")
print("Expected label patches: 62")


if patch_count == 62:

    print()
    print("SUCCESS: Image and label patch counts match!")

else:

    print()
    print("WARNING: Patch counts do not match!")