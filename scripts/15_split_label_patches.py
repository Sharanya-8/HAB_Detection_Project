import shutil
from pathlib import Path


# --------------------------------------------------
# DIRECTORIES
# --------------------------------------------------

image_base = Path("data/patches")

label_base = Path("data/labels/patches")

label_output_base = Path("data/labels")


# --------------------------------------------------
# DATASET SPLITS
# --------------------------------------------------

splits = [
    "train",
    "val",
    "test"
]


# --------------------------------------------------
# CREATE OUTPUT FOLDERS
# --------------------------------------------------

for split in splits:

    output_dir = label_output_base / split

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


# --------------------------------------------------
# COPY MATCHING LABELS
# --------------------------------------------------

total_copied = 0


for split in splits:

    image_dir = image_base / split

    label_dir = label_output_base / split

    print()
    print("Processing:", split)
    print("----------------------------")


    image_files = sorted(
        image_dir.glob("*.npy")
    )


    copied = 0


    for image_file in image_files:

        # Same filename as image
        label_file = label_base / image_file.name


        if not label_file.exists():

            print(
                "ERROR: Missing label:",
                label_file
            )

            continue


        destination = label_dir / label_file.name


        shutil.copy2(
            label_file,
            destination
        )


        copied += 1
        total_copied += 1


        print(
            f"Copied: {label_file.name}"
        )


    print()
    print(
        f"{split} label patches:",
        copied
    )


# --------------------------------------------------
# FINAL CHECK
# --------------------------------------------------

print()
print("============================")
print("LABEL DATASET SPLIT COMPLETE")
print("============================")

print()
print("Total label patches copied:", total_copied)

print()
print("Expected:")
print("Train: 43")
print("Validation: 9")
print("Test: 10")


if total_copied == 62:

    print()
    print("SUCCESS: All 62 label patches were split!")

else:

    print()
    print("WARNING: Expected 62 label patches.")