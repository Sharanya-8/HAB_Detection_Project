import numpy as np
from pathlib import Path


# --------------------------------------------------
# TRAINING PATCH DIRECTORY
# --------------------------------------------------

patch_dir = Path("data/patches/train")


# --------------------------------------------------
# CHANNEL NAMES
# --------------------------------------------------

channel_names = [
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


# --------------------------------------------------
# COLLECT PATCH FILES
# --------------------------------------------------

patch_files = sorted(
    patch_dir.glob("*.npy")
)


print("PATCH CHANNEL ANALYSIS")
print("=======================")

print()
print("Training patches:", len(patch_files))
print()


# --------------------------------------------------
# STORAGE FOR CHANNEL VALUES
# --------------------------------------------------

channel_values = [
    [] for _ in range(14)
]


# --------------------------------------------------
# READ TRAINING PATCHES
# --------------------------------------------------

for patch_file in patch_files:

    patch = np.load(patch_file)

    if patch.shape != (14, 256, 256):

        print(
            "WARNING:",
            patch_file.name,
            "has shape",
            patch.shape
        )

        continue


    for channel in range(14):

        values = patch[channel]

        valid = values[np.isfinite(values)]

        # Sample values to keep memory usage low
        if len(valid) > 10000:

            valid = np.random.choice(
                valid,
                10000,
                replace=False
            )

        channel_values[channel].extend(valid)


# --------------------------------------------------
# PRINT STATISTICS
# --------------------------------------------------

print(
    f"{'Channel':<10}"
    f"{'Minimum':>15}"
    f"{'Maximum':>15}"
    f"{'Mean':>15}"
    f"{'Median':>15}"
)

print("-" * 70)


for i, name in enumerate(channel_names):

    values = np.array(
        channel_values[i],
        dtype=np.float32
    )

    print(
        f"{name:<10}"
        f"{np.min(values):>15.6f}"
        f"{np.max(values):>15.6f}"
        f"{np.mean(values):>15.6f}"
        f"{np.median(values):>15.6f}"
    )


print()
print("Channel analysis completed.")