import os
import glob
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio


# ============================================================
# STEP 84 - TRAINING VS SHAMIRPET SPECTRAL COMPARISON
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAINING_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "sentinel2",
    "multi_waterbody"
)

SHAMIRPET_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "sentinel2",
    "shamirpet_test"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "spectral_comparison"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 14 INPUT CHANNELS
# ------------------------------------------------------------

BAND_NAMES = [
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


# ------------------------------------------------------------
# FUNCTION TO READ TIFF STATISTICS
# ------------------------------------------------------------

def calculate_statistics(tif_path, dataset_name):

    values = []

    try:
        with rasterio.open(tif_path) as src:

            data = src.read().astype(np.float32)

            # data shape:
            # (14, height, width)

            for band_index in range(data.shape[0]):

                band = data[band_index]

                # Remove invalid values
                valid = band[np.isfinite(band)]

                # Remove extreme invalid values
                valid = valid[np.abs(valid) < 1e6]

                if len(valid) == 0:
                    continue

                values.append({
                    "Dataset": dataset_name,
                    "Image": os.path.basename(tif_path),
                    "Band": BAND_NAMES[band_index],
                    "Mean": np.mean(valid),
                    "Std": np.std(valid),
                    "Min": np.min(valid),
                    "Max": np.max(valid),
                    "Median": np.median(valid)
                })

    except Exception as e:

        print(f"ERROR reading {tif_path}")
        print(e)

    return values


# ============================================================
# FIND TRAINING IMAGES
# ============================================================

print("=" * 70)
print("STEP 84 - TRAINING VS SHAMIRPET SPECTRAL COMPARISON")
print("=" * 70)

print("\nSearching for training images...")

training_files = glob.glob(
    os.path.join(TRAINING_DIR, "**", "*.tif"),
    recursive=True
)

shamirpet_files = glob.glob(
    os.path.join(SHAMIRPET_DIR, "*.tif")
)

print(f"Training TIFF files found : {len(training_files)}")
print(f"Shamirpet TIFF files found: {len(shamirpet_files)}")


# ============================================================
# LIMIT TRAINING SAMPLE
# ============================================================

# We don't need to read every training image.
# Use up to 100 images for a representative comparison.

if len(training_files) > 100:

    indices = np.linspace(
        0,
        len(training_files) - 1,
        100,
        dtype=int
    )

    training_sample = [
        training_files[i]
        for i in indices
    ]

else:

    training_sample = training_files


# Use all 100 Shamirpet test images
shamirpet_sample = shamirpet_files


print(f"\nTraining images used for comparison : {len(training_sample)}")
print(f"Shamirpet images used               : {len(shamirpet_sample)}")


# ============================================================
# CALCULATE STATISTICS
# ============================================================

all_statistics = []


print("\nReading training images...")

for i, tif_path in enumerate(training_sample, start=1):

    print(
        f"Training image {i}/{len(training_sample)}",
        end="\r"
    )

    all_statistics.extend(
        calculate_statistics(
            tif_path,
            "Training"
        )
    )


print("\n\nReading Shamirpet images...")

for i, tif_path in enumerate(shamirpet_sample, start=1):

    print(
        f"Shamirpet image {i}/{len(shamirpet_sample)}",
        end="\r"
    )

    all_statistics.extend(
        calculate_statistics(
            tif_path,
            "Shamirpet"
        )
    )


# ============================================================
# SAVE IMAGE-LEVEL STATISTICS
# ============================================================

statistics_df = pd.DataFrame(all_statistics)

statistics_path = os.path.join(
    OUTPUT_DIR,
    "step84_image_level_statistics.csv"
)

statistics_df.to_csv(
    statistics_path,
    index=False
)


# ============================================================
# GROUP SUMMARY
# ============================================================

summary = (
    statistics_df
    .groupby(["Dataset", "Band"])
    .agg(
        Mean=("Mean", "mean"),
        Std=("Mean", "std"),
        Minimum=("Min", "min"),
        Maximum=("Max", "max"),
        Median=("Median", "mean")
    )
    .reset_index()
)


summary_path = os.path.join(
    OUTPUT_DIR,
    "step84_training_vs_shamirpet_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("SPECTRAL SUMMARY")
print("=" * 70)

for band in BAND_NAMES:

    training_values = statistics_df[
        (statistics_df["Dataset"] == "Training") &
        (statistics_df["Band"] == band)
    ]["Mean"]

    shamirpet_values = statistics_df[
        (statistics_df["Dataset"] == "Shamirpet") &
        (statistics_df["Band"] == band)
    ]["Mean"]

    training_mean = training_values.mean()
    shamirpet_mean = shamirpet_values.mean()

    difference = shamirpet_mean - training_mean

    if training_mean != 0:

        percentage_difference = (
            abs(difference) /
            abs(training_mean)
        ) * 100

    else:

        percentage_difference = 0

    print(
        f"{band:6s} | "
        f"Training Mean: {training_mean:10.5f} | "
        f"Shamirpet Mean: {shamirpet_mean:10.5f} | "
        f"Difference: {difference:10.5f} | "
        f"Diff %: {percentage_difference:8.2f}%"
    )


# ============================================================
# CREATE MEAN COMPARISON GRAPH
# ============================================================

training_means = []
shamirpet_means = []

for band in BAND_NAMES:

    training_means.append(
        statistics_df[
            (statistics_df["Dataset"] == "Training") &
            (statistics_df["Band"] == band)
        ]["Mean"].mean()
    )

    shamirpet_means.append(
        statistics_df[
            (statistics_df["Dataset"] == "Shamirpet") &
            (statistics_df["Band"] == band)
        ]["Mean"].mean()
    )


x = np.arange(len(BAND_NAMES))
width = 0.38

plt.figure(figsize=(16, 7))

plt.bar(
    x - width / 2,
    training_means,
    width,
    label="Training Waterbodies"
)

plt.bar(
    x + width / 2,
    shamirpet_means,
    width,
    label="Shamirpet"
)

plt.xticks(
    x,
    BAND_NAMES,
    rotation=45
)

plt.ylabel("Mean Value")
plt.xlabel("Input Channel")
plt.title(
    "Training Waterbodies vs Shamirpet - Mean Spectral Values"
)

plt.legend()
plt.tight_layout()

mean_graph_path = os.path.join(
    OUTPUT_DIR,
    "step84_mean_comparison.png"
)

plt.savefig(
    mean_graph_path,
    dpi=200
)

plt.close()


# ============================================================
# CREATE STANDARD DEVIATION COMPARISON
# ============================================================

training_std = []
shamirpet_std = []

for band in BAND_NAMES:

    training_std.append(
        statistics_df[
            (statistics_df["Dataset"] == "Training") &
            (statistics_df["Band"] == band)
        ]["Mean"].std()
    )

    shamirpet_std.append(
        statistics_df[
            (statistics_df["Dataset"] == "Shamirpet") &
            (statistics_df["Band"] == band)
        ]["Mean"].std()
    )


plt.figure(figsize=(16, 7))

plt.plot(
    BAND_NAMES,
    training_std,
    marker="o",
    label="Training Waterbodies"
)

plt.plot(
    BAND_NAMES,
    shamirpet_std,
    marker="o",
    label="Shamirpet"
)

plt.xticks(rotation=45)

plt.ylabel("Standard Deviation")
plt.xlabel("Input Channel")

plt.title(
    "Training Waterbodies vs Shamirpet - Spectral Variation"
)

plt.legend()
plt.tight_layout()

std_graph_path = os.path.join(
    OUTPUT_DIR,
    "step84_standard_deviation_comparison.png"
)

plt.savefig(
    std_graph_path,
    dpi=200
)

plt.close()


# ============================================================
# CALCULATE DISTRIBUTION DIFFERENCE
# ============================================================

comparison_rows = []

for band in BAND_NAMES:

    train_mean = statistics_df[
        (statistics_df["Dataset"] == "Training") &
        (statistics_df["Band"] == band)
    ]["Mean"].mean()

    shamirpet_mean = statistics_df[
        (statistics_df["Dataset"] == "Shamirpet") &
        (statistics_df["Band"] == band)
    ]["Mean"].mean()

    train_std_value = statistics_df[
        (statistics_df["Dataset"] == "Training") &
        (statistics_df["Band"] == band)
    ]["Mean"].std()

    shamirpet_std_value = statistics_df[
        (statistics_df["Dataset"] == "Shamirpet") &
        (statistics_df["Band"] == band)
    ]["Mean"].std()

    comparison_rows.append({
        "Band": band,
        "Training_Mean": train_mean,
        "Shamirpet_Mean": shamirpet_mean,
        "Mean_Difference": shamirpet_mean - train_mean,
        "Training_Std": train_std_value,
        "Shamirpet_Std": shamirpet_std_value
    })


comparison_df = pd.DataFrame(comparison_rows)

comparison_path = os.path.join(
    OUTPUT_DIR,
    "step84_channel_comparison.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 70)
print("STEP 84 COMPLETE")
print("=" * 70)

print("\nFiles saved:")

print(
    f"\nImage-level statistics:\n"
    f"{statistics_path}"
)

print(
    f"\nTraining vs Shamirpet summary:\n"
    f"{summary_path}"
)

print(
    f"\nChannel comparison:\n"
    f"{comparison_path}"
)

print(
    f"\nMean comparison graph:\n"
    f"{mean_graph_path}"
)

print(
    f"\nStandard deviation graph:\n"
    f"{std_graph_path}"
)

print("\nDo NOT retrain the model yet.")
print("Send me the terminal output after this script finishes.")