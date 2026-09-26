import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# STEP 87 - TRAINING DATASET WATERBODY BALANCE
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SPLIT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "train_val_split",
    "multilocation_train_val_split.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "dataset_analysis",
    "step87_waterbody_balance"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 70)
print("STEP 87 - TRAINING DATASET WATERBODY BALANCE")
print("=" * 70)

print("\nReading split file:")
print(SPLIT_FILE)


# ============================================================
# READ DATA
# ============================================================

df = pd.read_csv(SPLIT_FILE)

print("\nColumns found:")
print(df.columns.tolist())

print(
    f"\nTotal tile records: {len(df)}"
)


# ============================================================
# FIND IMPORTANT COLUMNS
# ============================================================

# The project split contains waterbody and split information.
# Detect the exact column names safely.

waterbody_col = None
split_col = None

for col in df.columns:

    col_lower = col.lower()

    if (
        waterbody_col is None
        and "waterbody" in col_lower
    ):
        waterbody_col = col

    if (
        split_col is None
        and (
            col_lower == "split"
            or "split" in col_lower
        )
    ):
        split_col = col


if waterbody_col is None:

    raise ValueError(
        "Could not find waterbody column."
    )


if split_col is None:

    raise ValueError(
        "Could not find train/validation split column."
    )


print(
    f"\nWaterbody column: {waterbody_col}"
)

print(
    f"Split column    : {split_col}"
)


# ============================================================
# FIND HAB PERCENT COLUMN
# ============================================================

hab_col = None

for col in df.columns:

    col_lower = col.lower()

    if (
        "hab" in col_lower
        and (
            "percent" in col_lower
            or "%" in col
        )
    ):

        hab_col = col
        break


if hab_col is None:

    print(
        "\nNo HAB percentage column found."
    )

else:

    print(
        f"HAB percentage column: {hab_col}"
    )


# ============================================================
# WATERBODY COUNTS
# ============================================================

waterbody_counts = (
    df.groupby(waterbody_col)
    .size()
    .reset_index(name="tiles")
)

total_tiles = len(df)

waterbody_counts["percentage"] = (
    waterbody_counts["tiles"]
    / total_tiles
    * 100
)


# ============================================================
# TRAIN / VALIDATION COUNTS
# ============================================================

split_counts = (
    df.groupby(
        [waterbody_col, split_col]
    )
    .size()
    .reset_index(name="tiles")
)


# ============================================================
# PRINT WATERBODY BALANCE
# ============================================================

print("\n")
print("=" * 70)
print("WATERBODY TILE DISTRIBUTION")
print("=" * 70)

for _, row in waterbody_counts.iterrows():

    print(
        f"{str(row[waterbody_col]):20s} | "
        f"{int(row['tiles']):4d} tiles | "
        f"{row['percentage']:6.2f}%"
    )


# ============================================================
# PRINT TRAIN / VALIDATION
# ============================================================

print("\n")
print("=" * 70)
print("TRAIN / VALIDATION BY WATERBODY")
print("=" * 70)

print(
    split_counts.to_string(index=False)
)


# ============================================================
# HAB DISTRIBUTION
# ============================================================

hab_summary = None

if hab_col is not None:

    hab_summary = (
        df.groupby(waterbody_col)[hab_col]
        .agg(
            mean="mean",
            median="median",
            minimum="min",
            maximum="max"
        )
        .reset_index()
    )

    print("\n")
    print("=" * 70)
    print("HAB TILE DISTRIBUTION")
    print("=" * 70)

    print(
        hab_summary.to_string(
            index=False
        )
    )


# ============================================================
# SAVE WATERBODY SUMMARY
# ============================================================

waterbody_file = os.path.join(
    OUTPUT_DIR,
    "step87_waterbody_tile_distribution.csv"
)

waterbody_counts.to_csv(
    waterbody_file,
    index=False
)


# ============================================================
# SAVE TRAIN/VALIDATION SUMMARY
# ============================================================

split_file = os.path.join(
    OUTPUT_DIR,
    "step87_train_validation_distribution.csv"
)

split_counts.to_csv(
    split_file,
    index=False
)


# ============================================================
# SAVE HAB SUMMARY
# ============================================================

if hab_summary is not None:

    hab_file = os.path.join(
        OUTPUT_DIR,
        "step87_waterbody_hab_distribution.csv"
    )

    hab_summary.to_csv(
        hab_file,
        index=False
    )


# ============================================================
# WATERBODY TILE GRAPH
# ============================================================

plt.figure(figsize=(11, 6))

plt.bar(
    waterbody_counts[waterbody_col],
    waterbody_counts["tiles"]
)

plt.xlabel("Waterbody")
plt.ylabel("Number of tiles")
plt.title(
    "Training Dataset Tile Distribution by Waterbody"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

tile_graph = os.path.join(
    OUTPUT_DIR,
    "step87_waterbody_tile_distribution.png"
)

plt.savefig(
    tile_graph,
    dpi=200
)

plt.close()


# ============================================================
# TRAIN VS VALIDATION GRAPH
# ============================================================

pivot = split_counts.pivot(
    index=waterbody_col,
    columns=split_col,
    values="tiles"
).fillna(0)

pivot.plot(
    kind="bar",
    figsize=(11, 6)
)

plt.xlabel("Waterbody")
plt.ylabel("Number of tiles")
plt.title(
    "Training vs Validation Tiles by Waterbody"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

split_graph = os.path.join(
    OUTPUT_DIR,
    "step87_train_validation_distribution.png"
)

plt.savefig(
    split_graph,
    dpi=200
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

largest_waterbody = waterbody_counts.loc[
    waterbody_counts["tiles"].idxmax(),
    waterbody_col
]

largest_count = waterbody_counts["tiles"].max()

largest_percentage = waterbody_counts.loc[
    waterbody_counts["tiles"].idxmax(),
    "percentage"
]


print("\n")
print("=" * 70)
print("STEP 87 COMPLETE")
print("=" * 70)

print(
    f"\nLargest tile contributor: "
    f"{largest_waterbody}"
)

print(
    f"Tiles: {int(largest_count)}"
)

print(
    f"Dataset share: "
    f"{largest_percentage:.2f}%"
)

print("\nFiles saved:")

print(
    f"\nWaterbody distribution:\n"
    f"{waterbody_file}"
)

print(
    f"\nTrain/validation distribution:\n"
    f"{split_file}"
)

if hab_summary is not None:

    print(
        f"\nHAB distribution:\n"
        f"{hab_file}"
    )

print(
    f"\nTile graph:\n"
    f"{tile_graph}"
)

print(
    f"\nTrain/validation graph:\n"
    f"{split_graph}"
)

print("\nDo NOT retrain yet.")