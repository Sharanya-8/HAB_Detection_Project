from pathlib import Path
import pandas as pd


INPUT_FILE = Path(
    "data/processed/train_val_split/multilocation_train_val_split.csv"
)

OUTPUT_FILE = Path(
    "data/processed/train_val_split/train_val_balance_analysis.csv"
)


print("=" * 70)
print("STEP 72 - TRAIN / VALIDATION BALANCE ANALYSIS")
print("=" * 70)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print(f"Total tiles : {len(df)}")
print()


# ---------------------------------------------------------
# BASIC SPLIT COUNTS
# ---------------------------------------------------------

print("=" * 70)
print("TRAIN / VALIDATION TILE COUNTS")
print("=" * 70)

print(
    df["split"].value_counts()
)

print()


# ---------------------------------------------------------
# PIXEL BALANCE
# ---------------------------------------------------------

print("=" * 70)
print("PIXEL-LEVEL BALANCE")
print("=" * 70)

for split in ["train", "val"]:

    subset = df[df["split"] == split]

    hab_pixels = subset["hab_pixels"].sum()
    non_hab_pixels = subset["non_hab_pixels"].sum()

    total_pixels = hab_pixels + non_hab_pixels

    hab_percentage = (
        hab_pixels / total_pixels * 100
    )

    non_hab_percentage = (
        non_hab_pixels / total_pixels * 100
    )

    print(f"{split.upper()}")
    print(f"  HAB pixels     : {hab_pixels:,}")
    print(f"  Non-HAB pixels : {non_hab_pixels:,}")
    print(f"  HAB percentage : {hab_percentage:.2f}%")
    print(f"  Non-HAB        : {non_hab_percentage:.2f}%")
    print()


# ---------------------------------------------------------
# HAB PERCENTAGE BINS
# ---------------------------------------------------------

bins = [-0.001, 1, 5, 10, 20, 100]

labels = [
    "0-1%",
    "1-5%",
    "5-10%",
    "10-20%",
    ">20%"
]

df["hab_range"] = pd.cut(
    df["hab_percent"],
    bins=bins,
    labels=labels
)


print("=" * 70)
print("HAB-PERCENTAGE DISTRIBUTION")
print("=" * 70)

for split in ["train", "val"]:

    subset = df[
        df["split"] == split
    ]

    counts = (
        subset["hab_range"]
        .value_counts(sort=False)
    )

    print()
    print(split.upper())

    for category, count in counts.items():

        percentage = (
            count / len(subset) * 100
        )

        print(
            f"  {str(category):8s}: "
            f"{count:4d} tiles "
            f"({percentage:.2f}%)"
        )

print()


# ---------------------------------------------------------
# HAB PERCENTAGE STATISTICS
# ---------------------------------------------------------

print("=" * 70)
print("HAB-PERCENTAGE STATISTICS")
print("=" * 70)

for split in ["train", "val"]:

    subset = df[
        df["split"] == split
    ]

    print(f"\n{split.upper()}")

    print(
        f"  Minimum : "
        f"{subset['hab_percent'].min():.2f}%"
    )

    print(
        f"  Maximum : "
        f"{subset['hab_percent'].max():.2f}%"
    )

    print(
        f"  Mean    : "
        f"{subset['hab_percent'].mean():.2f}%"
    )

    print(
        f"  Median  : "
        f"{subset['hab_percent'].median():.2f}%"
    )

print()


# ---------------------------------------------------------
# WATERBODY-WISE
# ---------------------------------------------------------

print("=" * 70)
print("WATERBODY-WISE TRAIN / VALIDATION BALANCE")
print("=" * 70)

for waterbody in sorted(
    df["waterbody"].unique()
):

    print()
    print(waterbody)

    for split in ["train", "val"]:

        subset = df[
            (df["waterbody"] == waterbody) &
            (df["split"] == split)
        ]

        hab_pixels = subset["hab_pixels"].sum()
        non_hab_pixels = subset["non_hab_pixels"].sum()

        total = hab_pixels + non_hab_pixels

        hab_percentage = (
            hab_pixels / total * 100
            if total > 0 else 0
        )

        print(
            f"  {split:5s}: "
            f"{len(subset):3d} tiles | "
            f"HAB: {hab_percentage:.2f}%"
        )

print()


# ---------------------------------------------------------
# HIGH-HAB TILE COUNTS
# ---------------------------------------------------------

print("=" * 70)
print("HIGH-HAB TILE COUNTS")
print("=" * 70)

for split in ["train", "val"]:

    subset = df[
        df["split"] == split
    ]

    high_hab = subset[
        subset["hab_percent"] >= 20
    ]

    low_hab = subset[
        subset["hab_percent"] <= 1
    ]

    print(
        f"{split.upper():5s}: "
        f"HAB >=20% : {len(high_hab):3d} | "
        f"HAB <=1%  : {len(low_hab):3d}"
    )

print()


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("=" * 70)
print("STEP 72 COMPLETE")
print("=" * 70)

print(
    f"Report saved to:\n{OUTPUT_FILE}"
)