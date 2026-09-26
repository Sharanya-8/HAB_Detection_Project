from pathlib import Path
import pandas as pd
import numpy as np


REPORT = Path("data/processed/multilocation_tile_statistics.csv")

print("=" * 70)
print("STEP 70 - TILE BALANCE ANALYSIS")
print("=" * 70)

df = pd.read_csv(REPORT)

print(f"Total tiles: {len(df)}")
print()

# ---------------------------------------------------------
# OVERALL PIXEL BALANCE
# ---------------------------------------------------------

total_hab = df["hab_pixels"].sum()
total_non_hab = df["non_hab_pixels"].sum()
total_valid = total_hab + total_non_hab

print("=" * 70)
print("OVERALL TILE PIXEL BALANCE")
print("=" * 70)

print(f"HAB pixels     : {total_hab:,}")
print(f"Non-HAB pixels : {total_non_hab:,}")
print(f"Total valid    : {total_valid:,}")

print(
    f"HAB percentage     : "
    f"{100 * total_hab / total_valid:.2f}%"
)

print(
    f"Non-HAB percentage : "
    f"{100 * total_non_hab / total_valid:.2f}%"
)

print()


# ---------------------------------------------------------
# HAB PERCENTAGE RANGES
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
print("TILE HAB-PERCENTAGE DISTRIBUTION")
print("=" * 70)

range_counts = (
    df["hab_range"]
    .value_counts(sort=False)
)

for category, count in range_counts.items():

    percentage = (
        count / len(df)
    ) * 100

    print(
        f"{str(category):8s} : "
        f"{count:4d} tiles "
        f"({percentage:.2f}%)"
    )

print()


# ---------------------------------------------------------
# WATERBODY-WISE
# ---------------------------------------------------------

print("=" * 70)
print("WATERBODY-WISE TILE BALANCE")
print("=" * 70)

for waterbody, group in df.groupby("waterbody"):

    hab = group["hab_pixels"].sum()
    non_hab = group["non_hab_pixels"].sum()
    total = hab + non_hab

    print(
        f"{waterbody:20s} | "
        f"Tiles: {len(group):3d} | "
        f"HAB: {100 * hab / total:.2f}% | "
        f"Non-HAB: {100 * non_hab / total:.2f}%"
    )

print()


# ---------------------------------------------------------
# TILE HAB STATISTICS
# ---------------------------------------------------------

print("=" * 70)
print("HAB PERCENTAGE STATISTICS")
print("=" * 70)

print(
    f"Minimum : "
    f"{df['hab_percent'].min():.2f}%"
)

print(
    f"Maximum : "
    f"{df['hab_percent'].max():.2f}%"
)

print(
    f"Mean    : "
    f"{df['hab_percent'].mean():.2f}%"
)

print(
    f"Median  : "
    f"{df['hab_percent'].median():.2f}%"
)

print()


# ---------------------------------------------------------
# NON-HAB-ONLY TILES
# ---------------------------------------------------------

non_hab_only = df[
    df["hab_pixels"] == 0
]

print("=" * 70)
print("NON-HAB-ONLY TILES")
print("=" * 70)

print(
    f"Count: {len(non_hab_only)}"
)

if len(non_hab_only) > 0:

    print(
        non_hab_only[
            ["waterbody", "date", "tile_id"]
        ].to_string(index=False)
    )

print()


# ---------------------------------------------------------
# HIGH-HAB TILES
# ---------------------------------------------------------

high_hab = df[
    df["hab_percent"] >= 20
]

print("=" * 70)
print("HIGH-HAB TILES (>=20%)")
print("=" * 70)

print(
    f"Count: {len(high_hab)}"
)

if len(high_hab) > 0:

    print(
        high_hab[
            ["waterbody", "date", "tile_id", "hab_percent"]
        ].to_string(index=False)
    )

print()


# ---------------------------------------------------------
# SAVE ANALYSIS
# ---------------------------------------------------------

output = Path(
    "data/processed/multilocation_tile_balance_analysis.csv"
)

df.to_csv(output, index=False)

print("=" * 70)
print("STEP 70 COMPLETE")
print("=" * 70)

print(f"Analysis saved to:")
print(output)