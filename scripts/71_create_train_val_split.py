from pathlib import Path
import pandas as pd
import numpy as np


INPUT_REPORT = Path(
    "data/processed/multilocation_tile_statistics.csv"
)

OUTPUT_DIR = Path(
    "data/processed/train_val_split"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("STEP 71 - TRAIN / VALIDATION SPLIT")
print("=" * 70)


# ---------------------------------------------------------
# LOAD TILE INFORMATION
# ---------------------------------------------------------

df = pd.read_csv(INPUT_REPORT)

print(f"Total tiles: {len(df)}")
print()


# ---------------------------------------------------------
# GET UNIQUE WATERBODY-DATE GROUPS
# ---------------------------------------------------------

groups = (
    df[["waterbody", "date"]]
    .drop_duplicates()
)

groups["date"] = pd.to_datetime(groups["date"])

print(f"Unique waterbody-date groups: {len(groups)}")
print()


# ---------------------------------------------------------
# SPLIT EACH WATERBODY BY DATE
# ---------------------------------------------------------

train_groups = []
val_groups = []

random_seed = 42

for waterbody in sorted(groups["waterbody"].unique()):

    wb = groups[
        groups["waterbody"] == waterbody
    ].sort_values("date")

    # Deterministic shuffle
    wb = wb.sample(
        frac=1,
        random_state=random_seed
    ).reset_index(drop=True)

    n = len(wb)

    # 80% train
    n_train = int(n * 0.80)

    train = wb.iloc[:n_train]
    val = wb.iloc[n_train:]

    train_groups.append(train)
    val_groups.append(val)


train_groups = pd.concat(
    train_groups,
    ignore_index=True
)

val_groups = pd.concat(
    val_groups,
    ignore_index=True
)


# ---------------------------------------------------------
# ASSIGN SPLITS TO TILES
# ---------------------------------------------------------

df["date"] = pd.to_datetime(df["date"])

train_keys = set(
    zip(
        train_groups["waterbody"],
        train_groups["date"]
    )
)

val_keys = set(
    zip(
        val_groups["waterbody"],
        val_groups["date"]
    )
)


def assign_split(row):

    key = (
        row["waterbody"],
        row["date"]
    )

    if key in train_keys:
        return "train"

    if key in val_keys:
        return "val"

    return "unknown"


df["split"] = df.apply(
    assign_split,
    axis=1
)


# ---------------------------------------------------------
# CHECK
# ---------------------------------------------------------

print("=" * 70)
print("SPLIT SUMMARY")
print("=" * 70)

print(
    df["split"].value_counts()
)

print()


# ---------------------------------------------------------
# WATERBODY-WISE SUMMARY
# ---------------------------------------------------------

print("=" * 70)
print("WATERBODY-WISE SPLIT")
print("=" * 70)

summary = (
    df.groupby(
        ["waterbody", "split"]
    )
    .agg(
        tiles=("tile_id", "count"),
        dates=("date", "nunique")
    )
    .reset_index()
)

print(
    summary.to_string(index=False)
)

print()


# ---------------------------------------------------------
# CHECK FOR DATE LEAKAGE
# ---------------------------------------------------------

print("=" * 70)
print("DATE LEAKAGE CHECK")
print("=" * 70)

leakage_found = False

for waterbody in sorted(df["waterbody"].unique()):

    train_dates = set(
        df[
            (df["waterbody"] == waterbody) &
            (df["split"] == "train")
        ]["date"]
    )

    val_dates = set(
        df[
            (df["waterbody"] == waterbody) &
            (df["split"] == "val")
        ]["date"]
    )

    overlap = train_dates.intersection(val_dates)

    print(
        f"{waterbody}: "
        f"train dates={len(train_dates)}, "
        f"validation dates={len(val_dates)}, "
        f"overlap={len(overlap)}"
    )

    if overlap:
        leakage_found = True


print()

if leakage_found:
    print("WARNING: DATE LEAKAGE DETECTED")
else:
    print("No date leakage detected.")


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

output_file = (
    OUTPUT_DIR /
    "multilocation_train_val_split.csv"
)

df.to_csv(
    output_file,
    index=False
)


train_file = (
    OUTPUT_DIR /
    "train_tiles.csv"
)

val_file = (
    OUTPUT_DIR /
    "val_tiles.csv"
)

df[
    df["split"] == "train"
].to_csv(
    train_file,
    index=False
)

df[
    df["split"] == "val"
].to_csv(
    val_file,
    index=False
)


print()
print("=" * 70)
print("STEP 71 COMPLETE")
print("=" * 70)

print(f"Full split report:")
print(output_file)

print()
print(f"Training tiles:")
print(len(df[df["split"] == "train"]))

print(f"Validation tiles:")
print(len(df[df["split"] == "val"]))