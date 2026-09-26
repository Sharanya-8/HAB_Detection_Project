from pathlib import Path
import rasterio
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
LABEL_ROOT = Path("data/labels/hab_masks/multi_waterbody")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "multilocation_balance_analysis.csv"


# ---------------------------------------------------------
# STORAGE
# ---------------------------------------------------------
records = []


# ---------------------------------------------------------
# PROCESS ALL HAB LABELS
# ---------------------------------------------------------
label_files = sorted(LABEL_ROOT.rglob("*_HAB.tif"))

print("=" * 70)
print("STEP 67 - MULTI-WATERBODY HAB / NON-HAB BALANCE ANALYSIS")
print("=" * 70)

print(f"Label files found: {len(label_files)}")
print()


for label_file in label_files:

    # Waterbody name
    waterbody = label_file.parent.name

    # Date
    date = label_file.stem.replace("_HAB", "")

    try:
        with rasterio.open(label_file) as src:
            mask = src.read(1)

        # Valid water pixels:
        # 0 = Non-HAB
        # 1 = HAB
        # 255 = invalid/outside
        valid = mask != 255

        hab_pixels = np.sum(mask == 1)
        non_hab_pixels = np.sum(mask == 0)
        valid_pixels = np.sum(valid)

        if valid_pixels == 0:
            print(f"WARNING: No valid pixels -> {label_file}")
            continue

        hab_percent = (hab_pixels / valid_pixels) * 100
        non_hab_percent = (non_hab_pixels / valid_pixels) * 100

        records.append({
            "waterbody": waterbody,
            "date": date,
            "hab_pixels": int(hab_pixels),
            "non_hab_pixels": int(non_hab_pixels),
            "valid_pixels": int(valid_pixels),
            "hab_percent": hab_percent,
            "non_hab_percent": non_hab_percent
        })

    except Exception as e:
        print(f"ERROR: {label_file}")
        print(e)


# ---------------------------------------------------------
# CREATE DATAFRAME
# ---------------------------------------------------------
df = pd.DataFrame(records)

if df.empty:
    print("No valid label data found.")
    raise SystemExit


# ---------------------------------------------------------
# SAVE DETAILED RESULTS
# ---------------------------------------------------------
df.to_csv(OUTPUT_CSV, index=False)


# ---------------------------------------------------------
# OVERALL BALANCE
# ---------------------------------------------------------
total_hab = df["hab_pixels"].sum()
total_non_hab = df["non_hab_pixels"].sum()
total_valid = total_hab + total_non_hab

overall_hab_percent = (total_hab / total_valid) * 100
overall_non_hab_percent = (total_non_hab / total_valid) * 100


print("=" * 70)
print("OVERALL DATASET BALANCE")
print("=" * 70)

print(f"Usable images          : {len(df)}")
print(f"Total valid pixels     : {total_valid:,}")
print(f"HAB pixels             : {total_hab:,}")
print(f"Non-HAB pixels         : {total_non_hab:,}")
print(f"HAB percentage         : {overall_hab_percent:.2f}%")
print(f"Non-HAB percentage     : {overall_non_hab_percent:.2f}%")

print()


# ---------------------------------------------------------
# WATERBODY-WISE BALANCE
# ---------------------------------------------------------
print("=" * 70)
print("WATERBODY-WISE BALANCE")
print("=" * 70)

waterbody_summary = []

for waterbody, group in df.groupby("waterbody"):

    hab = group["hab_pixels"].sum()
    non_hab = group["non_hab_pixels"].sum()
    total = hab + non_hab

    hab_pct = (hab / total) * 100
    non_hab_pct = (non_hab / total) * 100

    waterbody_summary.append({
        "waterbody": waterbody,
        "images": len(group),
        "hab_pixels": int(hab),
        "non_hab_pixels": int(non_hab),
        "hab_percent": hab_pct,
        "non_hab_percent": non_hab_pct
    })

    print(
        f"{waterbody:20s} | "
        f"Images: {len(group):3d} | "
        f"HAB: {hab_pct:6.2f}% | "
        f"Non-HAB: {non_hab_pct:6.2f}%"
    )


# ---------------------------------------------------------
# IMAGE-LEVEL STATISTICS
# ---------------------------------------------------------
print()
print("=" * 70)
print("IMAGE-LEVEL HAB PERCENTAGE")
print("=" * 70)

print(f"Minimum HAB % : {df['hab_percent'].min():.2f}%")
print(f"Maximum HAB % : {df['hab_percent'].max():.2f}%")
print(f"Mean HAB %    : {df['hab_percent'].mean():.2f}%")
print(f"Median HAB %  : {df['hab_percent'].median():.2f}%")

print()


# ---------------------------------------------------------
# VERY LOW HAB IMAGES
# ---------------------------------------------------------
low_hab = df[df["hab_percent"] < 5]

print("=" * 70)
print("IMAGES WITH HAB < 5%")
print("=" * 70)

print(f"Count: {len(low_hab)}")

if len(low_hab) > 0:
    print(low_hab[
        ["waterbody", "date", "hab_percent"]
    ].to_string(index=False))

print()


# ---------------------------------------------------------
# HIGH HAB IMAGES
# ---------------------------------------------------------
high_hab = df[df["hab_percent"] > 20]

print("=" * 70)
print("IMAGES WITH HAB > 20%")
print("=" * 70)

print(f"Count: {len(high_hab)}")

if len(high_hab) > 0:
    print(high_hab[
        ["waterbody", "date", "hab_percent"]
    ].to_string(index=False))

print()


# ---------------------------------------------------------
# SAVE WATERBODY SUMMARY
# ---------------------------------------------------------
summary_df = pd.DataFrame(waterbody_summary)

summary_file = OUTPUT_DIR / "multilocation_balance_summary.csv"
summary_df.to_csv(summary_file, index=False)


# ---------------------------------------------------------
# FINAL
# ---------------------------------------------------------
print("=" * 70)
print("STEP 67 COMPLETE")
print("=" * 70)

print(f"Detailed report : {OUTPUT_CSV}")
print(f"Summary report  : {summary_file}")