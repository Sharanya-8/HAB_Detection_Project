from pathlib import Path
import re
import numpy as np
import pandas as pd
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "sentinel2" / "multi_waterbody"
LABEL_ROOT = PROJECT_ROOT / "data" / "labels" / "hab_masks" / "multi_waterbody"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "multiwaterbody_temporal_dataset.csv"
SKIPPED_FILE = PROJECT_ROOT / "data" / "processed" / "multiwaterbody_temporal_skipped.csv"

TRAINING_WATERBODIES = [
    "Hussain_Sagar", "Saroor_Nagar", "Osman_Sagar", "Himayat_Sagar"
]

def extract_date(name):
    m = re.search(r"(20\d{2})[-_](\d{2})[-_](\d{2})", name)
    if not m:
        return None
    return pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))

def find_label(waterbody, date):
    d = LABEL_ROOT / waterbody
    candidates = [
        d / f"{date:%Y-%m-%d}_HAB.tif",
        d / f"{date:%Y_%m_%d}_HAB.tif",
        d / f"{waterbody}_{date:%Y-%m-%d}_HAB.tif",
        d / f"{waterbody}_{date:%Y_%m_%d}_HAB.tif",
    ]
    for p in candidates:
        if p.exists():
            return p
    if d.exists():
        for p in d.glob("*.tif"):
            if date.strftime("%Y-%m-%d") in p.name or date.strftime("%Y_%m_%d") in p.name:
                return p
    return None

def stats(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan, np.nan, np.nan
    return float(a.mean()), float(np.median(a)), float(a.max())

def process(image_path, label_path):
    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float32)
    with rasterio.open(label_path) as src:
        label = src.read(1)

    if image.shape[0] != 14:
        raise ValueError(f"Expected 14 bands, found {image.shape[0]}")
    if label.shape != image.shape[1:]:
        raise ValueError("Image/label shape mismatch")

    valid = (
        (label != 255)
        & np.isfinite(image[10])
        & np.isfinite(image[11])
        & np.isfinite(image[12])
        & np.isfinite(image[13])
    )
    if not valid.any():
        raise ValueError("No valid water pixels")

    ndwi_m, ndwi_med, _ = stats(image[10][valid])
    mndwi_m, mndwi_med, _ = stats(image[11][valid])
    ndci_m, ndci_med, ndci_max = stats(image[12][valid])
    fai_m, fai_med, fai_max = stats(image[13][valid])

    water_n = int(valid.sum())
    hab_n = int((label[valid] == 1).sum())

    return {
        "valid_water_pixels": water_n,
        "hab_pixels": hab_n,
        "hab_percentage": 100.0 * hab_n / water_n,
        "ndci_mean": ndci_m,
        "ndci_median": ndci_med,
        "ndci_max": ndci_max,
        "ndwi_mean": ndwi_m,
        "ndwi_median": ndwi_med,
        "mndwi_mean": mndwi_m,
        "mndwi_median": mndwi_med,
        "fai_mean": fai_m,
        "fai_median": fai_med,
        "fai_max": fai_max,
    }

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []

    for waterbody in TRAINING_WATERBODIES:
        image_dir = IMAGE_ROOT / waterbody
        if not image_dir.exists():
            skipped.append({"waterbody": waterbody, "reason": "image directory missing"})
            continue

        for image_path in sorted(image_dir.glob("*.tif")):
            date = extract_date(image_path.name)
            if date is None:
                skipped.append({"waterbody": waterbody, "image": str(image_path), "reason": "date not found"})
                continue

            label_path = find_label(waterbody, date)
            if label_path is None:
                skipped.append({"waterbody": waterbody, "date": str(date.date()), "reason": "label not found"})
                continue

            try:
                rows.append({
                    "waterbody": waterbody,
                    "date": date,
                    **process(image_path, label_path),
                    "image_path": str(image_path.relative_to(PROJECT_ROOT)),
                    "label_path": str(label_path.relative_to(PROJECT_ROOT)),
                })
            except Exception as e:
                skipped.append({
                    "waterbody": waterbody,
                    "date": str(date.date()),
                    "image": str(image_path),
                    "reason": str(e),
                })

    if not rows:
        raise RuntimeError("No records were created. Check the multi-waterbody image/label folders.")

    df = pd.DataFrame(rows).sort_values(["waterbody", "date"]).reset_index(drop=True)

    if df.duplicated(["waterbody", "date"]).any():
        raise RuntimeError("Duplicate waterbody/date records found.")

    # Target = NDCI at the next actual observation of the SAME waterbody.
    df["target_next_ndci"] = df.groupby("waterbody")["ndci_mean"].shift(-1)
    df["next_date"] = df.groupby("waterbody")["date"].shift(-1)
    df["days_to_next_observation"] = (df["next_date"] - df["date"]).dt.days

    features = [
        "ndci_mean", "ndci_median", "ndci_max",
        "ndwi_mean", "ndwi_median",
        "mndwi_mean", "mndwi_median",
        "fai_mean", "fai_median", "fai_max"
    ]
    finite = np.ones(len(df), dtype=bool)
    for c in features:
        finite &= np.isfinite(df[c].to_numpy())
    df["usable_for_supervised_learning"] = finite & df["target_next_ndci"].notna().to_numpy()

    df.to_csv(OUTPUT_FILE, index=False)
    pd.DataFrame(skipped).to_csv(SKIPPED_FILE, index=False)

    print("=" * 70)
    print("STEP 115: MULTI-WATERBODY TEMPORAL DATASET")
    print("=" * 70)
    print(f"Total records: {len(df)}")
    print(f"Usable supervised records: {int(df['usable_for_supervised_learning'].sum())}")
    print(f"Skipped records: {len(skipped)}")
    print()
    print("Records by waterbody:")
    print(df.groupby("waterbody").size().to_string())
    print()
    print("Date ranges:")
    print(df.groupby("waterbody")["date"].agg(["min", "max", "count"]).to_string())
    print()
    print("Next-observation intervals (days):")
    print(df.groupby("waterbody")["days_to_next_observation"].agg(["count", "median", "min", "max"]).to_string())
    print()
    print("Saved:")
    print(OUTPUT_FILE)
    print(SKIPPED_FILE)
    print("=" * 70)
    print("STEP 115 COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
