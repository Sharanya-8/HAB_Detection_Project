import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# STEP 86 - SHAMIRPET ERROR ANALYSIS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PREDICTION_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "swin",
    "shamirpet_test",
    "tile_based",
    "predictions"
)

LABEL_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "labels",
    "hab_masks",
    "shamirpet_test"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "swin",
    "shamirpet_test",
    "error_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 70)
print("STEP 86 - SHAMIRPET ERROR ANALYSIS")
print("=" * 70)

print("\nPrediction directory:")
print(PREDICTION_DIR)

print("\nLabel directory:")
print(LABEL_DIR)


# ============================================================
# FIND FILES
# ============================================================

prediction_files = sorted(
    [
        os.path.join(PREDICTION_DIR, f)
        for f in os.listdir(PREDICTION_DIR)
        if f.endswith(".tif")
    ]
)

print(
    f"\nPrediction masks found: "
    f"{len(prediction_files)}"
)


# ============================================================
# FIND CORRESPONDING LABEL
# ============================================================

def find_label(prediction_path):

    filename = os.path.basename(prediction_path)

    # Prediction names are expected to contain the date.
    # Extract YYYY-MM-DD.
    date = filename[:10]

    possible = [
        os.path.join(
            LABEL_DIR,
            f"{date}_HAB.tif"
        ),
        os.path.join(
            LABEL_DIR,
            f"{date}.tif"
        )
    ]

    for path in possible:

        if os.path.exists(path):
            return path

    # Search recursively if exact filename isn't found
    for root, _, files in os.walk(LABEL_DIR):

        for file in files:

            if file.startswith(date) and file.endswith(".tif"):
                return os.path.join(root, file)

    return None


# ============================================================
# ANALYSIS
# ============================================================

results = []

total_tp = 0
total_fp = 0
total_fn = 0
total_tn = 0


print("\nAnalyzing images...")


for index, prediction_path in enumerate(
    prediction_files,
    start=1
):

    label_path = find_label(prediction_path)

    if label_path is None:

        print(
            f"\nWARNING: label not found for "
            f"{os.path.basename(prediction_path)}"
        )

        continue


    with rasterio.open(prediction_path) as src:

        prediction = src.read(1)


    with rasterio.open(label_path) as src:

        label = src.read(1)


    # --------------------------------------------------------
    # MAKE SURE SHAPES MATCH
    # --------------------------------------------------------

    if prediction.shape != label.shape:

        print(
            f"\nWARNING: shape mismatch for "
            f"{os.path.basename(prediction_path)}"
        )

        print(
            f"Prediction: {prediction.shape}"
        )

        print(
            f"Label: {label.shape}"
        )

        continue


    # --------------------------------------------------------
    # VALID PIXELS
    # --------------------------------------------------------

    valid = (
        np.isfinite(prediction) &
        np.isfinite(label) &
        (label != 255)
    )


    pred = (
        prediction[valid] > 0
    ).astype(np.uint8)

    true = (
        label[valid] > 0
    ).astype(np.uint8)


    # --------------------------------------------------------
    # CONFUSION COMPONENTS
    # --------------------------------------------------------

    tp = np.sum(
        (pred == 1) &
        (true == 1)
    )

    fp = np.sum(
        (pred == 1) &
        (true == 0)
    )

    fn = np.sum(
        (pred == 0) &
        (true == 1)
    )

    tn = np.sum(
        (pred == 0) &
        (true == 0)
    )


    total_tp += tp
    total_fp += fp
    total_fn += fn
    total_tn += tn


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    iou = (
        tp / (tp + fp + fn)
        if (tp + fp + fn) > 0
        else 0
    )


    # --------------------------------------------------------
    # PERCENTAGES
    # --------------------------------------------------------

    total_valid = len(true)

    actual_hab_percent = (
        np.sum(true == 1) /
        total_valid *
        100
    )

    predicted_hab_percent = (
        np.sum(pred == 1) /
        total_valid *
        100
    )

    fp_percent = (
        fp /
        total_valid *
        100
    )

    fn_percent = (
        fn /
        total_valid *
        100
    )


    results.append({

        "date":
            os.path.basename(
                prediction_path
            )[:10],

        "actual_hab_percent":
            actual_hab_percent,

        "predicted_hab_percent":
            predicted_hab_percent,

        "false_positive_percent":
            fp_percent,

        "false_negative_percent":
            fn_percent,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "iou":
            iou,

        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn)
    })


    print(
        f"Processed {index}/{len(prediction_files)}",
        end="\r"
    )


# ============================================================
# SAVE IMAGE-LEVEL RESULTS
# ============================================================

df = pd.DataFrame(results)

csv_path = os.path.join(
    OUTPUT_DIR,
    "shamirpet_error_analysis.csv"
)

df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# OVERALL METRICS
# ============================================================

precision = (
    total_tp /
    (total_tp + total_fp)
    if (total_tp + total_fp) > 0
    else 0
)

recall = (
    total_tp /
    (total_tp + total_fn)
    if (total_tp + total_fn) > 0
    else 0
)

f1 = (
    2 * precision * recall /
    (precision + recall)
    if (precision + recall) > 0
    else 0
)

iou = (
    total_tp /
    (
        total_tp +
        total_fp +
        total_fn
    )
    if (
        total_tp +
        total_fp +
        total_fn
    ) > 0
    else 0
)

total_pixels = (
    total_tp +
    total_fp +
    total_fn +
    total_tn
)

actual_hab_percent = (
    (total_tp + total_fn) /
    total_pixels *
    100
)

predicted_hab_percent = (
    (total_tp + total_fp) /
    total_pixels *
    100
)

fp_percent = (
    total_fp /
    total_pixels *
    100
)

fn_percent = (
    total_fn /
    total_pixels *
    100
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("OVERALL ERROR ANALYSIS")
print("=" * 70)

print(
    f"\nTotal images analyzed : {len(df)}"
)

print(
    f"Total valid pixels    : {total_pixels:,}"
)

print(
    f"\nTrue Positives        : {total_tp:,}"
)

print(
    f"False Positives       : {total_fp:,}"
)

print(
    f"False Negatives       : {total_fn:,}"
)

print(
    f"True Negatives        : {total_tn:,}"
)

print(
    f"\nActual HAB            : "
    f"{actual_hab_percent:.2f}%"
)

print(
    f"Predicted HAB         : "
    f"{predicted_hab_percent:.2f}%"
)

print(
    f"False Positive pixels : "
    f"{fp_percent:.2f}%"
)

print(
    f"False Negative pixels : "
    f"{fn_percent:.2f}%"
)

print(
    f"\nPrecision             : "
    f"{precision:.4f}"
)

print(
    f"Recall                : "
    f"{recall:.4f}"
)

print(
    f"F1                    : "
    f"{f1:.4f}"
)

print(
    f"IoU                   : "
    f"{iou:.4f}"
)


# ============================================================
# CREATE ERROR MAPS
# ============================================================

# Select five representative images:
# first, middle, last and two evenly spaced images.

if len(results) > 0:

    selected_indices = np.linspace(
        0,
        len(results) - 1,
        min(5, len(results)),
        dtype=int
    )

    selected_dates = [
        results[i]["date"]
        for i in selected_indices
    ]

else:

    selected_dates = []


print("\n")
print("=" * 70)
print("CREATING ERROR MAPS")
print("=" * 70)


for date in selected_dates:

    prediction_path = None

    for path in prediction_files:

        if os.path.basename(path).startswith(date):

            prediction_path = path
            break


    if prediction_path is None:
        continue


    label_path = find_label(prediction_path)

    if label_path is None:
        continue


    with rasterio.open(prediction_path) as src:

        prediction = src.read(1)


    with rasterio.open(label_path) as src:

        label = src.read(1)


    # --------------------------------------------------------
    # ERROR MAP
    #
    # 0 = background / TN
    # 1 = TP
    # 2 = FP
    # 3 = FN
    # --------------------------------------------------------

    error_map = np.zeros(
        prediction.shape,
        dtype=np.uint8
    )

    valid = label != 255

    error_map[
        valid &
        (label == 1) &
        (prediction == 1)
    ] = 1

    error_map[
        valid &
        (label == 0) &
        (prediction == 1)
    ] = 2

    error_map[
        valid &
        (label == 1) &
        (prediction == 0)
    ] = 3


    plt.figure(figsize=(8, 7))

    plt.imshow(
        error_map,
        interpolation="nearest"
    )

    plt.title(
        f"Shamirpet Error Map - {date}\n"
        "1=TP, 2=FP, 3=FN"
    )

    plt.colorbar(
        ticks=[0, 1, 2, 3],
        label="Error Class"
    )

    plt.tight_layout()


    output_path = os.path.join(
        OUTPUT_DIR,
        f"{date}_error_map.png"
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()


# ============================================================
# SUMMARY CSV
# ============================================================

summary = pd.DataFrame([
    {
        "images_analyzed":
            len(df),

        "actual_hab_percent":
            actual_hab_percent,

        "predicted_hab_percent":
            predicted_hab_percent,

        "false_positive_percent":
            fp_percent,

        "false_negative_percent":
            fn_percent,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "iou":
            iou,

        "TP":
            total_tp,

        "FP":
            total_fp,

        "FN":
            total_fn,

        "TN":
            total_tn
    }
])

summary_path = os.path.join(
    OUTPUT_DIR,
    "shamirpet_error_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("STEP 86 COMPLETE")
print("=" * 70)

print("\nImage-level error analysis:")
print(csv_path)

print("\nOverall error summary:")
print(summary_path)

print("\nError maps saved in:")
print(OUTPUT_DIR)

print("\nDo NOT retrain yet.")
print("Send me the complete terminal output.")