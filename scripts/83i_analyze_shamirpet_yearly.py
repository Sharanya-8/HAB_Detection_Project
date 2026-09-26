import os
import pandas as pd
import matplotlib

# Use a non-GUI backend so Windows Tkinter is not required
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# STEP 83I
# YEAR-WISE ANALYSIS OF SHAMIRPET TILE-BASED RESULTS
# ============================================================

# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# ------------------------------------------------------------
# INPUT FILE
# Created by Step 83H
# ------------------------------------------------------------

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "swin",
    "shamirpet_test",
    "tile_based",
    "shamirpet_tile_based_image_results.csv"
)

# ------------------------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------------------------

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "swin",
    "shamirpet_test",
    "tile_based",
    "yearly_analysis"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ------------------------------------------------------------
# OUTPUT FILES
# ------------------------------------------------------------

YEARLY_CSV = os.path.join(
    OUTPUT_DIR,
    "shamirpet_yearly_analysis.csv"
)

GRAPH_HAB = os.path.join(
    OUTPUT_DIR,
    "shamirpet_yearly_hab_comparison.png"
)

GRAPH_F1 = os.path.join(
    OUTPUT_DIR,
    "shamirpet_yearly_f1_iou.png"
)

GRAPH_PREDICTED = os.path.join(
    OUTPUT_DIR,
    "shamirpet_yearly_prediction_difference.png"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 83I - YEAR-WISE SHAMIRPET ANALYSIS")
print("=" * 70)


# ============================================================
# CHECK INPUT FILE
# ============================================================

if not os.path.exists(
    INPUT_FILE
):

    raise FileNotFoundError(
        "\nStep 83H result file was not found:\n"
        + INPUT_FILE
    )


print(
    "\nReading Step 83H results:"
)

print(
    INPUT_FILE
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"\nImages found in results: "
    f"{len(df)}"
)


if len(df) == 0:

    raise RuntimeError(
        "The Step 83H result file is empty."
    )


# ============================================================
# EXTRACT YEAR
# ============================================================

df["date"] = pd.to_datetime(
    df["date"]
)

df["year"] = (
    df["date"]
    .dt.year
)


# ============================================================
# CALCULATE PREDICTION DIFFERENCE
# ============================================================

df["hab_percent_difference"] = (
    df["predicted_hab_percent"]
    - df["actual_hab_percent"]
)


df["absolute_hab_percent_difference"] = (
    df["hab_percent_difference"]
    .abs()
)


# ============================================================
# YEAR-WISE AGGREGATION
# ============================================================

yearly = (
    df.groupby("year")
    .agg(
        images=(
            "date",
            "count"
        ),

        actual_hab_percent_mean=(
            "actual_hab_percent",
            "mean"
        ),

        actual_hab_percent_median=(
            "actual_hab_percent",
            "median"
        ),

        predicted_hab_percent_mean=(
            "predicted_hab_percent",
            "mean"
        ),

        predicted_hab_percent_median=(
            "predicted_hab_percent",
            "median"
        ),

        precision_mean=(
            "precision",
            "mean"
        ),

        recall_mean=(
            "recall",
            "mean"
        ),

        f1_mean=(
            "f1",
            "mean"
        ),

        iou_mean=(
            "iou",
            "mean"
        ),

        dice_mean=(
            "dice",
            "mean"
        ),

        accuracy_mean=(
            "accuracy",
            "mean"
        ),

        hab_percent_difference_mean=(
            "hab_percent_difference",
            "mean"
        ),

        absolute_hab_percent_difference_mean=(
            "absolute_hab_percent_difference",
            "mean"
        )
    )
    .reset_index()
)


# ============================================================
# SORT BY YEAR
# ============================================================

yearly = yearly.sort_values(
    "year"
)


# ============================================================
# SAVE YEARLY CSV
# ============================================================

yearly.to_csv(
    YEARLY_CSV,
    index=False
)


# ============================================================
# PRINT YEARLY TABLE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "YEAR-WISE RESULTS"
)

print(
    "=" * 70
)

print(
    "\n"
)

for _, row in yearly.iterrows():

    print(
        f"{int(row['year'])} | "
        f"Images: {int(row['images'])} | "
        f"Actual HAB: "
        f"{row['actual_hab_percent_mean']:.2f}% | "
        f"Predicted HAB: "
        f"{row['predicted_hab_percent_mean']:.2f}% | "
        f"F1: "
        f"{row['f1_mean']:.4f} | "
        f"IoU: "
        f"{row['iou_mean']:.4f}"
    )


# ============================================================
# OVERALL YEAR-WISE SUMMARY
# ============================================================

best_f1_row = yearly.loc[
    yearly["f1_mean"].idxmax()
]

worst_f1_row = yearly.loc[
    yearly["f1_mean"].idxmin()
]

smallest_difference_row = yearly.loc[
    yearly[
        "absolute_hab_percent_difference_mean"
    ].idxmin()
]

largest_difference_row = yearly.loc[
    yearly[
        "absolute_hab_percent_difference_mean"
    ].idxmax()
]


print(
    "\n" + "=" * 70
)

print(
    "YEAR-WISE SUMMARY"
)

print(
    "=" * 70
)

print(
    f"\nHighest mean F1 year: "
    f"{int(best_f1_row['year'])} "
    f"({best_f1_row['f1_mean']:.4f})"
)

print(
    f"Lowest mean F1 year: "
    f"{int(worst_f1_row['year'])} "
    f"({worst_f1_row['f1_mean']:.4f})"
)

print(
    f"\nClosest HAB-area agreement: "
    f"{int(smallest_difference_row['year'])} "
    f"("
    f"{smallest_difference_row['absolute_hab_percent_difference_mean']:.2f}"
    f" percentage points)"
)

print(
    f"Largest HAB-area difference: "
    f"{int(largest_difference_row['year'])} "
    f"("
    f"{largest_difference_row['absolute_hab_percent_difference_mean']:.2f}"
    f" percentage points)"
)


# ============================================================
# GRAPH 1
# ACTUAL VS PREDICTED HAB %
# ============================================================

plt.figure(
    figsize=(11, 6)
)

plt.plot(
    yearly["year"],
    yearly["actual_hab_percent_mean"],
    marker="o",
    label="Actual HAB %"
)

plt.plot(
    yearly["year"],
    yearly["predicted_hab_percent_mean"],
    marker="o",
    label="Predicted HAB %"
)

plt.xlabel(
    "Year"
)

plt.ylabel(
    "HAB Percentage (%)"
)

plt.title(
    "Shamirpet Lake - Year-wise Actual vs Predicted HAB"
)

plt.xticks(
    yearly["year"]
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    GRAPH_HAB,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GRAPH 2
# F1 AND IoU
# ============================================================

plt.figure(
    figsize=(11, 6)
)

plt.plot(
    yearly["year"],
    yearly["f1_mean"],
    marker="o",
    label="F1 Score"
)

plt.plot(
    yearly["year"],
    yearly["iou_mean"],
    marker="o",
    label="IoU"
)

plt.xlabel(
    "Year"
)

plt.ylabel(
    "Score"
)

plt.title(
    "Shamirpet Lake - Year-wise F1 and IoU"
)

plt.xticks(
    yearly["year"]
)

plt.ylim(
    0,
    1
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    GRAPH_F1,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GRAPH 3
# HAB PREDICTION DIFFERENCE
# ============================================================

plt.figure(
    figsize=(11, 6)
)

plt.bar(
    yearly["year"],
    yearly["hab_percent_difference_mean"]
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel(
    "Year"
)

plt.ylabel(
    "Predicted HAB % - Actual HAB %"
)

plt.title(
    "Shamirpet Lake - Year-wise HAB Prediction Difference"
)

plt.xticks(
    yearly["year"]
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    GRAPH_PREDICTED,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STEP 83I COMPLETE"
)

print(
    "=" * 70
)

print(
    "\nYear-wise CSV saved to:"
)

print(
    YEARLY_CSV
)

print(
    "\nHAB comparison graph saved to:"
)

print(
    GRAPH_HAB
)

print(
    "\nF1 / IoU graph saved to:"
)

print(
    GRAPH_F1
)

print(
    "\nPrediction difference graph saved to:"
)

print(
    GRAPH_PREDICTED
)

print(
    "\n" + "=" * 70
)