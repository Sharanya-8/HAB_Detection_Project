import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# STEP 85 - WATERBODY-WISE MODEL PERFORMANCE REPORT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "swin",
    "step82_diagnosis",
    "waterbody_wise_results.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "swin",
    "step85_waterbody_report"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 70)
print("STEP 85 - WATERBODY-WISE MODEL PERFORMANCE REPORT")
print("=" * 70)

print("\nReading:")
print(INPUT_FILE)


# ============================================================
# READ STEP 82 RESULTS
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("\nWaterbodies found:")
print(df["waterbody"].to_string(index=False))


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("WATERBODY-WISE PERFORMANCE")
print("=" * 70)

for _, row in df.iterrows():

    print(f"\n{row['waterbody']}")

    print(
        f"  Validation tiles : "
        f"{int(row['validation_tiles'])}"
    )

    print(
        f"  Precision        : "
        f"{row['precision']:.4f}"
    )

    print(
        f"  Recall           : "
        f"{row['recall']:.4f}"
    )

    print(
        f"  F1               : "
        f"{row['f1']:.4f}"
    )

    print(
        f"  IoU              : "
        f"{row['iou']:.4f}"
    )

    print(
        f"  Actual HAB       : "
        f"{row['actual_hab_percent']:.2f}%"
    )

    print(
        f"  Predicted HAB    : "
        f"{row['predicted_hab_percent']:.2f}%"
    )


# ============================================================
# SAVE REPORT
# ============================================================

report_file = os.path.join(
    OUTPUT_DIR,
    "step85_waterbody_performance.csv"
)

df.to_csv(
    report_file,
    index=False
)


# ============================================================
# PERFORMANCE GRAPH
# ============================================================

metrics = [
    "precision",
    "recall",
    "f1",
    "iou"
]

metric_labels = [
    "Precision",
    "Recall",
    "F1",
    "IoU"
]

waterbodies = df["waterbody"].tolist()

x = list(range(len(waterbodies)))

width = 0.18

plt.figure(figsize=(13, 7))

for i, metric in enumerate(metrics):

    values = df[metric].values

    positions = [
        value + (i - 1.5) * width
        for value in x
    ]

    plt.bar(
        positions,
        values,
        width=width,
        label=metric_labels[i]
    )


plt.xticks(
    x,
    waterbodies,
    rotation=20
)

plt.ylabel("Score")
plt.xlabel("Waterbody")

plt.title(
    "Corrected Swin Transformer - Waterbody-wise Performance"
)

plt.legend()

plt.tight_layout()

performance_graph = os.path.join(
    OUTPUT_DIR,
    "step85_waterbody_performance.png"
)

plt.savefig(
    performance_graph,
    dpi=200
)

plt.close()


# ============================================================
# ACTUAL VS PREDICTED HAB
# ============================================================

plt.figure(figsize=(12, 7))

x = list(range(len(waterbodies)))

plt.bar(
    [i - 0.2 for i in x],
    df["actual_hab_percent"],
    width=0.4,
    label="Pseudo-label HAB"
)

plt.bar(
    [i + 0.2 for i in x],
    df["predicted_hab_percent"],
    width=0.4,
    label="Swin predicted HAB"
)

plt.xticks(
    x,
    waterbodies,
    rotation=20
)

plt.ylabel("HAB Area (%)")
plt.xlabel("Waterbody")

plt.title(
    "Actual vs Predicted HAB Coverage"
)

plt.legend()

plt.tight_layout()

hab_graph = os.path.join(
    OUTPUT_DIR,
    "step85_actual_vs_predicted_hab.png"
)

plt.savefig(
    hab_graph,
    dpi=200
)

plt.close()


# ============================================================
# CALCULATE HAB DIFFERENCE
# ============================================================

df["hab_difference_percentage_points"] = (
    df["predicted_hab_percent"]
    - df["actual_hab_percent"]
)


difference_file = os.path.join(
    OUTPUT_DIR,
    "step85_hab_area_difference.csv"
)

df[
    [
        "waterbody",
        "actual_hab_percent",
        "predicted_hab_percent",
        "hab_difference_percentage_points"
    ]
].to_csv(
    difference_file,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 70)
print("STEP 85 COMPLETE")
print("=" * 70)

print("\nReport saved:")
print(report_file)

print("\nPerformance graph saved:")
print(performance_graph)

print("\nHAB comparison graph saved:")
print(hab_graph)

print("\nHAB area difference saved:")
print(difference_file)

print("\nDo NOT retrain yet.")
print("Send me the complete terminal output.")