"""
STEP 98
Create historical HAB graphs for Shamirpet.

Outputs:
1. HAB area vs year
2. HAB coverage vs year
3. Clean yearly summary CSV
"""

from pathlib import Path

import os
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt


# ================================================================
# PROJECT ROOT
# ================================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)


# ================================================================
# INPUT
# ================================================================

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "swin"
    / "shamirpet_yearly_results.csv"
)


# ================================================================
# OUTPUT DIRECTORY
# ================================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "swin"
    / "shamirpet_yearly"
    / "historical_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# OUTPUT FILES
# ================================================================

AREA_GRAPH = (
    OUTPUT_DIR
    / "shamirpet_hab_area_by_year.png"
)

COVERAGE_GRAPH = (
    OUTPUT_DIR
    / "shamirpet_hab_coverage_by_year.png"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "shamirpet_yearly_summary.csv"
)


# ================================================================
# LOAD DATA
# ================================================================

print("=" * 70)
print("STEP 98 — SHAMIRPET HISTORICAL ANALYSIS")
print("=" * 70)

print()
print("Reading:")
print(INPUT_CSV)

df = pd.read_csv(
    INPUT_CSV
)

df = df[
    df["status"] == "success"
].copy()

df = df.sort_values(
    "year"
)


# ================================================================
# DISPLAY DATA
# ================================================================

print()
print("=" * 70)
print("YEARLY DATA")
print("=" * 70)

print()

for _, row in df.iterrows():

    print(
        f"{int(row['year'])} | "
        f"{row['date']} | "
        f"{row['hab_percentage']:.2f}% | "
        f"{row['hab_area_ha']:.4f} ha"
    )


# ================================================================
# SAVE CLEAN SUMMARY
# ================================================================

summary = df[
    [
        "year",
        "date",
        "water_pixels",
        "hab_pixels",
        "hab_percentage",
        "hab_area_ha"
    ]
].copy()

summary.to_csv(
    SUMMARY_CSV,
    index=False
)


# ================================================================
# GRAPH 1 — HAB AREA
# ================================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    summary["year"],
    summary["hab_area_ha"],
    marker="o",
    linewidth=2
)

plt.xlabel(
    "Year"
)

plt.ylabel(
    "HAB Area (hectares)"
)

plt.title(
    "Shamirpet Lake — Yearly HAB Area"
)

plt.xticks(
    summary["year"]
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    AREA_GRAPH,
    dpi=200
)

plt.close()


# ================================================================
# GRAPH 2 — HAB COVERAGE
# ================================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    summary["year"],
    summary["hab_percentage"],
    marker="o",
    linewidth=2
)

plt.xlabel(
    "Year"
)

plt.ylabel(
    "HAB Coverage (%)"
)

plt.title(
    "Shamirpet Lake — Yearly HAB Coverage"
)

plt.xticks(
    summary["year"]
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    COVERAGE_GRAPH,
    dpi=200
)

plt.close()


# ================================================================
# SUMMARY
# ================================================================

highest_area = summary.loc[
    summary["hab_area_ha"].idxmax()
]

lowest_area = summary.loc[
    summary["hab_area_ha"].idxmin()
]

print()
print("=" * 70)
print("HISTORICAL SUMMARY")
print("=" * 70)

print()

print(
    f"Highest predicted HAB area: "
    f"{int(highest_area['year'])} "
    f"({highest_area['hab_area_ha']:.4f} ha)"
)

print(
    f"Lowest predicted HAB area: "
    f"{int(lowest_area['year'])} "
    f"({lowest_area['hab_area_ha']:.4f} ha)"
)

print()
print("Area graph:")
print(AREA_GRAPH)

print()
print("Coverage graph:")
print(COVERAGE_GRAPH)

print()
print("Summary CSV:")
print(SUMMARY_CSV)

print()
print("=" * 70)
print("STEP 98 COMPLETE")
print("=" * 70)