from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

LSTM_RESULTS = (
    BASE_DIR /
    "results" /
    "lstm" /
    "lstm_test_predictions.csv"
)

GRU_RESULTS = (
    BASE_DIR /
    "results" /
    "gru" /
    "gru_test_predictions.csv"
)

OUTPUT_DIR = (
    BASE_DIR /
    "results" /
    "forecasting"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD RESULTS
# ============================================================

print("=" * 65)
print("LSTM vs GRU FORECASTING COMPARISON")
print("=" * 65)

lstm = pd.read_csv(LSTM_RESULTS)
gru = pd.read_csv(GRU_RESULTS)

lstm["date"] = pd.to_datetime(lstm["date"])
gru["date"] = pd.to_datetime(gru["date"])


# ============================================================
# CALCULATE METRICS
# ============================================================

lstm_mae = lstm["absolute_error"].mean()

lstm_rmse = (
    (
        lstm["actual_ndci"]
        -
        lstm["predicted_ndci"]
    ) ** 2
).mean() ** 0.5


gru_mae = gru["absolute_error"].mean()

gru_rmse = (
    (
        gru["actual_ndci"]
        -
        gru["predicted_ndci"]
    ) ** 2
).mean() ** 0.5


# R²
lstm_ss_res = (
    (
        lstm["actual_ndci"]
        -
        lstm["predicted_ndci"]
    ) ** 2
).sum()

lstm_ss_tot = (
    (
        lstm["actual_ndci"]
        -
        lstm["actual_ndci"].mean()
    ) ** 2
).sum()

lstm_r2 = 1 - (
    lstm_ss_res /
    lstm_ss_tot
)


gru_ss_res = (
    (
        gru["actual_ndci"]
        -
        gru["predicted_ndci"]
    ) ** 2
).sum()

gru_ss_tot = (
    (
        gru["actual_ndci"]
        -
        gru["actual_ndci"].mean()
    ) ** 2
).sum()

gru_r2 = 1 - (
    gru_ss_res /
    gru_ss_tot
)


# ============================================================
# COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({

    "Model": [
        "LSTM",
        "GRU"
    ],

    "MAE": [
        lstm_mae,
        gru_mae
    ],

    "RMSE": [
        lstm_rmse,
        gru_rmse
    ],

    "R2": [
        lstm_r2,
        gru_r2
    ]
})


# ============================================================
# SAVE COMPARISON
# ============================================================

comparison_file = (
    OUTPUT_DIR /
    "lstm_vs_gru_comparison.csv"
)

comparison.to_csv(
    comparison_file,
    index=False
)


# ============================================================
# PRINT COMPARISON
# ============================================================

print("\nModel comparison:")
print()

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_model = comparison.loc[
    comparison["MAE"].idxmin(),
    "Model"
]

print(
    f"\nSelected forecasting model: {best_model}"
)

print(
    "Selection criterion: lowest MAE"
)


# ============================================================
# CREATE ACTUAL vs PREDICTED DATA
# ============================================================

plot_data = pd.DataFrame({

    "date": lstm["date"],

    "actual_ndci": lstm["actual_ndci"],

    "lstm_prediction": lstm["predicted_ndci"],

    "gru_prediction": gru["predicted_ndci"]

})


plot_file = (
    OUTPUT_DIR /
    "lstm_vs_gru_predictions.csv"
)

plot_data.to_csv(
    plot_file,
    index=False
)


# ============================================================
# PLOT 1 — LSTM
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    plot_data["date"],
    plot_data["actual_ndci"],
    marker="o",
    label="Actual NDCI"
)

plt.plot(
    plot_data["date"],
    plot_data["lstm_prediction"],
    marker="x",
    label="LSTM Prediction"
)

plt.xlabel("Date")
plt.ylabel("NDCI")
plt.title("LSTM: Actual vs Predicted NDCI")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

lstm_plot = (
    OUTPUT_DIR /
    "lstm_actual_vs_predicted.png"
)

plt.savefig(
    lstm_plot,
    dpi=300
)

plt.close()


# ============================================================
# PLOT 2 — GRU
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    plot_data["date"],
    plot_data["actual_ndci"],
    marker="o",
    label="Actual NDCI"
)

plt.plot(
    plot_data["date"],
    plot_data["gru_prediction"],
    marker="x",
    label="GRU Prediction"
)

plt.xlabel("Date")
plt.ylabel("NDCI")
plt.title("GRU: Actual vs Predicted NDCI")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

gru_plot = (
    OUTPUT_DIR /
    "gru_actual_vs_predicted.png"
)

plt.savefig(
    gru_plot,
    dpi=300
)

plt.close()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\nFiles created:")

print(comparison_file)
print(plot_file)
print(lstm_plot)
print(gru_plot)

print("\n" + "=" * 65)
print("COMPARISON COMPLETED")
print("=" * 65)