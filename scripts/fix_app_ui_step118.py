from pathlib import Path
import re

# ============================================================
# STEP 118 - FIX OBJECTIVE 3 MODEL ARCHITECTURE
# ============================================================

APP_FILE = Path("app_ui.py")

if not APP_FILE.exists():
    raise FileNotFoundError(
        "app_ui.py was not found. "
        "Run this script from the project root."
    )

text = APP_FILE.read_text(encoding="utf-8")

# ------------------------------------------------------------
# Correct MultiWaterbodyTemporalModel
# ------------------------------------------------------------

correct_model = '''class MultiWaterbodyTemporalModel(nn.Module):
    """
    Model architecture compatible with the Step 116
    multi-waterbody LSTM and GRU checkpoints.

    Step 116 saved:

        LSTM:
            lstm.weight_ih_l0
            lstm.weight_hh_l0
            ...

        GRU:
            gru.weight_ih_l0
            gru.weight_hh_l0
            ...

    Therefore the recurrent layer must be stored under
    self.lstm for LSTM and self.gru for GRU.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout,
        kind
    ):
        super().__init__()

        self.kind = kind.upper()

        if self.kind == "LSTM":

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=(
                    dropout
                    if num_layers > 1
                    else 0.0
                )
            )

        elif self.kind == "GRU":

            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=(
                    dropout
                    if num_layers > 1
                    else 0.0
                )
            )

        else:

            raise ValueError(
                f"Unsupported temporal model type: {kind}"
            )

        self.fc = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        if self.kind == "LSTM":

            output, _ = self.lstm(x)

        else:

            output, _ = self.gru(x)

        last_output = output[:, -1, :]

        return self.fc(
            last_output
        )


'''

# ------------------------------------------------------------
# Find existing class
# ------------------------------------------------------------

pattern = re.compile(
    r"class MultiWaterbodyTemporalModel\(nn\.Module\):.*?"
    r"(?=\n@st\.cache_data|\n@st\.cache_resource|\ndef )",
    re.DOTALL
)

match = pattern.search(text)

if match:

    print("Found existing MultiWaterbodyTemporalModel.")

    text = (
        text[:match.start()]
        + correct_model
        + text[match.end():]
    )

else:

    print(
        "MultiWaterbodyTemporalModel was not found."
    )

    # Try to insert before the first Objective 3 helper.
    objective_marker = (
        "# ============================================================\n"
        "# OBJECTIVE 3"
    )

    if objective_marker in text:

        text = text.replace(
            objective_marker,
            correct_model
            + objective_marker,
            1
        )

    else:

        raise RuntimeError(
            "Could not locate the Objective 3 section "
            "inside app_ui.py."
        )


# ------------------------------------------------------------
# Make sure the Objective 3 paths use the real Step 116 files
# ------------------------------------------------------------

replacements = {

    'TEMPORAL_DATASET = DATA / "processed" / "Hussain_Sagar_Clean_Temporal_Dataset.csv"':
        'TEMPORAL_DATASET = DATA / "processed" / "multiwaterbody_temporal_dataset.csv"',

    'TEMPORAL_MODEL_DIR = MODELS / "temporal_ui"':
        'TEMPORAL_MODEL_DIR = MODELS / "temporal"',

    'LSTM_UI_CHECKPOINT = TEMPORAL_MODEL_DIR / "ui_lstm_forecaster.pth"':
        'TEMPORAL_LSTM_CHECKPOINT = TEMPORAL_MODEL_DIR / "multiwaterbody_lstm_best.pth"',

    'GRU_UI_CHECKPOINT = TEMPORAL_MODEL_DIR / "ui_gru_forecaster.pth"':
        'TEMPORAL_GRU_CHECKPOINT = TEMPORAL_MODEL_DIR / "multiwaterbody_gru_best.pth"',

    'LSTM_UI_SCALER = TEMPORAL_MODEL_DIR / "ui_lstm_scaler.npz"':
        'TEMPORAL_SCALER_FILE = TEMPORAL_MODEL_DIR / "multiwaterbody_temporal_feature_scaler.pkl"',

    'GRU_UI_SCALER = TEMPORAL_MODEL_DIR / "ui_gru_scaler.npz"':
        '',
}

for old, new in replacements.items():

    if old in text:

        text = text.replace(
            old,
            new
        )


# ------------------------------------------------------------
# Correct feature list if old 13-feature list is still present
# ------------------------------------------------------------

old_feature_block = re.compile(
    r"FORECAST_FEATURES = \[\s*"
    r'"ndci_mean".*?'
    r'"wind_speed_ms"\s*'
    r"\]",
    re.DOTALL
)

new_feature_block = '''FORECAST_FEATURES = [
    "ndci_mean",
    "ndci_median",
    "ndci_max",
    "ndwi_mean",
    "ndwi_median",
    "mndwi_mean",
    "mndwi_median",
    "fai_mean",
    "fai_median",
    "fai_max",
]'''

text = old_feature_block.sub(
    new_feature_block,
    text,
    count=1
)


# ------------------------------------------------------------
# Remove old Hussain-Sagar-only warning if present
# ------------------------------------------------------------

text = text.replace(
    'The current trained temporal dataset/model is Hussain Sagar-specific.',
    'The current temporal model was trained on multiple waterbodies.'
)

text = text.replace(
    'The current temporal dataset is Hussain Sagar-specific.',
    'The current temporal model was trained on multiple waterbodies.'
)


# ------------------------------------------------------------
# Backup original file
# ------------------------------------------------------------

backup = Path("app_ui_before_step118_fix.py")

if not backup.exists():

    backup.write_text(
        APP_FILE.read_text(encoding="utf-8"),
        encoding="utf-8"
    )

    print(
        f"Backup created: {backup}"
    )


# ------------------------------------------------------------
# Save corrected app
# ------------------------------------------------------------

APP_FILE.write_text(
    text,
    encoding="utf-8"
)

print()
print("=" * 70)
print("STEP 118 FIX COMPLETE")
print("=" * 70)

print()
print("Updated:")
print("    app_ui.py")

print()
print("Backup:")
print("    app_ui_before_step118_fix.py")

print()
print("Objective 3 now uses:")

print("    Multi-waterbody temporal dataset")
print("    10 spectral/index features")
print("    Sequence length = 3")

print()
print("LSTM checkpoint:")
print("    models/temporal/multiwaterbody_lstm_best.pth")

print()
print("GRU checkpoint:")
print("    models/temporal/multiwaterbody_gru_best.pth")

print()
print("Scaler:")
print("    models/temporal/multiwaterbody_temporal_feature_scaler.pkl")

print()
print("The LSTM checkpoint will now load using self.lstm")
print("and the GRU checkpoint using self.gru.")

print()
print("=" * 70)