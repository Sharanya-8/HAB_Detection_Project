from pathlib import Path

import pandas as pd
import rasterio


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MAPS_DIR = BASE_DIR / "results" / "maps"

# GRU predictions are stored here
FORECAST_FILE = (
    BASE_DIR /
    "results" /
    "gru" /
    "gru_test_predictions.csv"
)

OUTPUT_DIR = BASE_DIR / "results" / "integration"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHECK INPUT FILE
# ============================================================

if not FORECAST_FILE.exists():

    raise FileNotFoundError(
        f"GRU prediction file not found:\n{FORECAST_FILE}"
    )


# ============================================================
# LOAD GRU FORECAST RESULTS
# ============================================================

print("=" * 70)
print("HAB SPATIAL + TEMPORAL INTEGRATION")
print("=" * 70)

forecast = pd.read_csv(
    FORECAST_FILE
)

forecast["date"] = pd.to_datetime(
    forecast["date"]
)

print(
    "\nGRU forecast observations:",
    len(forecast)
)

print(
    "GRU prediction file:",
    FORECAST_FILE
)


# ============================================================
# FIND SWIN MAPS
# ============================================================

map_files = sorted(
    MAPS_DIR.glob(
        "Hussain_Sagar_HAB_Swin_*.tif"
    )
)

if len(map_files) == 0:

    raise FileNotFoundError(
        "No Hussain Sagar Swin HAB prediction maps were found."
    )

print(
    "\nSwin HAB maps found:",
    len(map_files)
)


# ============================================================
# EXTRACT MAP STATISTICS
# ============================================================

map_records = []

for map_file in map_files:

    # --------------------------------------------------------
    # Extract date from filename
    # --------------------------------------------------------

    date_text = (
        map_file.stem
        .replace(
            "Hussain_Sagar_HAB_Swin_",
            ""
        )
    )

    map_date = pd.to_datetime(
        date_text,
        format="%Y_%m_%d"
    )

    # --------------------------------------------------------
    # Read prediction map
    # --------------------------------------------------------

    with rasterio.open(map_file) as src:

        data = src.read(1)

        # 255 = ignored / outside valid prediction area
        valid = data != 255

        water_pixels = int(
            valid.sum()
        )

        hab_pixels = int(
            ((data == 1) & valid).sum()
        )

        non_hab_pixels = int(
            ((data == 0) & valid).sum()
        )

        if water_pixels > 0:

            hab_percentage = (
                hab_pixels /
                water_pixels
            ) * 100

        else:

            hab_percentage = 0.0

        map_records.append({

            "date": map_date,

            "water_pixels": water_pixels,

            "hab_pixels": hab_pixels,

            "non_hab_pixels": non_hab_pixels,

            "hab_percentage": hab_percentage

        })


# ============================================================
# CREATE SPATIAL DATAFRAME
# ============================================================

spatial = pd.DataFrame(
    map_records
)

spatial = spatial.sort_values(
    "date"
).reset_index(
    drop=True
)

print(
    "\nSpatial map observations:",
    len(spatial)
)


# ============================================================
# MERGE SPATIAL + TEMPORAL RESULTS
# ============================================================

integration = pd.merge(
    spatial,
    forecast,
    on="date",
    how="outer"
)

integration = integration.sort_values(
    "date"
).reset_index(
    drop=True
)


# ============================================================
# IDENTIFY DATA AVAILABILITY
# ============================================================

integration["spatial_available"] = (
    integration["hab_pixels"].notna()
)

integration["forecast_available"] = (
    integration["predicted_ndci"].notna()
)


# ============================================================
# CREATE INTEGRATION STATUS
# ============================================================

integration["integrated_status"] = "No Data"


# Both spatial and temporal information available
integration.loc[
    (
        integration["spatial_available"]
        &
        integration["forecast_available"]
    ),
    "integrated_status"
] = "Spatial + Temporal"


# Only spatial information available
integration.loc[
    (
        integration["spatial_available"]
        &
        ~integration["forecast_available"]
    ),
    "integrated_status"
] = "Spatial Only"


# Only temporal information available
integration.loc[
    (
        ~integration["spatial_available"]
        &
        integration["forecast_available"]
    ),
    "integrated_status"
] = "Temporal Only"


# ============================================================
# SAVE COMPLETE INTEGRATION TABLE
# ============================================================

output_file = (
    OUTPUT_DIR /
    "Hussain_Sagar_Spatial_Temporal_Integration.csv"
)

integration.to_csv(
    output_file,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\nIntegrated dataset:")

print(
    integration.to_string(
        index=False
    )
)


# ============================================================
# PRINT STATUS SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("INTEGRATION SUMMARY")
print("=" * 70)

print(
    "\nTotal integrated dates:",
    len(integration)
)

print(
    "Spatial + Temporal:",
    (
        integration["integrated_status"]
        == "Spatial + Temporal"
    ).sum()
)

print(
    "Spatial Only:",
    (
        integration["integrated_status"]
        == "Spatial Only"
    ).sum()
)

print(
    "Temporal Only:",
    (
        integration["integrated_status"]
        == "Temporal Only"
    ).sum()
)


# ============================================================
# OUTPUT LOCATION
# ============================================================

print("\nIntegrated file saved to:")

print(output_file)

print("\n" + "=" * 70)
print("INTEGRATION COMPLETED SUCCESSFULLY")
print("=" * 70)