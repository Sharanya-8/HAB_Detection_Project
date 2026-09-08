# ============================================================
# HAB DETECTION PROJECT CONFIGURATION
# ============================================================

from pathlib import Path


# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------
# DATA DIRECTORIES
# ------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
SENTINEL_DIR = RAW_DIR / "sentinel2"

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_SENTINEL_DIR = PROCESSED_DIR / "sentinel2"

LABEL_DIR = DATA_DIR / "labels"
HAB_MASK_DIR = LABEL_DIR / "hab_masks"

TILES_DIR = DATA_DIR / "tiles"
IMAGE_TILES_DIR = TILES_DIR / "images"
MASK_TILES_DIR = TILES_DIR / "masks"


# ------------------------------------------------------------
# MODEL DIRECTORIES
# ------------------------------------------------------------

MODELS_DIR = PROJECT_ROOT / "models"

SWIN_MODEL_DIR = MODELS_DIR / "swin"
SEGFORMER_MODEL_DIR = MODELS_DIR / "segformer"


# ------------------------------------------------------------
# RESULTS DIRECTORIES
# ------------------------------------------------------------

RESULTS_DIR = PROJECT_ROOT / "results"

SWIN_RESULTS_DIR = RESULTS_DIR / "swin"
SEGFORMER_RESULTS_DIR = RESULTS_DIR / "segformer"
MAP_RESULTS_DIR = RESULTS_DIR / "maps"


# ------------------------------------------------------------
# GOOGLE EARTH ENGINE
# ------------------------------------------------------------

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"

S2_CLOUD_PROBABILITY = (
    "COPERNICUS/S2_CLOUD_PROBABILITY"
)


# ------------------------------------------------------------
# STUDY AREA
# ------------------------------------------------------------
# These are currently TEST coordinates.
# Replace them with the selected waterbody coordinates
# before final data collection.

MIN_LON = 78.35
MIN_LAT = 17.30

MAX_LON = 78.55
MAX_LAT = 17.50


# ------------------------------------------------------------
# DATE RANGE
# ------------------------------------------------------------

START_DATE = "2023-01-01"
END_DATE = "2025-12-31"


# ------------------------------------------------------------
# CLOUD SETTINGS
# ------------------------------------------------------------

MAX_SCENE_CLOUD = 30

CLOUD_PROBABILITY_THRESHOLD = 40


# ------------------------------------------------------------
# PROCESSING SETTINGS
# ------------------------------------------------------------

EXPORT_SCALE = 10

TILE_SIZE = 256


# ------------------------------------------------------------
# SENTINEL-2 BANDS
# ------------------------------------------------------------

SENTINEL_BANDS = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12"
]


# ------------------------------------------------------------
# DERIVED FEATURES
# ------------------------------------------------------------

DERIVED_FEATURES = [
    "NDWI",
    "MNDWI",
    "NDCI",
    "FAI"
]


# ------------------------------------------------------------
# CREATE REQUIRED DIRECTORIES
# ------------------------------------------------------------

ALL_DIRECTORIES = [
    SENTINEL_DIR,
    PROCESSED_SENTINEL_DIR,
    HAB_MASK_DIR,
    IMAGE_TILES_DIR,
    MASK_TILES_DIR,
    SWIN_MODEL_DIR,
    SEGFORMER_MODEL_DIR,
    SWIN_RESULTS_DIR,
    SEGFORMER_RESULTS_DIR,
    MAP_RESULTS_DIR
]


for directory in ALL_DIRECTORIES:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


print("HAB Detection Project configuration loaded.")