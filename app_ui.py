import os
import sys
import json
import math
import time
import re
import pickle
import importlib.util
import textwrap
import smtplib
from email.message import EmailMessage
from datetime import datetime, date
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import requests
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape, mapping
from shapely.ops import transform
import pyproj

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import folium
from streamlit_folium import st_folium

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

if ROOT.name == "scripts":
    ROOT = ROOT.parent

DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

RAW_GENERIC = DATA / "raw" / "sentinel2" / "ui_selected"
MASK_GENERIC = RESULTS / "water_masks" / "ui_selected"
PRED_GENERIC = RESULTS / "swin" / "ui_analysis"
MULTI_WATERBODY_RAW = DATA / "raw" / "sentinel2" / "multi_waterbody"
HISTORICAL_UI_DIR = RESULTS / "swin" / "ui_historical"
HISTORICAL_UI_DIR.mkdir(parents=True, exist_ok=True)

# Objective 3 temporal forecasting paths
TEMPORAL_DATASET = DATA / "processed" / "multiwaterbody_temporal_dataset.csv"
TEMPORAL_RESULTS_DIR = RESULTS / "temporal"
TEMPORAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GENERIC_TEMPORAL_RAW = DATA / "raw" / "sentinel2" / "ui_temporal"
GENERIC_TEMPORAL_MASKS = RESULTS / "water_masks" / "ui_temporal"
GENERIC_TEMPORAL_DIR = TEMPORAL_RESULTS_DIR / "ui_waterbodies"
GENERIC_TEMPORAL_RAW.mkdir(parents=True, exist_ok=True)
GENERIC_TEMPORAL_MASKS.mkdir(parents=True, exist_ok=True)
GENERIC_TEMPORAL_DIR.mkdir(parents=True, exist_ok=True)
TEMPORAL_MODEL_DIR = MODELS / "temporal"
TEMPORAL_LSTM_CHECKPOINT = TEMPORAL_MODEL_DIR / "multiwaterbody_lstm_best.pth"
TEMPORAL_GRU_CHECKPOINT = TEMPORAL_MODEL_DIR / "multiwaterbody_gru_best.pth"
TEMPORAL_SCALER_FILE = TEMPORAL_MODEL_DIR / "multiwaterbody_temporal_feature_scaler.pkl"
TEMPORAL_METRICS_FILE = TEMPORAL_RESULTS_DIR / "multiwaterbody_lstm_gru_metrics.csv"
SHAMIRPET_TEMPORAL_RESULTS_DIR = TEMPORAL_RESULTS_DIR / "shamirpet_test"
SHAMIRPET_TEMPORAL_METRICS_FILE = SHAMIRPET_TEMPORAL_RESULTS_DIR / "shamirpet_lstm_gru_metrics.csv"
SHAMIRPET_TEMPORAL_PREDICTIONS_FILE = SHAMIRPET_TEMPORAL_RESULTS_DIR / "shamirpet_lstm_gru_predictions.csv"
FORECAST_FEATURES = [
    "ndci_mean", "ndci_median", "ndci_max",
    "ndwi_mean", "ndwi_median",
    "mndwi_mean", "mndwi_median",
    "fai_mean", "fai_median", "fai_max"
]
FORECAST_SEQUENCE_LENGTH = 3
HAB_NDCI_ALERT_THRESHOLD = 0.30


RAW_GENERIC.mkdir(parents=True, exist_ok=True)
MASK_GENERIC.mkdir(parents=True, exist_ok=True)
PRED_GENERIC.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="HAB WATCH",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# SAFE HTML RENDERING
# ============================================================
# Streamlit's Markdown parser treats indented HTML as a code block.
# The original UI contains many deliberately indented HTML cards.
# Route HTML-looking content through st.html so every page renders
# as designed instead of displaying <div>/<p> tags as code.
_original_st_markdown = st.markdown

def _hab_markdown(body, *args, **kwargs):
    if isinstance(body, str):
        stripped = body.lstrip()
        html_markers = ("<div", "<style", "<hr", "<section", "<table", "<p>")
        if stripped.startswith(html_markers) or "<div class=" in body or "<style>" in body:
            return st.html(textwrap.dedent(body).strip())
    return _original_st_markdown(body, *args, **kwargs)

st.markdown = _hab_markdown


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    .stApp {
        background: #f5f8f9;
        color: #294b5b !important;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    /* ALL NORMAL TEXT */

    p,
    span,
    label,
    div {
        color: #294b5b;
    }

    /* ======================================================
       HEADINGS
       ====================================================== */

    h1,
    h2,
    h3,
    h4,
    h5,
    h6 {
        color: #294b5b !important;
    }

    /* Streamlit title */

    [data-testid="stHeading"] {
        color: #294b5b !important;
    }

    [data-testid="stHeading"] h1,
    [data-testid="stHeading"] h2,
    [data-testid="stHeading"] h3 {
        color: #294b5b !important;
    }

    /* ======================================================
       HAB WATCH HEADER
       ====================================================== */

    .hab-header {
        background: linear-gradient(
            135deg,
            #183746 0%,
            #21495a 50%,
            #477f80 100%
        );

        padding: 20px 28px;
        border-radius: 12px;
        margin-bottom: 22px;

        box-shadow:
            0 4px 14px rgba(0,0,0,0.10);
    }

    .hab-logo {
        color: #ffffff !important;
        font-size: 30px;
        font-weight: 800;
        letter-spacing: 1px;
    }

    .hab-subtitle {
        color: #b8dcde !important;
        font-size: 12px;
        letter-spacing: 1.2px;
        margin-top: 3px;
    }

    /* ======================================================
       HERO
       ====================================================== */

    .hero {
        background: linear-gradient(
            90deg,
            #183746 0%,
            #285463 50%,
            #4d8d8c 100%
        );

        border-radius: 14px;
        padding: 58px 55px;
        color: white !important;
        min-height: 270px;

        display: flex;
        flex-direction: column;
        justify-content: center;

        box-shadow:
            0 5px 18px rgba(0,0,0,0.10);

        margin-bottom: 25px;
    }

    .hero h1 {
        color: #ffffff !important;
        font-size: 48px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #f4fbfb !important;
        font-size: 19px;
        max-width: 850px;
    }

    /* ======================================================
       CARDS
       ====================================================== */

    .card {
        background: #ffffff;
        border: 1px solid #dce5e7;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 18px;

        box-shadow:
            0 3px 12px rgba(0,0,0,0.05);

        color: #294b5b !important;
    }

    .card-title {
        color: #294b5b !important;
        font-size: 20px;
        font-weight: 750;
        margin-bottom: 10px;
    }

    /* ======================================================
       METRIC CARDS
       ====================================================== */

    .metric-card {
        background: #ffffff;
        border: 1px solid #dce5e7;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        min-height: 115px;

        box-shadow:
            0 3px 10px rgba(0,0,0,0.04);
    }

    .metric-label {
        color: #72858d !important;
        font-size: 13px;
        margin-bottom: 7px;
    }

    .metric-value {
        color: #244b5b !important;
        font-size: 27px;
        font-weight: 800;
    }

    /* ======================================================
       WATERBODY CARDS
       ====================================================== */

    .waterbody-card {
        background: #ffffff;
        border: 1px solid #dce5e7;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;

        box-shadow:
            0 3px 10px rgba(0,0,0,0.04);

        color: #294b5b !important;
    }

    .waterbody-name {
        color: #284c5c !important;
        font-size: 18px;
        font-weight: 750;
    }

    .small-muted {
        color: #73858c !important;
        font-size: 13px;
    }

    /* ======================================================
       INFO / SUCCESS / WARNING / ERROR
       ====================================================== */

    .success-box {
        background: #e7f5f1;
        border-left: 5px solid #4e9088;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        color: #24574f !important;
    }

    .success-box * {
        color: #24574f !important;
    }

    .warning-box {
        background: #fff5df;
        border-left: 5px solid #e1aa45;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        color: #76551b !important;
    }

    .warning-box * {
        color: #76551b !important;
    }

    .error-box {
        background: #fdeaea;
        border-left: 5px solid #d85b5b;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        color: #8b3030 !important;
    }

    .error-box * {
        color: #8b3030 !important;
    }

    .info-box {
        background: #eaf3fb;
        border-left: 5px solid #659ac8;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        color: #285675 !important;
    }

    .info-box * {
        color: #285675 !important;
    }

    /* ======================================================
       STREAMLIT ALERTS
       ====================================================== */

    [data-testid="stAlert"] {
        color: #294b5b !important;
    }

    [data-testid="stAlert"] p {
        color: #294b5b !important;
    }

    [data-testid="stAlert"] div {
        color: #294b5b !important;
    }

    /* ======================================================
       BUTTONS
       ====================================================== */

    div.stButton > button {
        background: #4f908a !important;
        color: #ffffff !important;

        border: none !important;
        border-radius: 8px !important;

        padding: 0.65rem 1.2rem;

        font-weight: 650;
    }

    div.stButton > button p {
        color: #ffffff !important;
    }

    div.stButton > button span {
        color: #ffffff !important;
    }

    div.stButton > button:hover {
        background: #3e7773 !important;
        color: #ffffff !important;
    }

    /* ======================================================
       INPUT LABELS
       ====================================================== */

    [data-testid="stWidgetLabel"] {
        color: #526c76 !important;
    }

    [data-testid="stWidgetLabel"] p {
        color: #526c76 !important;
    }

    /* ======================================================
       INPUT BOXES
       ====================================================== */

    input {
        color: #ffffff !important;
    }

    textarea {
        color: #ffffff !important;
    }

    /* ======================================================
       SELECTBOX
       ====================================================== */

    [data-baseweb="select"] {
        color: #ffffff !important;
    }

    [data-baseweb="select"] * {
        color: #ffffff !important;
    }

    /* ======================================================
       RADIO BUTTONS
       ====================================================== */

    [data-testid="stRadio"] label {
        color: #526c76 !important;
    }

    [data-testid="stRadio"] p {
        color: #526c76 !important;
    }

    /* ======================================================
       DATAFRAME
       ====================================================== */

    [data-testid="stDataFrame"] {
        background: #ffffff !important;
    }

    /* ======================================================
       PROGRESS BAR
       ====================================================== */

    [data-testid="stProgressBar"] {
        margin-top: 10px;
        margin-bottom: 15px;
    }

    /* ======================================================
       CAPTIONS
       ====================================================== */

    [data-testid="stCaptionContainer"] {
        color: #72858d !important;
    }

    [data-testid="stCaptionContainer"] p {
        color: #72858d !important;
    }

    /* ======================================================
       FOOTER
       ====================================================== */

    .hab-footer {
        color: #788991 !important;
        font-size: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "page": "Home",
    "location_name": "",
    "latitude": None,
    "longitude": None,
    "radius": 10,
    "waterbody_type": "Auto-detect",
    "waterbodies": [],
    "selected_waterbody": None,
    "selected_geometry": None,
    "analysis": None,
    "history": None,
    "historical_results": None,
    "historical_error": None,
    "search_error": None,
    "forecast_result": None,
    "forecast_error": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# NAVIGATION
# ============================================================

nav_cols = st.columns(7)

pages = [
    ("Home", "Home"),
    ("Waterbodies", "Waterbodies"),
    ("Analysis", "Analysis"),
    ("Historical", "Historical"),
    ("Future Prediction", "Future Prediction"),
    ("Alerts", "Alerts"),
    ("Methodology", "Methodology"),
]

for col, (label, value) in zip(nav_cols, pages):
    with col:
        if st.button(label, use_container_width=True):
            st.session_state.page = value
            st.rerun()


# ============================================================
# HELPERS
# ============================================================

def clean_name(name):
    if not name:
        return "Unnamed waterbody"

    name = str(name).strip()

    if not name:
        return "Unnamed waterbody"

    return name


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dl / 2) ** 2
    )

    return 2 * R * math.asin(math.sqrt(a))


# ============================================================
# GEOCODING
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def geocode_place(place):

    url = "https://nominatim.openstreetmap.org/search"

    params = {
        "q": place,
        "format": "json",
        "limit": 1,
    }

    headers = {
        "User-Agent": "HAB-WATCH/1.0 research project"
    }

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    if not data:
        raise ValueError(
            f"Could not find location: {place}"
        )

    return {
        "display_name": data[0]["display_name"],
        "latitude": float(data[0]["lat"]),
        "longitude": float(data[0]["lon"]),
    }


# ============================================================
# OVERPASS
# ============================================================

OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]


def overpass_request(query):

    headers = {
        "User-Agent": "HAB-WATCH/1.0"
    }

    last_error = None

    for server in OVERPASS_SERVERS:

        try:

            response = requests.post(
                server,
                data=query,
                headers=headers,
                timeout=90
            )

            if response.status_code == 200:
                return response.json()

            last_error = (
                f"{server}: HTTP {response.status_code}"
            )

        except Exception as exc:
            last_error = str(exc)

    raise RuntimeError(
        "All OpenStreetMap Overpass servers failed. "
        + str(last_error)
    )


@st.cache_data(ttl=600, show_spinner=False)
def discover_waterbodies(
    latitude,
    longitude,
    radius_km,
    waterbody_type
):

    radius_m = int(radius_km * 1000)

    if waterbody_type == "Lake":
        type_filter = '["natural"="water"]["water"="lake"]'
    elif waterbody_type == "Reservoir":
        type_filter = '["landuse"="reservoir"]'
    elif waterbody_type == "River":
        type_filter = '["waterway"="river"]'
    else:
        type_filter = ""

    if type_filter:

        query = f"""
        [out:json][timeout:60];

        (
          way{type_filter}(around:{radius_m},{latitude},{longitude});
          relation{type_filter}(around:{radius_m},{latitude},{longitude});
        );

        out center tags;
        """

    else:

        query = f"""
        [out:json][timeout:60];

        (
          way["natural"="water"](around:{radius_m},{latitude},{longitude});
          relation["natural"="water"](around:{radius_m},{latitude},{longitude});

          way["landuse"="reservoir"](around:{radius_m},{latitude},{longitude});
          relation["landuse"="reservoir"](around:{radius_m},{latitude},{longitude});

          way["waterway"="river"](around:{radius_m},{latitude},{longitude});
          relation["waterway"="river"](around:{radius_m},{latitude},{longitude});
        );

        out center tags;
        """

    data = overpass_request(query)

    results = []

    for element in data.get("elements", []):

        tags = element.get("tags", {})

        center = element.get("center")

        if not center:
            continue

        lat = center.get("lat")
        lon = center.get("lon")

        if lat is None or lon is None:
            continue

        name = clean_name(tags.get("name"))

        if tags.get("waterway") == "river":
            wtype = "River"
        elif tags.get("landuse") == "reservoir":
            wtype = "Reservoir"
        elif tags.get("natural") == "water":
            wtype = tags.get("water", "Waterbody").title()
        else:
            wtype = "Waterbody"

        distance = haversine_km(
            latitude,
            longitude,
            lat,
            lon
        )

        results.append(
            {
                "name": name,
                "type": wtype,
                "latitude": lat,
                "longitude": lon,
                "distance_km": distance,
                "osm_type": element.get("type"),
                "osm_id": element.get("id"),
                "tags": tags,
            }
        )

    # Remove duplicates
    unique = {}

    for item in results:

        key = (
            item["osm_type"],
            item["osm_id"]
        )

        unique[key] = item

    results = list(unique.values())

    # Sort by distance
    results.sort(
        key=lambda x: x["distance_km"]
    )

    # Prefer named waterbodies first
    results.sort(
        key=lambda x: (
            x["name"] == "Unnamed waterbody",
            x["distance_km"]
        )
    )

    return results[:50]


# ============================================================
# GET OSM GEOMETRY
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_osm_geometry(osm_type, osm_id):
    """Get selected OSM geometry, with a direct OSM API fallback.

    Overpass is used first. If Overpass is temporarily unavailable (for
    example HTTP 504), an already-selected OSM way can still be retrieved
    directly from the official OSM API, so analysis does not fail merely
    because the discovery service is busy.
    """
    from shapely.geometry import LineString, Polygon
    from shapely.ops import unary_union, polygonize

    if osm_type not in {"way", "relation"}:
        raise ValueError("Unsupported OSM element.")

    # ---------- Try Overpass first ----------
    try:
        query = f"""
        [out:json][timeout:60];
        {osm_type}({int(osm_id)});
        out body geom;
        """
        data = overpass_request(query)
        elements = data.get("elements", [])

        if elements:
            element = elements[0]

            if osm_type == "way":
                coords = [
                    [p["lon"], p["lat"]]
                    for p in element.get("geometry", [])
                ]

                if len(coords) >= 2:
                    if coords[0] == coords[-1] and len(coords) >= 4:
                        geom = Polygon(coords)
                        if not geom.is_valid:
                            geom = geom.buffer(0)
                        return geom
                    return LineString(coords)

            else:
                lines = []
                for member in element.get("members", []):
                    if member.get("type") != "way":
                        continue
                    coords = [
                        (p["lon"], p["lat"])
                        for p in member.get("geometry", [])
                    ]
                    if len(coords) >= 2:
                        lines.append(LineString(coords))

                if lines:
                    polygons = list(polygonize(lines))
                    if polygons:
                        return unary_union(polygons)
                    return unary_union(lines)

    except Exception:
        # Fall through to the direct OSM API below.
        pass

    # ---------- Direct OSM API fallback for ways ----------
    if osm_type == "way":
        url = (
            "https://api.openstreetmap.org/api/0.6/way/"
            f"{int(osm_id)}/full.json"
        )

        response = requests.get(
            url,
            headers={"User-Agent": "HAB-WATCH/1.0"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        nodes = {
            int(el["id"]): (float(el["lon"]), float(el["lat"]))
            for el in data.get("elements", [])
            if el.get("type") == "node"
            and "lon" in el and "lat" in el
        }

        way = next(
            (
                el for el in data.get("elements", [])
                if el.get("type") == "way"
                and int(el.get("id", -1)) == int(osm_id)
            ),
            None,
        )

        if way is None:
            raise ValueError(
                f"OSM way {osm_id} was not returned by the OSM API."
            )

        coords = [
            nodes[node_id]
            for node_id in way.get("nodes", [])
            if node_id in nodes
        ]

        if len(coords) < 2:
            raise ValueError(
                f"OSM way {osm_id} has insufficient geometry."
            )

        if coords[0] == coords[-1] and len(coords) >= 4:
            geom = Polygon(coords)
            if not geom.is_valid:
                geom = geom.buffer(0)
            return geom

        return LineString(coords)

    raise RuntimeError(
        f"Could not retrieve OSM relation {osm_id}. "
        "Overpass is temporarily unavailable and relation geometry "
        "requires member reconstruction."
    )


# ============================================================
# ANALYSIS GEOMETRY
# ============================================================

def prepare_analysis_geometry(geom):

    from shapely.geometry import Polygon, MultiPolygon, LineString

    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom

    if isinstance(geom, LineString):

        project_to_m = pyproj.Transformer.from_crs(
            "EPSG:4326",
            "EPSG:3857",
            always_xy=True
        ).transform

        project_to_deg = pyproj.Transformer.from_crs(
            "EPSG:3857",
            "EPSG:4326",
            always_xy=True
        ).transform

        metric_geom = transform(
            project_to_m,
            geom
        )

        buffered = metric_geom.buffer(100)

        return transform(
            project_to_deg,
            buffered
        )

    raise ValueError(
        "Unsupported waterbody geometry."
    )


# ============================================================
# IMPORT GEE UTILS
# ============================================================

@st.cache_resource
def load_gee_utils():

    path = ROOT / "utils" / "gee_utils.py"

    spec = importlib.util.spec_from_file_location(
        "hab_gee_utils",
        str(path)
    )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


# ============================================================
# GEE IMAGE ACQUISITION
# ============================================================

def acquire_sentinel_image(
    geometry_geojson,
    waterbody_name
):

    """Acquire a real 14-channel Sentinel-2 image for the selected waterbody.

    Important: the model must receive the exact 14 channels created by
    utils.gee_utils.create_14_channel_image().  Passing the raw Sentinel-2
    image directly can produce a TIFF with the wrong number of bands.
    """

    import ee

    gee_utils = load_gee_utils()
    gee_utils.initialize_gee()

    # Direct ee.Geometry construction supports Polygon and MultiPolygon.
    geometry = ee.Geometry(geometry_geojson)

    # Prefer the most recent usable observation.  Fall back year by year.
    search_windows = [
        ("2026-01-01", "2027-01-01"),
        ("2025-01-01", "2026-01-01"),
        ("2024-01-01", "2025-01-01"),
    ]

    collection = None
    count = 0

    for start_date, end_date in search_windows:
        candidate = gee_utils.get_masked_sentinel2_collection(
            geometry,
            start_date,
            end_date,
            cloud_threshold=40
        )
        candidate_count = candidate.size().getInfo()
        if candidate_count > 0:
            collection = candidate
            count = candidate_count
            break

    if collection is None or count == 0:
        raise RuntimeError(
            "No usable Sentinel-2 imagery was found for this waterbody "
            "after the <40% cloud-probability filter."
        )

    # Most recent usable observation in the selected window.
    image = ee.Image(
        collection.sort("system:time_start", False).first()
    )

    selected_date = ee.Date(
        image.get("system:time_start")
    ).format("YYYY-MM-dd").getInfo()

    system_index = image.get("system:index").getInfo()

    safe_name = (
        str(waterbody_name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    output_path = (
        RAW_GENERIC
        / f"{safe_name}_{selected_date}_14Channel.tif"
    )

    # Always make the model-ready 14-band image explicitly.
    image_14 = gee_utils.create_14_channel_image(image)

    needs_download = True

    if output_path.exists():
        try:
            with rasterio.open(output_path) as src:
                valid_existing = (
                    src.count == 14
                    and src.width > 0
                    and src.height > 0
                )
            if valid_existing:
                needs_download = False
            else:
                output_path.unlink()
        except Exception:
            try:
                output_path.unlink()
            except Exception:
                pass

    if needs_download:
        gee_utils.download_14_channel_image(
            image_14,
            geometry,
            str(output_path),
            scale=10
        )

    # Hard validation before anything reaches Swin.
    with rasterio.open(output_path) as src:
        channel_count = src.count
        height = src.height
        width = src.width

    if channel_count != 14:
        try:
            output_path.unlink()
        except Exception:
            pass
        raise RuntimeError(
            f"Sentinel-2 model input validation failed: downloaded TIFF has "
            f"{channel_count} channels; expected exactly 14."
        )

    if height < 1 or width < 1:
        raise RuntimeError("Downloaded Sentinel-2 TIFF has invalid dimensions.")

    return {
        "path": str(output_path),
        "date": selected_date,
        "system_index": system_index,
        "channels": channel_count,
        "height": height,
        "width": width,
    }


# ============================================================
# CREATE WATER MASK
# ============================================================

def create_water_mask(
    geometry,
    image_path,
    output_path
):

    with rasterio.open(image_path) as src:

        height = src.height
        width = src.width
        transform_affine = src.transform
        crs = src.crs

        gdf = gpd.GeoDataFrame(
            geometry=[geometry],
            crs="EPSG:4326"
        )

        gdf = gdf.to_crs(crs)

        mask = rasterize(
            [
                (geom, 1)
                for geom in gdf.geometry
                if geom is not None and not geom.is_empty
            ],
            out_shape=(height, width),
            transform=transform_affine,
            fill=0,
            dtype="uint8"
        )

        profile = src.profile.copy()

        profile.update(
            count=1,
            dtype="uint8",
            nodata=0,
            compress="lzw"
        )

    with rasterio.open(
        output_path,
        "w",
        **profile
    ) as dst:

        dst.write(mask, 1)

    return mask


# ============================================================
# LOAD SWIN INFERENCE
# ============================================================

@st.cache_resource
def load_inference_module():

    path = (
        ROOT
        / "scripts"
        / "89_generic_waterbody_inference.py"
    )

    if not path.exists():
        raise FileNotFoundError(
            "scripts/89_generic_waterbody_inference.py "
            "was not found."
        )

    spec = importlib.util.spec_from_file_location(
        "generic_inference",
        str(path)
    )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


# ============================================================
# RUN SWIN
# ============================================================

def _calculate_pixel_area_m2(transform_affine, crs):

    if crs is not None and crs.is_projected:
        return abs(transform_affine.a * transform_affine.e)

    geod = pyproj.Geod(ellps="WGS84")
    x0 = transform_affine.c
    y0 = transform_affine.f
    x1 = x0 + transform_affine.a
    y1 = y0 + transform_affine.e

    # pyproj.Geod.polygon_area_perimeter() returns exactly TWO values:
    # (area_m2, perimeter_m). The previous UI backend incorrectly tried
    # to unpack three values, which caused:
    # "not enough values to unpack (expected 3, got 2)"
    area, _ = geod.polygon_area_perimeter(
        [x0, x1, x1, x0],
        [y0, y0, y1, y1]
    )
    return abs(area)


def _stretch_rgb(image):

    """Create a readable Sentinel-2 true-colour image.

    Input order is B2, B3, B4, ... so RGB is B4/B3/B2.
    Percentile stretching avoids the almost-black display produced by
    global min/max scaling on reflectance data.
    """

    if image.shape[0] < 3:
        return None

    channels = [
        image[2].astype(np.float32),  # B4 red
        image[1].astype(np.float32),  # B3 green
        image[0].astype(np.float32),  # B2 blue
    ]

    output = np.zeros(
        (image.shape[1], image.shape[2], 3),
        dtype=np.float32
    )

    for i, band in enumerate(channels):
        finite = np.isfinite(band)
        if not finite.any():
            continue

        lo, hi = np.nanpercentile(
            band[finite],
            [2, 98]
        )

        if hi <= lo:
            output[:, :, i] = np.clip(band, 0, None)
            continue

        output[:, :, i] = np.clip(
            (band - lo) / (hi - lo),
            0,
            1
        )

    return output


def run_swin_prediction(
    image_path,
    mask_path,
    waterbody_name,
    date
):

    module = load_inference_module()

    safe_name = (
        waterbody_name
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    prediction_path = (
        PRED_GENERIC
        / f"{safe_name}_{date}_HAB_prediction.tif"
    )

    prediction, water_mask = module.run_waterbody_inference(
        image_path=str(image_path),
        water_mask_path=str(mask_path),
        output_prediction_path=str(prediction_path)
    )

    with rasterio.open(image_path) as src:
        image = src.read()
        transform_affine = src.transform
        crs = src.crs
        height = src.height
        width = src.width

    with rasterio.open(mask_path) as mask_src:
        mask = mask_src.read(1)

    valid_water = mask == 1
    prediction = np.asarray(prediction)

    if prediction.shape != (height, width):
        from PIL import Image
        pred_img = Image.fromarray(prediction.astype(np.uint8))
        pred_img = pred_img.resize(
            (width, height),
            Image.Resampling.NEAREST
        )
        prediction = np.asarray(pred_img)

    prediction = np.where(
        np.isin(prediction, [0, 1]),
        prediction,
        0
    ).astype(np.uint8)

    hab = (prediction == 1) & valid_water

    water_pixels = int(valid_water.sum())
    hab_pixels = int(hab.sum())
    non_hab_pixels = max(water_pixels - hab_pixels, 0)

    pixel_area_m2 = _calculate_pixel_area_m2(
        transform_affine,
        crs
    )

    water_area_ha = water_pixels * pixel_area_m2 / 10000.0
    hab_area_ha = hab_pixels * pixel_area_m2 / 10000.0
    non_hab_area_ha = non_hab_pixels * pixel_area_m2 / 10000.0

    hab_coverage = (
        100.0 * hab_pixels / water_pixels
        if water_pixels > 0
        else 0.0
    )

    rgb = _stretch_rgb(image)

    return {
        "prediction": prediction,
        "water_mask": mask,
        "hab_mask": hab.astype(np.uint8),
        "rgb": rgb,
        "water_pixels": water_pixels,
        "hab_pixels": hab_pixels,
        "non_hab_pixels": non_hab_pixels,
        "water_area_ha": water_area_ha,
        "hab_area_ha": hab_area_ha,
        "non_hab_area_ha": non_hab_area_ha,
        "hab_coverage": hab_coverage,
        "prediction_path": str(prediction_path),
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "date": date,
    }


# ============================================================
# COMPLETE ANALYSIS PIPELINE
# ============================================================

def perform_analysis(waterbody):

    name = clean_name(
        waterbody["name"]
    )

    osm_type = waterbody["osm_type"]
    osm_id = waterbody["osm_id"]

    # 1. Geometry
    geom = get_osm_geometry(
        osm_type,
        osm_id
    )

    analysis_geom = prepare_analysis_geometry(
        geom
    )

    geometry_geojson = mapping(
        analysis_geom
    )

    # 2. Sentinel-2
    sentinel = acquire_sentinel_image(
        geometry_geojson,
        name
    )

    image_path = sentinel["path"]
    date = sentinel["date"]

    # 3. Water mask
    safe_name = (
        name
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    mask_path = (
        MASK_GENERIC
        / f"{safe_name}_water_mask_{date}.tif"
    )

    if mask_path.exists():

        mask = rasterio.open(
            mask_path
        ).read(1)

    else:

        mask = create_water_mask(
            analysis_geom,
            image_path,
            str(mask_path)
        )

    # 4. Swin
    analysis = run_swin_prediction(
        image_path,
        str(mask_path),
        name,
        date
    )

    analysis.update(
        {
            "name": name,
            "type": waterbody["type"],
            "latitude": waterbody["latitude"],
            "longitude": waterbody["longitude"],
            "distance_km": waterbody[
                "distance_km"
            ],
            "geometry": analysis_geom,
            "osm_id": osm_id,
            "osm_type": osm_type,
            "system_index": sentinel[
                "system_index"
            ],
        }
    )

    return analysis


# ============================================================
# DISPLAY HELPERS
# ============================================================

def metric_card(label, value):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                {label}
            </div>
            <div class="metric-value">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_rgb(image):

    if image is None:
        st.warning("RGB visualization unavailable.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title("Sentinel-2 True Colour Composite (B4/B3/B2)")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def show_heatmap(analysis):

    prediction = analysis[
        "prediction"
    ]

    water_mask = analysis[
        "water_mask"
    ]

    display = np.full(
        prediction.shape,
        np.nan
    )

    display[
        water_mask == 1
    ] = prediction[
        water_mask == 1
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    im = ax.imshow(
        display,
        interpolation="nearest"
    )

    ax.set_title(
        "Swin Transformer HAB Detection"
    )

    ax.axis("off")

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

    cbar.set_ticks(
        [0, 1]
    )

    cbar.set_ticklabels(
        ["Non-HAB", "HAB"]
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


def show_waterbody_map(analysis):

    lat = analysis["latitude"]
    lon = analysis["longitude"]

    m = folium.Map(
        location=[lat, lon],
        zoom_start=12,
        tiles="OpenStreetMap"
    )

    geom = analysis["geometry"]

    folium.GeoJson(
        mapping(geom),
        name="Selected Waterbody",
        style_function=lambda x: {
            "fillColor": "#4f908a",
            "color": "#245d61",
            "weight": 3,
            "fillOpacity": 0.25,
        }
    ).add_to(m)

    folium.Marker(
        [lat, lon],
        tooltip=analysis["name"],
        popup=analysis["name"]
    ).add_to(m)

    st_folium(
        m,
        use_container_width=True,
        height=500
    )


# ============================================================
# HOME
# ============================================================

def render_home():

    st.html(
        """
        <div class="hero">
            <h1 style="color:#ffffff !important;">Protect Our Waters</h1>
            <p style="color:#f4fbfb !important;">Detect. Monitor. Understand.</p>
            <p style="color:#f4fbfb !important;">
                AI-powered monitoring of harmful algal blooms
                using Sentinel-2 satellite imagery, Swin Transformer
                segmentation and temporal deep learning.
            </p>
        </div>
        """
    )

    st.markdown(
        '<div class="card-title">🔎 Search for a Waterbody</div>',
        unsafe_allow_html=True
    )

    method = st.radio(
        "Search method",
        [
            "Search by place",
            "Enter coordinates"
        ],
        horizontal=True
    )

    if method == "Search by place":

        place = st.text_input(
            "Location",
            value=st.session_state.location_name,
            placeholder="Example: Varanasi, Hyderabad, Mumbai"
        )

        if place.strip():

            if (
                place.strip()
                != st.session_state.location_name
            ):

                try:

                    with st.spinner(
                        "Finding location..."
                    ):

                        result = geocode_place(
                            place.strip()
                        )

                    st.session_state.location_name = (
                        result["display_name"]
                    )

                    st.session_state.latitude = (
                        result["latitude"]
                    )

                    st.session_state.longitude = (
                        result["longitude"]
                    )

                except Exception as exc:

                    st.error(
                        f"Location could not be found: {exc}"
                    )

    else:

        c1, c2 = st.columns(2)

        with c1:

            lat = st.number_input(
                "Latitude",
                value=(
                    float(
                        st.session_state.latitude
                    )
                    if st.session_state.latitude
                    is not None
                    else 17.4222
                ),
                format="%.6f"
            )

        with c2:

            lon = st.number_input(
                "Longitude",
                value=(
                    float(
                        st.session_state.longitude
                    )
                    if st.session_state.longitude
                    is not None
                    else 78.4739
                ),
                format="%.6f"
            )

        st.session_state.latitude = lat
        st.session_state.longitude = lon

    if (
        st.session_state.latitude is not None
        and st.session_state.longitude is not None
    ):

        c1, c2 = st.columns(2)

        with c1:

            st.number_input(
                "Latitude",
                value=float(
                    st.session_state.latitude
                ),
                disabled=True,
                format="%.6f"
            )

        with c2:

            st.number_input(
                "Longitude",
                value=float(
                    st.session_state.longitude
                ),
                disabled=True,
                format="%.6f"
            )

    c1, c2 = st.columns(2)

    with c1:

        radius = st.selectbox(
            "Search radius",
            [1, 2, 5, 10, 20, 30, 50],
            index=3
        )

    with c2:

        water_type = st.selectbox(
            "Waterbody type",
            [
                "Auto-detect",
                "Lake",
                "Reservoir",
                "River"
            ]
        )

    st.markdown("")

    if st.button(
        "🌊 Find Waterbodies →",
        use_container_width=True
    ):

        if (
            st.session_state.latitude
            is None
            or st.session_state.longitude
            is None
        ):

            st.error(
                "Please enter a place or coordinates first."
            )

        else:

            try:

                with st.spinner(
                    "Searching OpenStreetMap for nearby waterbodies..."
                ):

                    results = discover_waterbodies(
                        st.session_state.latitude,
                        st.session_state.longitude,
                        radius,
                        water_type
                    )

                st.session_state.radius = radius
                st.session_state.waterbody_type = (
                    water_type
                )
                st.session_state.waterbodies = (
                    results
                )
                st.session_state.page = (
                    "Waterbodies"
                )

                st.rerun()

            except Exception as exc:

                st.error(
                    "Waterbody discovery failed."
                )

                st.warning(
                    "The OpenStreetMap Overpass service "
                    "may be temporarily busy. Try again "
                    "in a few seconds."
                )


# ============================================================
# WATERBODIES
# ============================================================

def render_waterbodies():

    st.title("Nearby Waterbodies")

    if not st.session_state.waterbodies:

        st.info(
            "Search for a location first."
        )

        if st.button("← Back to Search"):
            st.session_state.page = "Home"
            st.rerun()

        return

    lat = st.session_state.latitude
    lon = st.session_state.longitude

    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">
                Search Area
            </div>
            <div class="small-muted">
                Latitude: {lat:.6f}
                &nbsp;&nbsp;|&nbsp;&nbsp;
                Longitude: {lon:.6f}
                &nbsp;&nbsp;|&nbsp;&nbsp;
                Radius: {st.session_state.radius} km
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    left, right = st.columns(
        [0.85, 1.7]
    )

    with left:

        st.subheader(
            f"{len(st.session_state.waterbodies)} "
            "waterbodies found"
        )

        for idx, waterbody in enumerate(
            st.session_state.waterbodies
        ):

            name = waterbody["name"]

            st.markdown(
                f"""
                <div class="waterbody-card">
                    <div class="waterbody-name">
                        {name}
                    </div>
                    <div class="small-muted">
                        {waterbody["type"]}
                        &nbsp;•&nbsp;
                        {waterbody["distance_km"]:.2f} km
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "Select",
                key=f"select_{idx}",
                use_container_width=True
            ):

                st.session_state.selected_waterbody = (
                    waterbody
                )

                st.session_state.page = (
                    "Analysis"
                )

                st.session_state.analysis = None
                st.session_state.historical_results = None
                st.session_state.historical_error = None

                st.rerun()

    with right:

        m = folium.Map(
            location=[lat, lon],
            zoom_start=11,
            tiles="OpenStreetMap"
        )

        folium.Marker(
            [lat, lon],
            tooltip="Search location",
            popup="Search location"
        ).add_to(m)

        folium.Circle(
            [lat, lon],
            radius=st.session_state.radius * 1000,
            color="#356f70",
            fill=False
        ).add_to(m)

        for idx, waterbody in enumerate(
            st.session_state.waterbodies
        ):

            color = (
                "#e85c4a"
                if waterbody["name"]
                != "Unnamed waterbody"
                else "#7b8790"
            )

            folium.CircleMarker(
                [
                    waterbody["latitude"],
                    waterbody["longitude"]
                ],
                radius=7,
                color=color,
                fill=True,
                fill_opacity=0.85,
                tooltip=waterbody["name"],
                popup=waterbody["name"]
            ).add_to(m)

        st_folium(
            m,
            use_container_width=True,
            height=620
        )


# ============================================================
# ANALYSIS
# ============================================================

def render_analysis():

    selected = (
        st.session_state.selected_waterbody
    )

    if selected is None:

        st.info(
            "Select a waterbody from the Waterbodies page."
        )

        return

    name = selected["name"]

    st.title(
        f"HAB Analysis — {name}"
    )

    if st.session_state.analysis is None:

        st.markdown(
            f"""
            <div class="info-box">
                <b>{name}</b> has been selected.
                The next step uses the actual backend:
                OSM geometry → Sentinel-2 → water mask →
                Swin Transformer → HAB detection.
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "🚀 Run HAB Detection",
            use_container_width=True
        ):

            try:

                progress = st.progress(0)

                status = st.empty()

                status.info(
                    "Getting actual waterbody geometry..."
                )

                progress.progress(10)

                status.info(
                    "Acquiring Sentinel-2 imagery..."
                )

                progress.progress(30)

                status.info(
                    "Creating water mask..."
                )

                progress.progress(50)

                status.info(
                    "Running Swin Transformer..."
                )

                progress.progress(75)

                analysis = perform_analysis(
                    selected
                )

                progress.progress(100)

                st.session_state.analysis = (
                    analysis
                )

                status.success(
                    "HAB detection completed."
                )

                st.rerun()

            except Exception as exc:

                st.error(
                    "The analysis could not be completed."
                )

                with st.expander(
                    "Technical details"
                ):
                    st.code(
                        str(exc)
                    )

        return

    analysis = st.session_state.analysis

    st.markdown(
        f"""
        <div class="success-box">
            <b>Analysis completed successfully.</b><br>
            Sentinel-2 observation:
            {analysis["date"]}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Basic information
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Waterbody",
            analysis["name"]
        )

    with c2:
        metric_card(
            "Observation",
            analysis["date"]
        )

    with c3:
        metric_card(
            "Water Area",
            f'{analysis["water_area_ha"]:.2f} ha'
        )

    with c4:
        metric_card(
            "Predicted HAB",
            f'{analysis["hab_area_ha"]:.2f} ha'
        )

    st.markdown("")

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "HAB Coverage",
            f'{analysis["hab_coverage"]:.2f}%'
        )

    with c2:
        metric_card(
            "Non-HAB Area",
            f'{analysis["non_hab_area_ha"]:.2f} ha'
        )

    with c3:
        metric_card(
            "HAB Pixels",
            f'{analysis["hab_pixels"]:,}'
        )

    st.markdown("")

    if analysis["water_pixels"] < 100:

        st.warning(
            f"This waterbody contains only {analysis["water_pixels"]} "
            "Sentinel-2 pixels at 10 m resolution. HAB predictions for "
            "very small waterbodies should be interpreted cautiously."
        )

    # Actual imagery and prediction
    left, right = st.columns(2)

    with left:

        st.markdown(
            '<div class="card-title">🛰️ Sentinel-2 Image</div>',
            unsafe_allow_html=True
        )

        show_rgb(
            analysis["rgb"]
        )

    with right:

        st.markdown(
            '<div class="card-title">🔥 HAB Heatmap</div>',
            unsafe_allow_html=True
        )

        show_heatmap(
            analysis
        )

    # Map
    st.markdown(
        '<div class="card-title">🗺️ Waterbody Location</div>',
        unsafe_allow_html=True
    )

    show_waterbody_map(
        analysis
    )

    # Summary table
    st.markdown(
        '<div class="card-title">📊 Detection Summary</div>',
        unsafe_allow_html=True
    )

    summary = pd.DataFrame(
        [
            {
                "Parameter": "Water Area",
                "Value": f'{analysis["water_area_ha"]:.2f} ha'
            },
            {
                "Parameter": "Predicted HAB Area",
                "Value": f'{analysis["hab_area_ha"]:.2f} ha'
            },
            {
                "Parameter": "Non-HAB Area",
                "Value": f'{analysis["non_hab_area_ha"]:.2f} ha'
            },
            {
                "Parameter": "HAB Coverage",
                "Value": f'{analysis["hab_coverage"]:.2f}%'
            },
            {
                "Parameter": "Water Pixels",
                "Value": f'{analysis["water_pixels"]:,}'
            },
            {
                "Parameter": "HAB Pixels",
                "Value": f'{analysis["hab_pixels"]:,}'
            },
            {
                "Parameter": "Sentinel-2 Date",
                "Value": analysis["date"]
            },
        ]
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The HAB percentage is the Swin Transformer's "
        "predicted HAB coverage within the detected waterbody. "
        "It is not field-validated ground truth."
    )

    if st.button(
        "← Choose Another Waterbody"
    ):

        st.session_state.analysis = None
        st.session_state.historical_results = None
        st.session_state.historical_error = None
        st.session_state.page = "Waterbodies"

        st.rerun()


# ============================================================
# HISTORICAL
# ============================================================

def _safe_waterbody_name(name):
    return (
        clean_name(name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def _parse_date_from_filename(path):
    match = re.search(r"(20\d{2}-\d{2}-\d{2})", path.name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d").date()
    except ValueError:
        return None


def _local_historical_image(waterbody_name, year):
    """Return an already downloaded 14-channel image when the project has one.

    The multi-waterbody training dataset contains real Sentinel-2 images for
    Hussain Sagar, Saroor Nagar, Osman Sagar and Himayat Sagar for 2016-2025.
    Reusing those files makes the UI historical page fast and reproducible.
    """
    if not MULTI_WATERBODY_RAW.exists():
        return None

    name = clean_name(waterbody_name).lower().replace("-", " ")
    aliases = {
        "hussain sagar": "Hussain_Sagar",
        "saroor nagar": "Saroor_Nagar",
        "osman sagar": "Osman_Sagar",
        "himayat sagar": "Himayat_Sagar",
    }

    folder = None
    for key, value in aliases.items():
        if key in name:
            candidate = MULTI_WATERBODY_RAW / value
            if candidate.exists():
                folder = candidate
                break

    if folder is None:
        # Generic fallback: compare normalized folder names with the waterbody.
        normalized_name = re.sub(r"[^a-z0-9]", "", name)
        best = None
        best_score = 0
        for candidate in MULTI_WATERBODY_RAW.iterdir():
            if not candidate.is_dir():
                continue
            normalized_folder = re.sub(r"[^a-z0-9]", "", candidate.name.lower())
            score = 0
            if normalized_folder in normalized_name or normalized_name in normalized_folder:
                score = min(len(normalized_folder), len(normalized_name))
            else:
                name_tokens = set(re.findall(r"[a-z]+", name))
                folder_tokens = set(re.findall(r"[a-z]+", candidate.name.lower()))
                score = len(name_tokens & folder_tokens) * 10
            if score > best_score:
                best_score = score
                best = candidate
        folder = best

    if folder is None:
        return None

    candidates = []
    for path in folder.glob("*.tif"):
        d = _parse_date_from_filename(path)
        if d is None or d.year != year:
            continue
        try:
            with rasterio.open(path) as src:
                if src.count != 14:
                    continue
        except Exception:
            continue
        candidates.append((d, path))

    if not candidates:
        return None

    target = date(year, 6, 30)
    selected_date, selected_path = min(
        candidates,
        key=lambda item: abs((item[0] - target).days)
    )
    return str(selected_path), selected_date.strftime("%Y-%m-%d"), "Local project dataset"


def _download_one_historical_year(gee_utils, geometry, waterbody_name, year):
    """Find and download one real cloud-filtered Sentinel-2 observation for a year."""
    import ee

    collection = gee_utils.get_masked_sentinel2_collection(
        geometry,
        f"{year}-01-01",
        f"{year + 1}-01-01",
        cloud_threshold=40
    )

    dates = gee_utils.get_available_dates(collection)
    if not dates:
        return None, None, "No usable Sentinel-2 observation after the <40% cloud filter."

    parsed = []
    for d in dates:
        try:
            parsed.append((str(d), datetime.strptime(str(d), "%Y-%m-%d").date()))
        except ValueError:
            continue

    if not parsed:
        return None, None, "Usable observations existed, but dates could not be parsed."

    target = date(year, 6, 30)
    selected_date = min(
        parsed,
        key=lambda item: abs((item[1] - target).days)
    )[0]

    safe_name = _safe_waterbody_name(waterbody_name)
    out_dir = RAW_GENERIC / "historical" / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{safe_name}_{selected_date}_14Channel.tif"

    if output_path.exists():
        try:
            with rasterio.open(output_path) as src:
                if src.count == 14 and src.width > 0 and src.height > 0:
                    return str(output_path), selected_date, "Google Earth Engine (cached TIFF)"
        except Exception:
            pass
        try:
            output_path.unlink()
        except Exception:
            pass

    image = gee_utils.get_image_for_date(collection, selected_date)
    image_14 = gee_utils.create_14_channel_image(image)

    gee_utils.download_14_channel_image(
        image_14,
        geometry,
        str(output_path),
        scale=10
    )

    with rasterio.open(output_path) as src:
        if src.count != 14:
            try:
                output_path.unlink()
            except Exception:
                pass
            raise RuntimeError(
                f"Historical image {selected_date} has {src.count} channels; expected 14."
            )

    return str(output_path), selected_date, "Google Earth Engine"


def _load_historical_prediction_result(image_path, mask_path, prediction_path, name, selected_date):
    """Read an already-computed prediction so reruns do not repeat CPU inference."""
    with rasterio.open(image_path) as src:
        transform_affine = src.transform
        crs = src.crs
        height = src.height
        width = src.width
        image = src.read()

    with rasterio.open(mask_path) as src:
        water_mask = (src.read(1) > 0).astype(np.uint8)

    with rasterio.open(prediction_path) as src:
        prediction = (src.read(1) > 0).astype(np.uint8)

    if prediction.shape != (height, width) or water_mask.shape != (height, width):
        raise RuntimeError("Cached historical prediction and water mask dimensions do not match the image.")

    valid_water = water_mask == 1
    hab = (prediction == 1) & valid_water
    water_pixels = int(valid_water.sum())
    hab_pixels = int(hab.sum())
    non_hab_pixels = max(water_pixels - hab_pixels, 0)

    pixel_area_m2 = _calculate_pixel_area_m2(transform_affine, crs)
    water_area_ha = water_pixels * pixel_area_m2 / 10000.0
    hab_area_ha = hab_pixels * pixel_area_m2 / 10000.0
    non_hab_area_ha = non_hab_pixels * pixel_area_m2 / 10000.0
    hab_coverage = (hab_pixels / water_pixels * 100.0) if water_pixels else 0.0

    # Proper percentile stretch for B4/B3/B2 display.
    rgb = None
    if image.shape[0] >= 4:
        bands = np.stack([image[2].astype(float), image[1].astype(float), image[0].astype(float)], axis=-1)
        rgb_channels = []
        for channel in range(3):
            values = bands[..., channel]
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                rgb_channels.append(np.zeros_like(values))
                continue
            lo, hi = np.percentile(finite, [2, 98])
            if hi <= lo:
                rgb_channels.append(np.clip(values, 0, 1))
            else:
                rgb_channels.append(np.clip((values - lo) / (hi - lo), 0, 1))
        rgb = np.stack(rgb_channels, axis=-1)

    return {
        "year": int(selected_date[:4]),
        "date": selected_date,
        "status": "Completed",
        "water_area_ha": water_area_ha,
        "hab_area_ha": hab_area_ha,
        "non_hab_area_ha": non_hab_area_ha,
        "hab_coverage": hab_coverage,
        "water_pixels": water_pixels,
        "hab_pixels": hab_pixels,
        "prediction": prediction,
        "water_mask": water_mask,
        "rgb": rgb,
        "image_path": image_path,
        "mask_path": mask_path,
        "prediction_path": prediction_path,
        "source": "Cached prediction",
    }


def run_historical_backend(analysis, progress_bar=None, status_box=None):
    """Run a resumable, real 2016-2026 historical pipeline.

    Existing project images are reused for the four training waterbodies.
    Other waterbodies are queried year-by-year from GEE. Completed prediction
    TIFFs are reused on subsequent runs, so the historical page can resume
    rather than starting all eleven years again.
    """
    import ee

    name = analysis["name"]
    geometry = analysis["geometry"]
    geometry_geojson = mapping(geometry)

    # IMPORTANT: initialize Earth Engine before constructing ee.Geometry.
    # Creating ee objects first can fail with an uninitialized-client error.
    gee_utils = load_gee_utils()
    gee_utils.initialize_gee()
    ee_geometry = ee.Geometry(geometry_geojson)

    safe_name = _safe_waterbody_name(name)
    report_dir = HISTORICAL_UI_DIR / safe_name
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{safe_name}_historical_2016_2026.csv"

    existing = {}
    if report_path.exists():
        try:
            old_df = pd.read_csv(report_path)
            for _, row in old_df.iterrows():
                if str(row.get("Status", "")) == "Completed" and pd.notna(row.get("Sentinel-2 Date")):
                    existing[int(row["Year"])] = row.to_dict()
        except Exception:
            existing = {}

    results = []
    years = list(range(2016, 2027))
    total = len(years)

    for idx, year in enumerate(years, start=1):
        if progress_bar is not None:
            progress_bar.progress((idx - 1) / total)
        if status_box is not None:
            status_box.info(f"Historical analysis: {year} ({idx}/{total})")

        selected_date = None
        image_path = None
        source = None

        # 1. Reuse a completed local prediction if it still exists.
        old = existing.get(year)
        if old is not None:
            selected_date = str(old["Sentinel-2 Date"])
            pred_path = PRED_GENERIC / f"{safe_name}_{selected_date}_HAB_prediction.tif"
            mask_path = MASK_GENERIC / "historical" / safe_name / f"{safe_name}_water_mask_{selected_date}.tif"
            image_path = RAW_GENERIC / "historical" / safe_name / f"{safe_name}_{selected_date}_14Channel.tif"
            if pred_path.exists() and mask_path.exists() and image_path.exists():
                try:
                    cached = _load_historical_prediction_result(
                        str(image_path), str(mask_path), str(pred_path), name, selected_date
                    )
                    cached["year"] = year
                    cached["source"] = "Previously completed UI run"
                    results.append(cached)
                    if status_box is not None:
                        status_box.success(f"{year}: reused completed result")
                    continue
                except Exception:
                    pass

        try:
            # 2. Reuse the real multi-waterbody dataset whenever available.
            local = _local_historical_image(name, year)
            if local is not None:
                image_path, selected_date, source = local
            else:
                # 3. Otherwise query GEE for this single year.
                if status_box is not None:
                    status_box.info(f"{year}: querying Google Earth Engine for a representative image...")
                image_path, selected_date, source = _download_one_historical_year(
                    gee_utils, ee_geometry, name, year
                )

            if image_path is None or selected_date is None:
                results.append({
                    "year": year,
                    "date": None,
                    "status": "Unavailable",
                    "water_area_ha": np.nan,
                    "hab_area_ha": np.nan,
                    "non_hab_area_ha": np.nan,
                    "hab_coverage": np.nan,
                    "water_pixels": 0,
                    "hab_pixels": 0,
                    "prediction": None,
                    "water_mask": None,
                    "rgb": None,
                    "source": source or "No usable observation",
                })
                continue

            mask_dir = MASK_GENERIC / "historical" / safe_name
            mask_dir.mkdir(parents=True, exist_ok=True)
            mask_path = mask_dir / f"{safe_name}_water_mask_{selected_date}.tif"

            if not mask_path.exists():
                create_water_mask(
                    geometry,
                    str(image_path),
                    str(mask_path)
                )

            pred_path = PRED_GENERIC / f"{safe_name}_{selected_date}_HAB_prediction.tif"

            if pred_path.exists():
                try:
                    result = _load_historical_prediction_result(
                        str(image_path), str(mask_path), str(pred_path), name, selected_date
                    )
                    result["source"] = source
                except Exception:
                    result = run_swin_prediction(
                        str(image_path), str(mask_path), name, selected_date
                    )
                    result["source"] = source
            else:
                if status_box is not None:
                    status_box.info(f"{year}: running Swin inference...")
                result = run_swin_prediction(
                    str(image_path), str(mask_path), name, selected_date
                )
                result["source"] = source

            result["year"] = year
            result["date"] = selected_date
            result["status"] = "Completed"
            result["image_path"] = str(image_path)
            result["mask_path"] = str(mask_path)
            result["prediction_path"] = str(pred_path)
            results.append(result)

            # Save after every successful year so a later run can resume.
            report_df = pd.DataFrame([
                {
                    "Year": r["year"],
                    "Sentinel-2 Date": r["date"],
                    "Status": r["status"],
                    "Water Area (ha)": r["water_area_ha"],
                    "Predicted HAB Area (ha)": r["hab_area_ha"],
                    "Predicted HAB Coverage (%)": r["hab_coverage"],
                    "Water Pixels": r["water_pixels"],
                    "HAB Pixels": r["hab_pixels"],
                    "Source": r.get("source", ""),
                }
                for r in results
            ])
            report_df.to_csv(report_path, index=False)

        except Exception as exc:
            results.append({
                "year": year,
                "date": selected_date,
                "status": f"Failed: {exc}",
                "water_area_ha": np.nan,
                "hab_area_ha": np.nan,
                "non_hab_area_ha": np.nan,
                "hab_coverage": np.nan,
                "water_pixels": 0,
                "hab_pixels": 0,
                "prediction": None,
                "water_mask": None,
                "rgb": None,
                "source": source or "",
            })

            # Save failure too; the next run can retry it.
            report_df = pd.DataFrame([
                {
                    "Year": r["year"],
                    "Sentinel-2 Date": r["date"],
                    "Status": r["status"],
                    "Water Area (ha)": r["water_area_ha"],
                    "Predicted HAB Area (ha)": r["hab_area_ha"],
                    "Predicted HAB Coverage (%)": r["hab_coverage"],
                    "Water Pixels": r["water_pixels"],
                    "HAB Pixels": r["hab_pixels"],
                    "Source": r.get("source", ""),
                }
                for r in results
            ])
            report_df.to_csv(report_path, index=False)

    if progress_bar is not None:
        progress_bar.progress(1.0)
    if status_box is not None:
        status_box.success("Historical 2016–2026 processing completed.")

    report_df = pd.DataFrame([
        {
            "Year": r["year"],
            "Sentinel-2 Date": r["date"],
            "Status": r["status"],
            "Water Area (ha)": r["water_area_ha"],
            "Predicted HAB Area (ha)": r["hab_area_ha"],
            "Predicted HAB Coverage (%)": r["hab_coverage"],
            "Water Pixels": r["water_pixels"],
            "HAB Pixels": r["hab_pixels"],
            "Source": r.get("source", ""),
        }
        for r in results
    ])
    report_df.to_csv(report_path, index=False)

    return {
        "results": results,
        "table": report_df,
        "report_path": str(report_path),
    }



def _rebuild_historical_from_disk(name):
    safe_name = _safe_waterbody_name(name)
    report_path = HISTORICAL_UI_DIR / safe_name / f"{safe_name}_historical_2016_2026.csv"
    if not report_path.exists():
        return None

    try:
        df = pd.read_csv(report_path)
    except Exception:
        return None

    results = []
    for _, row in df.iterrows():
        year = int(row["Year"])
        selected_date = None if pd.isna(row.get("Sentinel-2 Date")) else str(row["Sentinel-2 Date"])
        status = str(row.get("Status", ""))
        if status != "Completed" or not selected_date:
            results.append({
                "year": year,
                "date": selected_date,
                "status": status,
                "water_area_ha": row.get("Water Area (ha)", np.nan),
                "hab_area_ha": row.get("Predicted HAB Area (ha)", np.nan),
                "hab_coverage": row.get("Predicted HAB Coverage (%)", np.nan),
                "water_pixels": int(row.get("Water Pixels", 0) or 0),
                "hab_pixels": int(row.get("HAB Pixels", 0) or 0),
                "prediction": None,
                "water_mask": None,
                "rgb": None,
            })
            continue

        image_path = RAW_GENERIC / "historical" / safe_name / f"{safe_name}_{selected_date}_14Channel.tif"
        mask_path = MASK_GENERIC / "historical" / safe_name / f"{safe_name}_water_mask_{selected_date}.tif"
        pred_path = PRED_GENERIC / f"{safe_name}_{selected_date}_HAB_prediction.tif"
        try:
            if image_path.exists() and mask_path.exists() and pred_path.exists():
                results.append(_load_historical_prediction_result(
                    str(image_path), str(mask_path), str(pred_path), name, selected_date
                ))
            else:
                results.append({
                    "year": year,
                    "date": selected_date,
                    "status": "Completed",
                    "water_area_ha": row.get("Water Area (ha)", np.nan),
                    "hab_area_ha": row.get("Predicted HAB Area (ha)", np.nan),
                    "hab_coverage": row.get("Predicted HAB Coverage (%)", np.nan),
                    "water_pixels": int(row.get("Water Pixels", 0) or 0),
                    "hab_pixels": int(row.get("HAB Pixels", 0) or 0),
                    "prediction": None,
                    "water_mask": None,
                    "rgb": None,
                })
        except Exception:
            results.append({
                "year": year,
                "date": selected_date,
                "status": status,
                "water_area_ha": row.get("Water Area (ha)", np.nan),
                "hab_area_ha": row.get("Predicted HAB Area (ha)", np.nan),
                "hab_coverage": row.get("Predicted HAB Coverage (%)", np.nan),
                "water_pixels": int(row.get("Water Pixels", 0) or 0),
                "hab_pixels": int(row.get("HAB Pixels", 0) or 0),
                "prediction": None,
                "water_mask": None,
                "rgb": None,
            })

    return {"results": results, "table": df, "report_path": str(report_path)}


def render_historical():

    st.title("Historical HAB Analysis")

    analysis = st.session_state.analysis
    if analysis is None:
        st.info("Run HAB detection first to view historical analysis for the selected waterbody.")
        return

    name = analysis["name"]
    safe_name = _safe_waterbody_name(name)

    st.markdown(
        f"""
        <div class="info-box">
            <b>Historical analysis for {name}</b><br>
            Real Sentinel-2 observations are processed for each year from
            <b>2016–2026</b>. The application never fills missing years with
            fabricated HAB values.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="warning-box">
            <b>How this page works:</b> one representative cloud-filtered
            Sentinel-2 observation closest to June 30 is selected for each year.
            Existing project images and completed Swin predictions are reused,
            so the process can resume instead of starting from zero.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Automatically restore an existing completed report after navigation/reload.
    if st.session_state.historical_results is None:
        restored = _rebuild_historical_from_disk(name)
        if restored is not None and len(restored["table"]) > 0:
            st.session_state.historical_results = restored

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.session_state.historical_results is None:
            st.info(
                "Click the button below. For Saroor Nagar, Hussain Sagar, Osman Sagar "
                "and Himayat Sagar, the existing real multi-waterbody Sentinel-2 dataset "
                "is reused for 2016–2025. Only missing years are queried from GEE."
            )
        else:
            completed_count = sum(
                r.get("status") == "Completed"
                for r in st.session_state.historical_results["results"]
            )
            st.success(f"Historical results available: {completed_count}/11 years completed.")
    with col2:
        if st.session_state.historical_results is not None:
            if st.button("↻ Re-run / Resume", use_container_width=True):
                st.session_state.historical_results = None
                st.rerun()

    if st.session_state.historical_results is None:
        if st.button("🚀 Run / Resume Real 2016–2026 Historical Analysis", use_container_width=True):
            progress = st.progress(0.0)
            status = st.empty()
            st.session_state.historical_error = None
            try:
                historical = run_historical_backend(
                    analysis,
                    progress_bar=progress,
                    status_box=status
                )
                st.session_state.historical_results = historical
                st.rerun()
            except Exception as exc:
                st.session_state.historical_error = str(exc)
                st.error("Historical backend processing failed.")
                st.code(str(exc))
        return

    historical = st.session_state.historical_results
    df = historical["table"].copy()

    st.success(f"Historical report: `{historical['report_path']}`")

    display_df = df.copy()
    for col in ["Water Area (ha)", "Predicted HAB Area (ha)", "Predicted HAB Coverage (%)"]:
        display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(3)

    st.markdown("### 📊 Year-by-Year Historical Detection")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    valid = display_df[display_df["Status"].eq("Completed")].copy()
    if not valid.empty:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(valid["Year"], valid["Predicted HAB Coverage (%)"], marker="o")
            ax.set_title(f"Predicted HAB Coverage — {name}")
            ax.set_xlabel("Year")
            ax.set_ylabel("HAB coverage (%)")
            ax.set_xticks(valid["Year"])
            ax.grid(alpha=0.25)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(valid["Year"], valid["Predicted HAB Area (ha)"], marker="o")
            ax.set_title(f"Predicted HAB Area — {name}")
            ax.set_xlabel("Year")
            ax.set_ylabel("HAB area (ha)")
            ax.set_xticks(valid["Year"])
            ax.grid(alpha=0.25)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.markdown("### 🛰️ Yearly Sentinel-2 / HAB Results")
    completed = [r for r in historical["results"] if r.get("status") == "Completed"]

    for i in range(0, len(completed), 2):
        cols = st.columns(2)
        for col, result in zip(cols, completed[i:i+2]):
            with col:
                st.markdown(f"#### {result['year']} — {result['date']}")
                left, right = st.columns(2)
                with left:
                    if result.get("rgb") is not None:
                        st.image(result["rgb"], caption="Sentinel-2 true-colour composite", use_container_width=True)
                with right:
                    if result.get("prediction") is not None and result.get("water_mask") is not None:
                        display = np.full(result["prediction"].shape, np.nan)
                        wm = result["water_mask"]
                        display[wm == 1] = result["prediction"][wm == 1]
                        fig, ax = plt.subplots(figsize=(4.5, 3.5))
                        im = ax.imshow(display, interpolation="nearest", vmin=0, vmax=1)
                        ax.set_title("Swin HAB heatmap")
                        ax.axis("off")
                        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        cbar.set_ticks([0, 1])
                        cbar.set_ticklabels(["Non-HAB", "HAB"])
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                st.caption(
                    f"Water: {result['water_area_ha']:.3f} ha | "
                    f"Predicted HAB: {result['hab_area_ha']:.3f} ha | "
                    f"Coverage: {result['hab_coverage']:.2f}% | "
                    f"Source: {result.get('source', 'project pipeline')}"
                )

    st.caption(
        "Historical values are Swin Transformer predictions for the selected "
        "Sentinel-2 observations. They are not field-validated HAB ground truth. "
        "The representative date for each year is selected from cloud-filtered "
        "Sentinel-2 observations closest to June 30."
    )


# ============================================================
# OBJECTIVE 3 — FUTURE TEMPORAL FORECASTING
# ============================================================

class MultiWaterbodyTemporalModel(nn.Module):
    """Exact state-dict-compatible architecture used by Step 116 training."""
    def __init__(self, input_size, hidden_size, num_layers, dropout, kind):
        super().__init__()
        self.kind = kind.upper()
        recurrent = nn.LSTM if self.kind == "LSTM" else nn.GRU
        layer = recurrent(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        if self.kind == "LSTM":
            self.lstm = layer
        else:
            self.gru = layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if self.kind == "LSTM":
            out, _ = self.lstm(x)
        else:
            out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


@st.cache_data(show_spinner=False)
def load_multiwaterbody_temporal_dataset():
    if not TEMPORAL_DATASET.exists():
        raise FileNotFoundError(
            f"Multi-waterbody temporal dataset not found: {TEMPORAL_DATASET}"
        )

    df = pd.read_csv(TEMPORAL_DATASET)
    required = ["waterbody", "date"] + FORECAST_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Multi-waterbody temporal dataset is missing: " + ", ".join(missing)
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in FORECAST_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["waterbody", "date"] + FORECAST_FEATURES)
        .sort_values(["waterbody", "date"])
        .reset_index(drop=True)
    )
    return df


@st.cache_resource(show_spinner=False)
def load_multiwaterbody_temporal_models():
    if not TEMPORAL_LSTM_CHECKPOINT.exists():
        raise FileNotFoundError(f"LSTM checkpoint not found: {TEMPORAL_LSTM_CHECKPOINT}")
    if not TEMPORAL_GRU_CHECKPOINT.exists():
        raise FileNotFoundError(f"GRU checkpoint not found: {TEMPORAL_GRU_CHECKPOINT}")
    if not TEMPORAL_SCALER_FILE.exists():
        raise FileNotFoundError(f"Temporal scaler not found: {TEMPORAL_SCALER_FILE}")

    device = torch.device("cpu")

    def load_one(path, kind):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = MultiWaterbodyTemporalModel(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
            kind=kind,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    lstm = load_one(TEMPORAL_LSTM_CHECKPOINT, "LSTM")
    gru = load_one(TEMPORAL_GRU_CHECKPOINT, "GRU")

    with open(TEMPORAL_SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)

    return lstm, gru, scaler


def _load_temporal_evaluation_metrics():
    """Load Step 116 metrics while tolerating harmless column-name differences."""
    overall = (
        pd.read_csv(TEMPORAL_METRICS_FILE)
        if TEMPORAL_METRICS_FILE.exists() else pd.DataFrame()
    )
    shamirpet = (
        pd.read_csv(SHAMIRPET_TEMPORAL_METRICS_FILE)
        if SHAMIRPET_TEMPORAL_METRICS_FILE.exists() else pd.DataFrame()
    )

    def normalise(df):
        if df.empty:
            return df
        out = df.copy()
        rename = {}
        aliases = {
            "model": "Model",
            "Model Name": "Model",
            "mae": "MAE",
            "rmse": "RMSE",
            "r2": "R2",
            "R²": "R2",
            "accuracy_within_0.05_ndci_percent": "Accuracy_within_0.05_NDCI_percent",
            "accuracy_within_0.05_percent": "Accuracy_within_0.05_NDCI_percent",
        }
        for col in out.columns:
            key = str(col).strip()
            low = key.lower()
            if key in aliases:
                rename[col] = aliases[key]
            elif low in aliases:
                rename[col] = aliases[low]
        out = out.rename(columns=rename)
        return out

    return normalise(overall), normalise(shamirpet)


def _normalise_waterbody_key(name):
    return re.sub(r"[^a-z0-9]", "", clean_name(name).lower())


def _waterbody_alias(name):
    key = _normalise_waterbody_key(name)
    aliases = {
        "hussainsagar": "Hussain_Sagar",
        "hussainsagarlake": "Hussain_Sagar",
        "saroornagar": "Saroor_Nagar",
        "saroornagarlake": "Saroor_Nagar",
        "osmansagar": "Osman_Sagar",
        "osmansagarlake": "Osman_Sagar",
        "himayatsagar": "Himayat_Sagar",
        "himayatsagarlake": "Himayat_Sagar",
        "shamirpet": "Shamirpet_Lake",
        "shamirpetlake": "Shamirpet_Lake",
    }
    return aliases.get(key)


def _find_shamirpet_images():
    roots = [
        DATA / "raw" / "sentinel2" / "shamirpet_test",
        DATA / "raw" / "sentinel2" / "shamirpet",
        DATA / "raw" / "sentinel2",
    ]
    files = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.tif"):
            low = str(path).lower()
            if "shamirpet" in low and "14channel" in low:
                files.append(path)
        if files:
            break
    return sorted(set(files))


def _extract_date_from_path(path):
    match = re.search(r"(20\d{2})[-_](\d{2})[-_](\d{2})", path.name)
    if not match:
        return None
    try:
        return pd.Timestamp(
            int(match.group(1)), int(match.group(2)), int(match.group(3))
        )
    except Exception:
        return None


def _find_shamirpet_mask(date_value):
    root = DATA / "labels" / "hab_masks" / "shamirpet_test"
    if not root.exists():
        return None
    d1 = date_value.strftime("%Y-%m-%d")
    d2 = date_value.strftime("%Y_%m_%d")
    for path in root.rglob("*.tif"):
        if d1 in path.name or d2 in path.name:
            return path
    return None


def _build_shamirpet_temporal_dataframe():
    """Build the same 10 spectral/index features used by Step 116."""
    files = _find_shamirpet_images()
    if not files:
        raise FileNotFoundError(
            "No Shamirpet 14-channel Sentinel-2 images were found in the project."
        )

    records = []
    for image_path in files:
        d = _extract_date_from_path(image_path)
        if d is None:
            continue
        mask_path = _find_shamirpet_mask(d)
        if mask_path is None:
            continue
        try:
            with rasterio.open(image_path) as src:
                image = src.read().astype(np.float32)
            with rasterio.open(mask_path) as src:
                mask = src.read(1)

            if image.shape[0] != 14 or mask.shape != image.shape[1:]:
                continue

            valid = (
                (mask != 255)
                & np.isfinite(image[10])
                & np.isfinite(image[11])
                & np.isfinite(image[12])
                & np.isfinite(image[13])
            )
            if not valid.any():
                continue

            def stats(arr):
                vals = arr[valid]
                return float(vals.mean()), float(np.median(vals)), float(vals.max())

            ndci_mean, ndci_median, ndci_max = stats(image[12])
            ndwi_mean, ndwi_median, _ = stats(image[10])
            mndwi_mean, mndwi_median, _ = stats(image[11])
            fai_mean, fai_median, fai_max = stats(image[13])

            records.append({
                "waterbody": "Shamirpet_Lake",
                "date": d,
                "ndci_mean": ndci_mean,
                "ndci_median": ndci_median,
                "ndci_max": ndci_max,
                "ndwi_mean": ndwi_mean,
                "ndwi_median": ndwi_median,
                "mndwi_mean": mndwi_mean,
                "mndwi_median": mndwi_median,
                "fai_mean": fai_mean,
                "fai_median": fai_median,
                "fai_max": fai_max,
            })
        except Exception:
            continue

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("Could not build Shamirpet temporal features.")
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def _compute_temporal_features_from_tiff(image_path, mask_path, waterbody_name, selected_date):
    """Compute the exact 10 Step-116 spectral/index features from one 14-band TIFF."""
    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float32)
    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    if image.shape[0] != 14:
        raise ValueError(f"Expected 14 channels, found {image.shape[0]} in {image_path}")
    if mask.shape != image.shape[1:]:
        raise ValueError("Water mask and image dimensions do not match.")

    valid = (
        (mask == 1)
        & np.isfinite(image[10])
        & np.isfinite(image[11])
        & np.isfinite(image[12])
        & np.isfinite(image[13])
    )
    if not valid.any():
        raise ValueError(f"No valid water pixels were found for {selected_date}.")

    def stats(arr):
        vals = arr[valid]
        return float(vals.mean()), float(np.median(vals)), float(vals.max())

    ndci_mean, ndci_median, ndci_max = stats(image[12])
    ndwi_mean, ndwi_median, _ = stats(image[10])
    mndwi_mean, mndwi_median, _ = stats(image[11])
    fai_mean, fai_median, fai_max = stats(image[13])

    return {
        "waterbody": waterbody_name,
        "date": pd.Timestamp(selected_date),
        "ndci_mean": ndci_mean,
        "ndci_median": ndci_median,
        "ndci_max": ndci_max,
        "ndwi_mean": ndwi_mean,
        "ndwi_median": ndwi_median,
        "mndwi_mean": mndwi_mean,
        "mndwi_median": mndwi_median,
        "fai_mean": fai_mean,
        "fai_median": fai_median,
        "fai_max": fai_max,
    }


def _build_generic_temporal_dataframe(waterbody_name, geometry, max_observations=18, status_box=None):
    """Build a cached temporal feature series for a waterbody not in the four training datasets.

    This is an inference-time feature acquisition pipeline. It does NOT retrain the
    LSTM/GRU and therefore does not claim that the temporal model was trained on the
    new waterbody. It supplies real Sentinel-2 history so the trained multi-waterbody
    model can at least produce a model-based forecast for the selected waterbody.
    """
    import ee

    gee_utils = load_gee_utils()
    gee_utils.initialize_gee()
    ee_geometry = ee.Geometry(mapping(geometry))

    collection = gee_utils.get_masked_sentinel2_collection(
        ee_geometry,
        "2016-01-01",
        "2026-09-25",
        cloud_threshold=40,
    )
    dates = gee_utils.get_available_dates(collection)
    parsed = []
    for d in dates:
        try:
            parsed.append(pd.Timestamp(str(d)))
        except Exception:
            continue
    parsed = sorted(set(parsed))

    if len(parsed) < FORECAST_SEQUENCE_LENGTH:
        raise RuntimeError(
            f"Only {len(parsed)} usable Sentinel-2 dates were found for {waterbody_name}. "
            "At least 3 are required for Objective 3."
        )

    # Evenly sample the full history and always include the latest observation.
    n = min(max_observations, len(parsed))
    indices = np.linspace(0, len(parsed) - 1, n, dtype=int)
    selected_dates = [parsed[i] for i in sorted(set(indices))]
    if parsed[-1] not in selected_dates:
        selected_dates[-1] = parsed[-1]
        selected_dates = sorted(set(selected_dates))

    safe_name = _safe_waterbody_name(waterbody_name)
    raw_dir = GENERIC_TEMPORAL_RAW / safe_name
    mask_dir = GENERIC_TEMPORAL_MASKS / safe_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    csv_path = GENERIC_TEMPORAL_DIR / f"{safe_name}_temporal_features.csv"

    records = []
    for idx, selected_date in enumerate(selected_dates, start=1):
        date_str = selected_date.strftime("%Y-%m-%d")
        image_path = raw_dir / f"{safe_name}_{date_str}_14Channel.tif"
        mask_path = mask_dir / f"{safe_name}_water_mask_{date_str}.tif"

        try:
            if not image_path.exists():
                image = gee_utils.get_image_for_date(collection, date_str)
                image_14 = gee_utils.create_14_channel_image(image)
                gee_utils.download_14_channel_image(
                    image_14,
                    ee_geometry,
                    str(image_path),
                    scale=10,
                )

            with rasterio.open(image_path) as src:
                if src.count != 14:
                    raise RuntimeError(
                        f"Downloaded {date_str} image has {src.count} channels; expected 14."
                    )

            if not mask_path.exists():
                create_water_mask(geometry, str(image_path), str(mask_path))

            records.append(
                _compute_temporal_features_from_tiff(
                    str(image_path), str(mask_path), safe_name, date_str
                )
            )
            if status_box is not None:
                status_box.info(
                    f"Preparing temporal history: {idx}/{len(selected_dates)} observations ({date_str})"
                )
        except Exception as exc:
            if status_box is not None:
                status_box.warning(f"Skipped {date_str}: {exc}")

    df = pd.DataFrame(records)
    if len(df) < FORECAST_SEQUENCE_LENGTH:
        raise RuntimeError(
            f"Only {len(df)} usable temporal observations could be prepared for {waterbody_name}. "
            "At least 3 valid observations are required."
        )

    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    return df, f"On-demand Sentinel-2 temporal history ({len(df)} observations)"


def _get_selected_temporal_dataframe(waterbody_name, geometry=None, allow_generic=True):
    alias = _waterbody_alias(waterbody_name)
    if alias == "Shamirpet_Lake":
        return _build_shamirpet_temporal_dataframe(), "Shamirpet local test dataset"

    df = load_multiwaterbody_temporal_dataset()
    if alias is not None:
        selected = df[df["waterbody"].astype(str).str.lower() == alias.lower()].copy()
        if not selected.empty:
            return selected.sort_values("date").reset_index(drop=True), "Multi-waterbody training dataset"

    if allow_generic and geometry is not None:
        safe_name = _safe_waterbody_name(waterbody_name)
        csv_path = GENERIC_TEMPORAL_DIR / f"{safe_name}_temporal_features.csv"
        if csv_path.exists():
            try:
                cached = pd.read_csv(csv_path)
                cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
                for col in FORECAST_FEATURES:
                    cached[col] = pd.to_numeric(cached[col], errors="coerce")
                cached = cached.dropna(subset=["date"] + FORECAST_FEATURES).sort_values("date")
                if len(cached) >= FORECAST_SEQUENCE_LENGTH:
                    return cached.reset_index(drop=True), "Cached on-demand Sentinel-2 temporal history"
            except Exception:
                pass

    return None, "No local temporal dataset is available for this waterbody"


def _recursive_forecast_multiwaterbody(model, scaler, df, target_date):
    target_date = pd.Timestamp(target_date)
    df = df.sort_values("date").reset_index(drop=True)
    last_date = pd.Timestamp(df["date"].iloc[-1])
    if target_date <= last_date:
        raise ValueError(
            f"Target date must be after the latest observation ({last_date.date()})."
        )
    if len(df) < FORECAST_SEQUENCE_LENGTH:
        raise ValueError("At least 3 temporal observations are required for forecasting.")

    intervals = df["date"].diff().dt.days.dropna()
    median_days = max(1, int(round(intervals.median()))) if not intervals.empty else 25
    current_date = last_date
    current_values = df[FORECAST_FEATURES].iloc[-FORECAST_SEQUENCE_LENGTH:].copy()
    predictions = []

    while current_date < target_date:
        next_date = min(current_date + pd.Timedelta(days=median_days), target_date)
        arr = current_values.to_numpy(dtype=np.float32)
        scaled_window = scaler.transform(arr).astype(np.float32)
        x = torch.from_numpy(scaled_window[None, :, :])
        with torch.no_grad():
            pred_scaled = float(model(x).cpu().numpy().reshape(-1)[0])

        # The Step 116 scaler was fitted on the 10 input features, with
        # ndci_mean as feature 0. The model predicts the scaled next-NDCI
        # target, so invert it with feature-0's scaler parameters.
        if not hasattr(scaler, "scale_") or not hasattr(scaler, "mean_"):
            raise TypeError("Temporal scaler is not a fitted StandardScaler-compatible object.")
        if len(scaler.scale_) < 1 or len(scaler.mean_) < 1:
            raise ValueError("Temporal scaler does not contain NDCI scaling parameters.")
        pred_ndci = float(pred_scaled * float(scaler.scale_[0]) + float(scaler.mean_[0]))
        if not np.isfinite(pred_ndci):
            raise ValueError("Temporal model produced a non-finite NDCI prediction.")

        next_row = current_values.iloc[-1].copy()
        next_row["ndci_mean"] = pred_ndci
        next_row["ndci_median"] = pred_ndci
        next_row["ndci_max"] = pred_ndci
        current_values = pd.concat(
            [current_values.iloc[1:], pd.DataFrame([next_row])],
            ignore_index=True,
        )
        current_date = pd.Timestamp(next_date)
        predictions.append({
            "date": current_date,
            "predicted_ndci": pred_ndci,
        })

    return pd.DataFrame(predictions), median_days


def _forecast_risk_label(ndci):
    if ndci >= HAB_NDCI_ALERT_THRESHOLD:
        return "Higher HAB-risk indicator"
    return "Lower HAB-risk indicator"


def _metric_value(row, column):
    try:
        return float(row[column])
    except Exception:
        return np.nan


def render_future_prediction():
    st.title("Future HAB Prediction")

    analysis = st.session_state.analysis
    selected = st.session_state.selected_waterbody
    waterbody_name = (
        analysis["name"] if analysis is not None
        else selected["name"] if selected is not None
        else ""
    )

    st.markdown(
        f"""
        <div class="info-box">
            <b>Objective 3 — Multi-Waterbody Temporal Forecasting</b><br>
            The trained LSTM and GRU models use the same 10 spectral/index features
            used during Step 116 training to forecast the next available NDCI indicator.
            This is different from the Swin Transformer spatial HAB map.
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        lstm_model, gru_model, scaler = load_multiwaterbody_temporal_models()
    except Exception as exc:
        st.error("Objective 3 model files could not be loaded.")
        st.code(str(exc))
        return

    # Use the geometry already produced by Analysis. If the user opens Future
    # Prediction first, reconstruct the same analysis geometry from OSM.
    temporal_geometry = None
    if analysis is not None and analysis.get("geometry") is not None:
        temporal_geometry = analysis["geometry"]
    elif selected is not None:
        try:
            temporal_geometry = prepare_analysis_geometry(
                get_osm_geometry(selected["osm_type"], selected["osm_id"])
            )
        except Exception:
            temporal_geometry = None

    try:
        temporal_df, source = _get_selected_temporal_dataframe(
            waterbody_name, geometry=temporal_geometry, allow_generic=True
        )
    except Exception as exc:
        st.error("Objective 3 temporal history could not be prepared.")
        st.code(str(exc))
        temporal_df, source = None, str(exc)

    if temporal_df is None:
        st.warning(
            f"No temporal history is currently available for {waterbody_name}."
        )
        st.info(
            "The LSTM/GRU checkpoints were trained on four waterbodies. "
            "For a new waterbody such as Musi River, the application must first "
            "collect real historical Sentinel-2 observations and calculate the same "
            "10 spectral/index features used during Step 116. This does not retrain "
            "the model; it prepares inference-time history for the selected waterbody."
        )
        if temporal_geometry is None:
            st.error("The selected waterbody geometry is not available. Run Analysis first, then return here.")
            return
        if st.button(
            "🛰️ Prepare Historical Sentinel-2 Data for Future Prediction",
            use_container_width=True,
            key="prepare_generic_temporal_history",
        ):
            try:
                progress = st.progress(0.0)
                status = st.empty()
                status.info("Initializing Google Earth Engine and finding usable dates...")
                progress.progress(10)
                temporal_df, source = _build_generic_temporal_dataframe(
                    waterbody_name, temporal_geometry, max_observations=18, status_box=status
                )
                progress.progress(100)
                st.session_state.generic_temporal_ready_for = waterbody_name
                status.success(
                    f"Prepared {len(temporal_df)} temporal observations for {waterbody_name}."
                )
                st.rerun()
            except Exception as exc:
                st.error("Temporal history preparation failed.")
                st.code(str(exc))
        return

    latest_date = pd.Timestamp(temporal_df["date"].iloc[-1])
    min_future = (latest_date + pd.Timedelta(days=1)).date()

    st.markdown(f"**Temporal data source:** {source}")
    

    st.caption(
    f"{len(temporal_df)} usable observations | "
    f"{temporal_df['date'].min().date()} to {latest_date.date()} | "
    "Sequence length: 3 observations"
    )

    c1, c2 = st.columns(2)
    with c1:
        target_date = st.date_input(
            "Future prediction date",
            value=min_future,
            min_value=min_future,
            key="future_target_date_multiwaterbody",
        )
    with c2:
        st.metric("Latest observation", latest_date.strftime("%Y-%m-%d"))

    intervals = temporal_df["date"].diff().dt.days.dropna()
    median_interval = int(round(intervals.median())) if not intervals.empty else 25
    st.caption(
        f"Observed sampling is irregular. The forecast advances using the waterbody's "
        f"median historical interval ({median_interval} days)."
    )

    if st.button(
        "🔮 Predict Future NDCI with LSTM + GRU",
        use_container_width=True,
        key="run_multiwaterbody_forecast",
    ):
        try:
            progress = st.progress(0.0)
            status = st.empty()
            status.info("Preparing the latest 3-observation sequence...")
            progress.progress(25)

            lstm_path, step_days = _recursive_forecast_multiwaterbody(
                lstm_model, scaler, temporal_df, target_date
            )
            progress.progress(60)

            gru_path, _ = _recursive_forecast_multiwaterbody(
                gru_model, scaler, temporal_df, target_date
            )
            progress.progress(100)

            lstm_pred = float(lstm_path.iloc[-1]["predicted_ndci"])
            gru_pred = float(gru_path.iloc[-1]["predicted_ndci"])
            mean_pred = float((lstm_pred + gru_pred) / 2.0)

            st.session_state.forecast_result = {
                "waterbody": waterbody_name,
                "target_date": str(target_date),
                "latest_observation": str(latest_date.date()),
                "lstm_ndci": lstm_pred,
                "gru_ndci": gru_pred,
                "mean_ndci": mean_pred,
                "lstm_risk": _forecast_risk_label(lstm_pred),
                "gru_risk": _forecast_risk_label(gru_pred),
                "mean_risk": _forecast_risk_label(mean_pred),
                "lstm_path": lstm_path,
                "gru_path": gru_path,
                "history": temporal_df[["date", "ndci_mean"]].copy(),
                "step_days": step_days,
                "source": source,
            }
            st.session_state.forecast_error = None
            status.success("Future NDCI prediction completed.")
        except Exception as exc:
            st.session_state.forecast_result = None
            st.session_state.forecast_error = str(exc)
            st.error("Future prediction could not be completed.")
            st.code(str(exc))

    if st.session_state.forecast_error:
        st.error("Technical details from the previous forecast run:")
        st.code(st.session_state.forecast_error)
        return

    result = st.session_state.forecast_result
    if result is None or result.get("waterbody") != waterbody_name:
        st.info("Choose a future date and click the prediction button.")
        return

    st.markdown("### 📌 Future Prediction Results")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Target Date", result["target_date"])
    with c2:
        metric_card("LSTM NDCI", f'{result["lstm_ndci"]:.4f}')
    with c3:
        metric_card("GRU NDCI", f'{result["gru_ndci"]:.4f}')
    with c4:
        metric_card("LSTM + GRU Mean", f'{result["mean_ndci"]:.4f}')

    st.markdown("### 🌿 HAB-Risk Indicator")
    risk_df = pd.DataFrame([
        {"Model": "LSTM", "Predicted NDCI": result["lstm_ndci"], "HAB-risk indicator": result["lstm_risk"]},
        {"Model": "GRU", "Predicted NDCI": result["gru_ndci"], "HAB-risk indicator": result["gru_risk"]},
        {"Model": "LSTM + GRU mean", "Predicted NDCI": result["mean_ndci"], "HAB-risk indicator": result["mean_risk"]},
    ])
    risk_df["Predicted NDCI"] = risk_df["Predicted NDCI"].round(4)
    st.dataframe(risk_df, use_container_width=True, hide_index=True)
    st.caption(
        f"NDCI ≥ {HAB_NDCI_ALERT_THRESHOLD:.2f} is shown only as a candidate HAB-risk indicator. "
        "It is not a measured HAB concentration or field-validated probability."
    )

    # --------------------------------------------------------
    # REAL MODEL EVALUATION METRICS
    # --------------------------------------------------------
    overall_metrics, shamirpet_metrics = _load_temporal_evaluation_metrics()

    st.markdown("### 📊 Model Evaluation")
    st.caption(
        "These are held-out evaluation results from the trained models. "
        "They are not the accuracy of the future date currently being predicted."
    )

    if not overall_metrics.empty:
        cols = st.columns(2)
        for i, model_name in enumerate(["LSTM", "GRU"]):
            row = overall_metrics[
                overall_metrics["Model"].astype(str).str.upper() == model_name
            ]
            with cols[i]:
                if not row.empty:
                    r = row.iloc[0]
                    st.markdown(f"**{model_name} — held-out multi-waterbody test**")
                    metric_card("MAE", f'{_metric_value(r, "MAE"):.4f}')
                    metric_card("RMSE", f'{_metric_value(r, "RMSE"):.4f}')
                    metric_card("R²", f'{_metric_value(r, "R2"):.4f}')
                    metric_card(
                        "±0.05 NDCI accuracy",
                        f'{_metric_value(r, "Accuracy_within_0.05_NDCI_percent"):.2f}%',
                    )

    # if not shamirpet_metrics.empty:
    #     st.markdown("#### Independent Shamirpet unseen-waterbody evaluation")
    #     st.dataframe(
    #         shamirpet_metrics.round(4),
    #         use_container_width=True,
    #         hide_index=True,
    #     )
    #     st.caption(
    #         "Shamirpet was excluded from training. Its negative R² values indicate that "
    #         "generalization to this unseen waterbody remains limited."
    #     )

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------
    st.markdown("### 📈 Historical NDCI + Future Forecast")
    hist = result["history"].tail(100)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hist["date"], hist["ndci_mean"], marker="o", markersize=3, label="Historical NDCI")
    ax.plot(result["lstm_path"]["date"], result["lstm_path"]["predicted_ndci"], label="LSTM forecast")
    ax.plot(result["gru_path"]["date"], result["gru_path"]["predicted_ndci"], label="GRU forecast")
    ax.axhline(HAB_NDCI_ALERT_THRESHOLD, linestyle="--", label="NDCI risk threshold")
    ax.set_xlabel("Date")
    ax.set_ylabel("NDCI")
    ax.set_title(f"Historical and Future NDCI — {waterbody_name}")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # --------------------------------------------------------
    # FORECAST PATH
    # --------------------------------------------------------
    st.markdown("### 🔮 Forecast Path")
    path_df = result["lstm_path"].copy()
    path_df = path_df.rename(columns={"predicted_ndci": "LSTM NDCI"})
    path_df["GRU NDCI"] = result["gru_path"]["predicted_ndci"].to_numpy()
    path_df["LSTM Risk"] = path_df["LSTM NDCI"].apply(_forecast_risk_label)
    path_df["GRU Risk"] = path_df["GRU NDCI"].apply(_forecast_risk_label)
    path_df["date"] = pd.to_datetime(path_df["date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(path_df.round(4), use_container_width=True, hide_index=True)

    st.info(
        "For a future date, the actual NDCI is not known yet. Therefore an actual-vs-predicted "
        "accuracy cannot be calculated for that future date. Once a later Sentinel-2 observation "
        "becomes available, the prediction can be compared with the observed NDCI."
    )


# ============================================================
# OBJECTIVE 4 — AUTHORITY EMAIL ALERTS
# ============================================================

def _send_authority_email(
    smtp_host, smtp_port, sender_email, sender_password,
    recipient_email, subject, body
):
    """Send an alert using authenticated SMTP. Credentials are never stored."""
    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(smtp_host, int(smtp_port), timeout=30) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)


def _recommended_authority_recipients(analysis):
    """Return verified, role-based contacts relevant to the selected waterbody.

    The current configured contacts are for Hyderabad/HMDA waterbodies.
    They are suggestions for notification, not a legal determination of
    incident responsibility. For waterbodies outside this mapped region,
    the UI asks the user to provide a local verified recipient.
    """
    if not analysis:
        return []

    name = str(analysis.get("name", "")).lower()
    lat = analysis.get("latitude")
    lon = analysis.get("longitude")

    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        lat = lon = None

    # Hyderabad/HMDA-area mapping. This includes the project waterbodies
    # such as Hussain Sagar, Osman Sagar, Himayat Sagar and Saroor Nagar.
    hyderabad_area = (
        lat is not None
        and lon is not None
        and 17.15 <= lat <= 17.65
        and 78.20 <= lon <= 78.75
    )

    known_hyderabad_name = any(
        key in name
        for key in [
            "hussain", "osman", "himayat", "saroor",
            "musi", "shamirpet", "durgam", "kapra",
        ]
    )

    if not (hyderabad_area or known_hyderabad_name):
        return []

    return [
        {
            "label": "HMDA — Lakes Protection Cell",
            "email": "hod-lpc@hmda.gov.in",
            "reason": "Lake/waterbody protection and catchment administration",
        },
        {
            "label": "GHMC — Commissioner",
            "email": "commissioner-ghmc@gov.in",
            "reason": "Municipal/civic authority for Hyderabad",
        },
        {
            "label": "Telangana Pollution Control Board — Member Secretary",
            "email": "ms-tspcb@telangana.gov.in",
            "reason": "Environmental pollution and water-quality regulation",
        },
    ]


def render_alerts():
    st.title("Authority Alerts")

    analysis = st.session_state.get("analysis")
    forecast = st.session_state.get("forecast_result")

    st.markdown(
        """
        <div class="info-box">
            <b>Objective 4 — HAB Alert System</b><br>
            Generate a waterbody-specific alert from the current Swin detection
            and/or LSTM/GRU forecast. Recommended recipients are shown as
            stakeholder suggestions; the user confirms the recipients before sending.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if analysis is not None:
        st.markdown("### Current HAB Detection Alert")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Waterbody", analysis.get("name", "Unknown"))
        c2.metric("HAB Area", f"{analysis.get('hab_area_ha', 0):.2f} ha")
        c3.metric("Coverage", f"{analysis.get('hab_coverage', 0):.2f}%")
        c4.metric("Date", str(analysis.get("date", "Unknown")))
    else:
        st.info("Run an Analysis first. The detected HAB result will appear here.")

    if forecast is not None:
        st.markdown("### Future Prediction Alert")
        st.write(
            f"Waterbody: **{forecast.get('waterbody', 'Unknown')}** | "
            f"Target date: **{forecast.get('target_date', 'Unknown')}**"
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("LSTM NDCI", f"{forecast.get('lstm_ndci', float('nan')):.4f}")
        c2.metric("GRU NDCI", f"{forecast.get('gru_ndci', float('nan')):.4f}")
        c3.metric("Mean NDCI", f"{forecast.get('mean_ndci', float('nan')):.4f}")
    else:
        st.info("Run Future Prediction first if you also want the temporal forecast included in the alert.")

    # --------------------------------------------------------
    # RECOMMENDED RECIPIENTS
    # --------------------------------------------------------
    st.markdown("### 📮 Recommended Stakeholders")

    recommendations = _recommended_authority_recipients(analysis)

    if recommendations:
        st.caption(
            "These contacts are suggested because of their documented roles in lake/waterbody "
            "protection, municipal administration or environmental regulation. They are not a "
            "legal determination of who is responsible for an individual HAB incident."
        )

        selected_labels = []
        for idx, contact in enumerate(recommendations):
            if st.checkbox(
                f"{contact['label']} — {contact['email']}",
                value=True,
                key=f"alert_recipient_{idx}",
            ):
                selected_labels.append(contact["label"])

        selected_emails = [
            contact["email"]
            for contact in recommendations
            if contact["label"] in selected_labels
        ]

        if selected_emails:
            st.success(
                "Selected recipients: " + ", ".join(selected_emails)
            )
    else:
        selected_emails = []
        st.info(
            "No verified automatic stakeholder mapping is configured for this waterbody's "
            "region. Please enter a locally verified authority or stakeholder email below."
        )

    # --------------------------------------------------------
    # CONFIGURE EMAIL
    # --------------------------------------------------------
    st.markdown("### Configure Email")
    st.caption(
        "For Gmail, use smtp.gmail.com, port 587, and a Google App Password rather than your normal account password."
    )

    c1, c2 = st.columns(2)
    with c1:
        smtp_host = st.text_input("SMTP host", value="smtp.gmail.com")
        smtp_port = st.number_input(
            "SMTP port",
            min_value=1,
            max_value=65535,
            value=587,
            step=1,
        )
        sender_email = st.text_input("Sender email")

    with c2:
        default_recipient = ", ".join(selected_emails)
        recipient_email = st.text_input(
            "Authority / recipient email(s)",
            value=default_recipient,
            help="You may enter multiple addresses separated by commas.",
        )
        sender_password = st.text_input(
            "SMTP / App password",
            type="password",
        )

        default_name = (
            analysis.get("name") if analysis is not None
            else forecast.get("waterbody") if forecast is not None
            else "Waterbody"
        )

    alert_type = st.radio(
        "Alert contents",
        ["Current HAB detection", "Future NDCI prediction", "Both"],
        horizontal=True,
    )

    # Dynamic subject changes with the selected waterbody and alert type.
    if alert_type == "Current HAB detection":
        default_subject = f"HAB WATCH Alert — Potential HAB Detected in {default_name}"
    elif alert_type == "Future NDCI prediction":
        default_subject = f"HAB WATCH Forecast Alert — {default_name}"
    else:
        default_subject = f"HAB WATCH Alert + Forecast — {default_name}"

    subject = st.text_input(
        "Email subject",
        value=default_subject,
    )

    def make_body():
        lines = [
            "HAB WATCH — Harmful Algal Bloom Monitoring Alert",
            "",
            f"Waterbody: {default_name}",
            "",
            "This message was generated dynamically from the selected waterbody's current model results.",
            "",
        ]

        if analysis is not None and alert_type in {"Current HAB detection", "Both"}:
            name = analysis.get("name", "Unknown")
            lines += [
                "CURRENT SPATIAL HAB DETECTION",
                f"Selected waterbody: {name}",
                f"Waterbody type: {analysis.get('type', 'Unknown')}",
                f"Observation date: {analysis.get('date', 'Unknown')}",
                f"Latitude: {analysis.get('latitude', 'Unknown')}",
                f"Longitude: {analysis.get('longitude', 'Unknown')}",
                f"Water area: {float(analysis.get('water_area_ha', 0)):.3f} ha",
                f"Predicted HAB area: {float(analysis.get('hab_area_ha', 0)):.3f} ha",
                f"Predicted non-HAB area: {float(analysis.get('non_hab_area_ha', 0)):.3f} ha",
                f"Predicted HAB coverage: {float(analysis.get('hab_coverage', 0)):.2f}%",
                f"Water pixels: {int(analysis.get('water_pixels', 0))}",
                f"Predicted HAB pixels: {int(analysis.get('hab_pixels', 0))}",
                f"Sentinel-2 system index: {analysis.get('system_index', 'Unknown')}",
                "Spatial model: Multi-waterbody Swin Transformer",
                "",
            ]

        if forecast is not None and alert_type in {"Future NDCI prediction", "Both"}:
            lines += [
                "FUTURE TEMPORAL PREDICTION",
                f"Selected waterbody: {forecast.get('waterbody', 'Unknown')}",
                f"Latest temporal observation: {forecast.get('latest_observation', 'Unknown')}",
                f"Target prediction date: {forecast.get('target_date', 'Unknown')}",
                f"LSTM predicted NDCI: {float(forecast.get('lstm_ndci', float('nan'))):.4f}",
                f"GRU predicted NDCI: {float(forecast.get('gru_ndci', float('nan'))):.4f}",
                f"LSTM + GRU mean NDCI: {float(forecast.get('mean_ndci', float('nan'))):.4f}",
                f"LSTM HAB-risk indicator: {forecast.get('lstm_risk', 'Unknown')}",
                f"GRU HAB-risk indicator: {forecast.get('gru_risk', 'Unknown')}",
                f"Candidate NDCI alert threshold: {HAB_NDCI_ALERT_THRESHOLD:.2f}",
                "Temporal model: Multi-waterbody LSTM + GRU",
                "",
            ]

        lines += [
            "IMPORTANT:",
            "These are model-derived remote-sensing indicators. Spatial HAB values are predictions against the project's pseudo-label-trained model pipeline, not field-validated observations.",
            "A future NDCI prediction is not a measured HAB concentration or probability.",
            "Please verify important operational decisions with appropriate field/laboratory observations.",
            "",
            "HAB WATCH — Any Waterbody. A Healthier Tomorrow.",
        ]
        return "\n".join(lines)

    preview_body = make_body()
    st.markdown("### ✉️ Dynamic Alert Message Preview")
    st.caption(
        "The message below is generated from the currently selected waterbody and its current model results. "
        "Changing the waterbody, observation or forecast automatically changes the message."
    )
    st.text_area(
        "Email body",
        value=preview_body,
        height=460,
        key="dynamic_email_preview",
        disabled=True,
    )

    if st.button(
        "📧 Send Alert Email",
        type="primary",
        use_container_width=True,
    ):
        if not recipient_email.strip() or not sender_email.strip() or not sender_password:
            st.error("Enter sender email, recipient/authority email(s) and SMTP/App password.")
        elif analysis is None and forecast is None:
            st.error("Run at least one detection or future prediction before sending an alert.")
        else:
            try:
                body = make_body()
                _send_authority_email(
                    smtp_host,
                    smtp_port,
                    sender_email.strip(),
                    sender_password,
                    recipient_email.strip(),
                    subject.strip(),
                    body,
                )
                st.success(
                    f"Alert email sent successfully to {recipient_email.strip()}."
                )
            except Exception as exc:
                st.error("The alert email could not be sent.")
                error_text = str(exc)
                st.code(error_text)
                if (
                    "BadCredentials" in error_text
                    or "Username and Password not accepted" in error_text
                    or "5.7.8" in error_text
                ):
                    st.warning(
                        "Gmail rejected the login. Use a Google App Password, not your normal Gmail password. "
                        "The Google account must have 2-Step Verification enabled before an App Password can be created. "
                        "Do not paste your normal Gmail password into this app."
                    )

    st.markdown("### Alert Design")
    st.write(
        "The system deliberately uses a user-triggered send button instead of silently sending emails on every Streamlit rerun. "
        "This prevents duplicate alerts and lets the user verify the model output before contacting a stakeholder."
    )


# ============================================================
# METHODOLOGY
# ============================================================

def render_methodology():

    st.title("Methodology")

    st.markdown(
        """
        ## 🌊 HAB WATCH — End-to-End Methodology

        This page documents the pipeline that is actually connected to the
        application. Values shown by the Analysis and Historical pages are
        generated from the live backend or from previously generated project
        files; historical values are not invented by the UI.
        """
    )

    st.markdown("### 1. User Location and Waterbody Discovery")
    st.write(
        "The user enters a place name or latitude/longitude and a search radius. "
        "Place names are geocoded through OpenStreetMap Nominatim. Nearby waterbodies "
        "are discovered through OpenStreetMap/Overpass and the user selects one for analysis."
    )

    st.markdown("### 2. OSM Geometry")
    st.write(
        "The selected OSM feature is retrieved as its actual geometry. Polygon and "
        "MultiPolygon waterbodies are used directly. Linear river features are converted "
        "to an analysis polygon by buffering the line geometry."
    )

    st.markdown("### 3. Sentinel-2 Data Acquisition")
    st.write(
        "The backend uses Sentinel-2 Surface Reflectance Harmonized imagery through Google Earth Engine. "
        "The existing project preprocessing joins Sentinel-2 cloud-probability data and applies a "
        "cloud-probability threshold of less than 40%."
    )

    st.markdown("### 4. Exact 14-Channel Model Input")
    st.write("The Swin Transformer receives exactly the same 14-channel structure used by the trained model:")
    st.code(
        "B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, "
        "NDWI, MNDWI, NDCI, FAI",
        language="text"
    )
    st.write(
        "NDWI, MNDWI, NDCI and FAI are calculated from the Sentinel-2 spectral bands. "
        "The UI explicitly validates that the downloaded raster contains exactly 14 bands "
        "before Swin inference."
    )

    st.markdown("### 5. Water Mask")
    st.write(
        "The selected OSM waterbody geometry is rasterized onto the Sentinel-2 image grid. "
        "This mask restricts the HAB prediction statistics to the selected waterbody. "
        "It should be interpreted as a geometry-constrained analysis mask, not as independent "
        "field-validated water classification."
    )

    st.markdown("### 6. Swin Transformer HAB Detection")
    st.write(
        "The multi-waterbody Swin Transformer performs pixel-level binary segmentation: "
        "Non-HAB versus HAB. The application tiles the input into 256×256 regions, runs the model, "
        "reconstructs the full prediction and then applies the waterbody mask."
    )

    st.markdown("### 7. Area and Coverage")
    st.write(
        "For the final prediction, water pixels, HAB pixels and Non-HAB pixels are counted. "
        "Pixel area is calculated from the raster georeferencing, including geographic-CRS area "
        "calculation where necessary."
    )
    st.latex(r"HAB\ Coverage(\%) = \frac{HAB\ pixels}{Water\ pixels}\times100")

    st.markdown("### 8. Training Dataset Used by the Current Swin Model")
    st.write(
        "The current multi-waterbody training setup uses Hussain Sagar, Saroor Nagar, Osman Sagar "
        "and Himayat Sagar. Shamirpet Lake was kept as an unseen test waterbody. The training/validation "
        "split was performed by waterbody-date groups to reduce date leakage."
    )

    st.markdown("### 9. Dataset Balance")
    st.write(
        "The multi-waterbody pseudo-labelled dataset contains 391 usable labelled images and about "
        "29.77 million valid pixels. HAB pixels account for approximately 8.31% and Non-HAB pixels "
        "91.69%, so class imbalance was explicitly considered during spatial model training."
    )

    st.markdown("### 10. Current Model Evaluation")
    st.write(
        "The current corrected multi-waterbody Swin experiment achieved a validation F1 of about "
        "0.394 in the recorded project experiment. Shamirpet, the unseen test waterbody, had lower "
        "F1 than the training waterbodies. These measurements are against pseudo-labels and should "
        "not be presented as field-validated accuracy."
    )

    st.markdown("### 11. Historical HAB Analysis — 2016–2026")
    st.write(
        "The Historical page selects one real cloud-filtered Sentinel-2 observation per calendar year, "
        "choosing the available date closest to June 30. Each selected image goes through the same "
        "14-channel preprocessing, water-mask generation and Swin inference used by the Analysis page."
    )
    st.write(
        "For the four project training waterbodies, the existing real multi-waterbody Sentinel-2 dataset "
        "is reused for 2016–2025. Missing years are queried from Google Earth Engine. Completed prediction "
        "files are cached so the historical job can resume instead of repeating completed years."
    )

    st.markdown("### 12. Historical Outputs")
    st.write(
        "The Historical page produces a year-by-year table, predicted HAB coverage graph, predicted HAB "
        "area graph, Sentinel-2 true-colour image and Swin HAB heatmap for each successfully processed year. "
        "A CSV report is also saved under results/swin/ui_historical."
    )

    st.markdown("### 13. Objective 3 — Future Temporal HAB Prediction")
    st.write(
        "Objective 3 uses the multi-waterbody temporal models trained on Hussain Sagar, "
        "Saroor Nagar, Osman Sagar and Himayat Sagar. The models use 10 spectral/index "
        "features and a sequence of 3 observations to forecast the next NDCI value."
    )
    st.code(
        "10 spectral/index features → 3-observation sequence → LSTM / GRU → next NDCI",
        language="text"
    )
    st.write(
        "The Future Prediction page loads the already-trained Step 116 LSTM and GRU checkpoints "
        "and the shared training scaler. It does not retrain a separate model inside the UI. "
        "For the supported project waterbodies, the latest temporal observations are used as the "
        "input sequence. Future predictions are recursive and advance using the waterbody's "
        "median historical observation interval."
    )

    st.markdown("### 14. Dynamic Temporal Generalization")
    st.write(
        "For a selected waterbody that is not already present in the local temporal training dataset, "
        "the application can prepare an inference-time historical series directly from Sentinel-2 through "
        "Google Earth Engine. The same 10 spectral/index features and water-mask procedure are used. "
        "This prepares real historical inputs for the trained LSTM/GRU; it does not retrain the temporal model "
        "for the newly selected waterbody."
    )
    st.code(
        "Selected waterbody → Sentinel-2 history → water mask → 10 temporal features → LSTM + GRU → future NDCI",
        language="text"
    )

    st.markdown("### 15. Objective 3 Evaluation")
    st.write(
        "The multi-waterbody held-out test and the independent Shamirpet unseen-waterbody test "
        "are reported using regression metrics: MAE, RMSE and R². The project also reports the "
        "percentage of predictions within ±0.05 NDCI as a transparent tolerance-based measure. "
        "This is not classification accuracy."
    )
    st.write(
        "For a genuinely future date, actual NDCI is not available at prediction time, so an "
        "actual-vs-predicted accuracy cannot be claimed until a later Sentinel-2 observation exists."
    )

    st.markdown("### ⚠️ Scientific Limitations")
    st.warning(
        "Swin results are model predictions, not field-validated HAB observations. Very small waterbodies "
        "may contain only a small number of 10 m Sentinel-2 pixels, making spatial prediction less reliable. "
        "Cloud filtering, image availability, OSM geometry quality and the pseudo-label methodology can all "
        "affect the result. Objective 3 uses a multi-waterbody temporal training dataset, while forecasts for "
        "previously unseen waterbodies are inference-time generalization and are not waterbody-specific validation."
    )



# ============================================================
# ROUTER
# ============================================================

if st.session_state.page == "Home":

    render_home()

elif st.session_state.page == "Waterbodies":

    render_waterbodies()

elif st.session_state.page == "Analysis":

    render_analysis()

elif st.session_state.page == "Historical":

    render_historical()

elif st.session_state.page == "Future Prediction":

    render_future_prediction()

elif st.session_state.page == "Alerts":

    render_alerts()

elif st.session_state.page == "Methodology":

    render_methodology()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <hr>
    <div style="
        color:#788991;
        font-size:12px;
        padding-top:8px;
    ">
        🌊 <b>HAB WATCH</b>
        &nbsp;&nbsp; Any Waterbody. A Healthier Tomorrow.
    </div>
    """,
    unsafe_allow_html=True
)
