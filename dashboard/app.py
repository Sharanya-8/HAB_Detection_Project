from pathlib import Path
import sys
import json
import requests
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from utils.gee_utils import (
    initialize_gee,
    get_masked_sentinel2_collection,
    get_available_dates,
    get_image_for_date,
    create_14_channel_image,
    download_14_channel_image,
)

from scripts.dashboard_swin_detection import (
    run_swin_inference,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI-Powered HAB Detection",
    page_icon="🌊",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🌊 AI-Powered Harmful Algal Bloom Detection"
)

st.subheader(
    "Dynamic Waterbody Detection System"
)

st.write(
    "Enter a latitude and longitude to discover nearby "
    "water bodies. Select a waterbody to continue with "
    "HAB analysis."
)


# ============================================================
# SESSION STATE
# ============================================================

if "waterbodies" not in st.session_state:
    st.session_state.waterbodies = []

if "selected_waterbody" not in st.session_state:
    st.session_state.selected_waterbody = None

if "selected_geometry" not in st.session_state:
    st.session_state.selected_geometry = None

if "sentinel_dates" not in st.session_state:
    st.session_state.sentinel_dates = []

if "selected_date" not in st.session_state:
    st.session_state.selected_date = None

if "sentinel_file" not in st.session_state:
    st.session_state.sentinel_file = None

if "detection_result" not in st.session_state:
    st.session_state.detection_result = None


# ============================================================
# LOCATION INPUT
# ============================================================

st.header("📍 Enter Location")

col1, col2 = st.columns(2)

with col1:

    latitude = st.number_input(
        "Latitude",
        value=17.4222,
        format="%.6f",
    )

with col2:

    longitude = st.number_input(
        "Longitude",
        value=78.4739,
        format="%.6f",
    )


radius_km = st.slider(
    "Search radius for nearby water bodies (km)",
    min_value=1,
    max_value=50,
    value=10,
)


# ============================================================
# OVERPASS WATERBODY SEARCH
# ============================================================

def find_waterbodies(
    lat,
    lon,
    radius_km
):

    radius_m = int(
        radius_km * 1000
    )

    query = f"""
    [out:json][timeout:60];

    (
      way["natural"="water"]
        (around:{radius_m},{lat},{lon});

      relation["natural"="water"]
        (around:{radius_m},{lat},{lon});

      way["water"="lake"]
        (around:{radius_m},{lat},{lon});

      relation["water"="lake"]
        (around:{radius_m},{lat},{lon});

      way["water"="reservoir"]
        (around:{radius_m},{lat},{lon});

      relation["water"="reservoir"]
        (around:{radius_m},{lat},{lon});
    );

    out center tags;
    """

    servers = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
    ]

    for server in servers:

        try:

            response = requests.post(
                server,
                data=query,
                timeout=90,
                headers={
                    "User-Agent":
                        "HAB-Detection-Project/1.0"
                },
            )

            if response.status_code != 200:
                continue

            data = response.json()

            results = []

            seen = set()

            for element in data.get(
                "elements",
                []
            ):

                tags = element.get(
                    "tags",
                    {}
                )

                name = tags.get(
                    "name"
                )

                if not name:
                    name = tags.get(
                        "name:en",
                        "Unnamed Waterbody"
                    )

                center = element.get(
                    "center",
                    {}
                )

                element_lat = center.get(
                    "lat"
                )

                element_lon = center.get(
                    "lon"
                )

                if (
                    element_lat is None
                    or element_lon is None
                ):
                    continue

                key = (
                    name,
                    round(
                        element_lat,
                        5
                    ),
                    round(
                        element_lon,
                        5
                    ),
                )

                if key in seen:
                    continue

                seen.add(key)

                results.append(
                    {
                        "name": name,
                        "lat": element_lat,
                        "lon": element_lon,
                        "osm_id": element.get(
                            "id"
                        ),
                        "osm_type": element.get(
                            "type"
                        ),
                    }
                )

            return results

        except Exception:
            continue

    return []


# ============================================================
# FIND WATERBODIES BUTTON
# ============================================================

if st.button(
    "🔎 Find Nearby Water Bodies",
    type="primary"
):

    with st.spinner(
        "Searching for nearby water bodies..."
    ):

        waterbodies = find_waterbodies(
            latitude,
            longitude,
            radius_km
        )

    st.session_state.waterbodies = (
        waterbodies
    )

    st.session_state.selected_waterbody = None
    st.session_state.selected_geometry = None
    st.session_state.sentinel_dates = []
    st.session_state.sentinel_file = None
    st.session_state.detection_result = None

    if waterbodies:

        st.success(
            f"{len(waterbodies)} water bodies found."
        )

    else:

        st.warning(
            "No water bodies were found. "
            "Try increasing the search radius."
        )


# ============================================================
# DISPLAY NEARBY WATERBODIES
# ============================================================

if st.session_state.waterbodies:

    st.header(
        "🗺️ Nearby Water Bodies"
    )

    map_center = [
        latitude,
        longitude
    ]

    nearby_map = folium.Map(
        location=map_center,
        zoom_start=12,
    )

    folium.Marker(
        [latitude, longitude],
        tooltip="Search Location",
        icon=folium.Icon(
            color="blue",
            icon="info-sign",
        ),
    ).add_to(
        nearby_map
    )

    for waterbody in st.session_state.waterbodies:

        folium.Marker(
            [
                waterbody["lat"],
                waterbody["lon"],
            ],
            tooltip=waterbody["name"],
            popup=(
                f"{waterbody['name']}<br>"
                f"Lat: {waterbody['lat']:.6f}<br>"
                f"Lon: {waterbody['lon']:.6f}"
            ),
            icon=folium.Icon(
                color="green",
                icon="tint",
                prefix="fa",
            ),
        ).add_to(
            nearby_map
        )

    st_folium(
        nearby_map,
        width=1000,
        height=500,
    )


# ============================================================
# WATERBODY SELECTION
# ============================================================

if st.session_state.waterbodies:

    st.header(
        "🌊 Select Water Body"
    )

    names = [
        item["name"]
        for item
        in st.session_state.waterbodies
    ]

    selected_name = st.selectbox(
        "Choose a waterbody for HAB analysis",
        names,
    )

    selected = next(
        item
        for item
        in st.session_state.waterbodies
        if item["name"] == selected_name
    )

    st.session_state.selected_waterbody = (
        selected
    )

    st.info(
        f"Selected: {selected['name']} | "
        f"Latitude: {selected['lat']:.6f} | "
        f"Longitude: {selected['lon']:.6f}"
    )


# ============================================================
# GET ACTUAL WATERBODY BOUNDARY
# ============================================================

def get_actual_boundary(
    lat,
    lon,
    name
):

    search_query = (
        f"{name}, {lat}, {lon}"
    )

    url = (
        "https://nominatim.openstreetmap.org/search"
    )

    params = {
        "q": search_query,
        "format": "json",
        "polygon_geojson": 1,
        "limit": 10,
    }

    response = requests.get(
        url,
        params=params,
        timeout=30,
        headers={
            "User-Agent":
                "HAB-Detection-Project/1.0"
        },
    )

    response.raise_for_status()

    results = response.json()

    # --------------------------------------------------------
    # Prefer polygon results
    # --------------------------------------------------------

    for result in results:

        geometry = result.get(
            "geojson"
        )

        if not geometry:
            continue

        if geometry.get(
            "type"
        ) in [
            "Polygon",
            "MultiPolygon",
        ]:

            return geometry

    return None


# ============================================================
# LOAD BOUNDARY
# ============================================================

if st.session_state.selected_waterbody:

    selected = (
        st.session_state.selected_waterbody
    )

    st.header(
        "🌊 Selected Waterbody"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Waterbody",
            selected["name"]
        )

    with col2:
        st.metric(
            "Latitude",
            f"{selected['lat']:.6f}"
        )

    with col3:
        st.metric(
            "Longitude",
            f"{selected['lon']:.6f}"
        )

    if st.button(
        "📐 Retrieve Actual Waterbody Boundary"
    ):

        with st.spinner(
            "Retrieving actual waterbody boundary..."
        ):

            try:

                geometry = get_actual_boundary(
                    selected["lat"],
                    selected["lon"],
                    selected["name"],
                )

                if geometry is None:

                    st.error(
                        "Could not retrieve a polygon "
                        "boundary for this waterbody."
                    )

                else:

                    st.session_state.selected_geometry = (
                        geometry
                    )

                    st.success(
                        "Actual waterbody boundary retrieved."
                    )

            except Exception as e:

                st.error(
                    f"Boundary retrieval failed: {e}"
                )


# ============================================================
# DISPLAY ACTUAL BOUNDARY
# ============================================================

if st.session_state.selected_geometry:

    geometry = (
        st.session_state.selected_geometry
    )

    st.subheader(
        "📐 Actual Waterbody Boundary"
    )

    boundary_map = folium.Map(
        location=[
            st.session_state.selected_waterbody[
                "lat"
            ],
            st.session_state.selected_waterbody[
                "lon"
            ],
        ],
        zoom_start=13,
    )

    folium.GeoJson(
        geometry,
        name="Waterbody Boundary",
        style_function=lambda feature: {
            "fillColor": "blue",
            "color": "blue",
            "weight": 3,
            "fillOpacity": 0.25,
        },
    ).add_to(
        boundary_map
    )

    st_folium(
        boundary_map,
        width=1000,
        height=500,
    )

    # --------------------------------------------------------
    # AOI BOUNDS
    # --------------------------------------------------------

    coordinates = []

    def collect_coordinates(
        obj
    ):

        if (
            isinstance(obj, list)
            and len(obj) >= 2
            and isinstance(
                obj[0],
                (float, int)
            )
            and isinstance(
                obj[1],
                (float, int)
            )
        ):

            coordinates.append(
                (
                    obj[0],
                    obj[1]
                )
            )

        elif isinstance(
            obj,
            list
        ):

            for item in obj:
                collect_coordinates(
                    item
                )

    collect_coordinates(
        geometry["coordinates"]
    )

    if coordinates:

        min_lon = min(
            x for x, y in coordinates
        )

        max_lon = max(
            x for x, y in coordinates
        )

        min_lat = min(
            y for x, y in coordinates
        )

        max_lat = max(
            y for x, y in coordinates
        )

        st.subheader(
            "🛰️ Dynamic Sentinel-2 AOI"
        )

        c1, c2 = st.columns(2)

        with c1:

            st.write(
                f"**Minimum Longitude:** "
                f"{min_lon:.6f}"
            )

            st.write(
                f"**Maximum Longitude:** "
                f"{max_lon:.6f}"
            )

        with c2:

            st.write(
                f"**Minimum Latitude:** "
                f"{min_lat:.6f}"
            )

            st.write(
                f"**Maximum Latitude:** "
                f"{max_lat:.6f}"
            )

        st.success(
            "The selected waterbody polygon has "
            "successfully been converted into a "
            "dynamic Sentinel-2 analysis area."
        )


# ============================================================
# SENTINEL-2 ANALYSIS
# ============================================================

if st.session_state.selected_geometry:

    st.header(
        "🛰️ Sentinel-2 HAB Analysis"
    )

    st.write(
        "The selected waterbody polygon is now used "
        "as the dynamic Sentinel-2 analysis area."
    )

    # --------------------------------------------------------
    # GEE INITIALIZATION
    # --------------------------------------------------------

    try:

        initialize_gee()

        st.success(
            "Google Earth Engine connected successfully."
        )

    except Exception as e:

        st.error(
            f"Google Earth Engine connection failed: {e}"
        )

    # --------------------------------------------------------
    # DATE RANGE
    # --------------------------------------------------------

    st.subheader(
        "📅 Sentinel-2 Image Search"
    )

    c1, c2 = st.columns(2)

    with c1:

        start_date = st.date_input(
            "Start date",
            value=None,
        )

    with c2:

        end_date = st.date_input(
            "End date",
            value=None,
        )

    # --------------------------------------------------------
    # DEFAULT DATES
    # --------------------------------------------------------

    if start_date is None:

        start_date = (
            __import__(
                "datetime"
            ).date(
                2025,
                1,
                1
            )
        )

    if end_date is None:

        end_date = (
            __import__(
                "datetime"
            ).date(
                2025,
                12,
                31
            )
        )

    # --------------------------------------------------------
    # SEARCH BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔎 Find Sentinel-2 Images"
    ):

        try:

            import ee

            aoi = (
                __import__(
                    "utils.gee_utils",
                    fromlist=[
                        "geojson_to_ee_geometry"
                    ]
                )
                .geojson_to_ee_geometry(
                    st.session_state.selected_geometry
                )
            )

            collection = (
                get_masked_sentinel2_collection(
                    aoi,
                    str(start_date),
                    str(end_date),
                )
            )

            dates = get_available_dates(
                collection
            )

            st.session_state.sentinel_dates = (
                dates
            )

            if dates:

                st.success(
                    f"{len(dates)} Sentinel-2 "
                    f"observation dates available."
                )

            else:

                st.warning(
                    "No Sentinel-2 images were found "
                    "for the selected period."
                )

        except Exception as e:

            st.error(
                f"Sentinel-2 search failed: {e}"
            )


# ============================================================
# SELECT SENTINEL DATE
# ============================================================

if st.session_state.sentinel_dates:

    selected_date = st.selectbox(
        "Select a Sentinel-2 observation date",
        st.session_state.sentinel_dates,
    )

    st.session_state.selected_date = (
        selected_date
    )

    st.info(
        f"Selected Sentinel-2 date: "
        f"**{selected_date}**"
    )


# ============================================================
# RETRIEVE SENTINEL DATA
# ============================================================

if (
    st.session_state.selected_geometry
    and st.session_state.selected_date
):

    if st.button(
        "📥 Retrieve Sentinel-2 Data"
    ):

        try:

            import ee

            geometry = (
                st.session_state.selected_geometry
            )

            aoi = (
                __import__(
                    "utils.gee_utils",
                    fromlist=[
                        "geojson_to_ee_geometry"
                    ]
                )
                .geojson_to_ee_geometry(
                    geometry
                )
            )

            collection = (
                get_masked_sentinel2_collection(
                    aoi,
                    str(start_date),
                    str(end_date),
                )
            )

            image = get_image_for_date(
                collection,
                st.session_state.selected_date,
            )

            if image is None:

                st.error(
                    "No Sentinel-2 image found "
                    "for the selected date."
                )

            else:

                image_14 = (
                    create_14_channel_image(
                        image
                    )
                )

                safe_name = (
                    st.session_state
                    .selected_waterbody[
                        "name"
                    ]
                    .replace(
                        " ",
                        "_"
                    )
                    .replace(
                        "/",
                        "_"
                    )
                )

                output_dir = (
                    PROJECT_ROOT
                    / "data"
                    / "raw"
                    / "sentinel2"
                )

                output_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                output_file = (
                    output_dir
                    / (
                        f"Dynamic_{safe_name}_"
                        f"{st.session_state.selected_date}_"
                        f"14Channel.tif"
                    )
                )

                download_14_channel_image(
                    image_14,
                    aoi,
                    output_file,
                    scale=10,
                )

                st.session_state.sentinel_file = (
                    output_file
                )

                # Clear old result
                st.session_state.detection_result = None

                st.success(
                    "Sentinel-2 data successfully retrieved "
                    "and saved."
                )

                st.subheader(
                    "📦 Retrieved Sentinel-2 Dataset"
                )

                st.write(
                    f"**File:** "
                    f"{output_file.name}"
                )

                st.write(
                    f"**Location:** "
                    f"{output_file}"
                )

                st.info(
                    "The dynamic Sentinel-2 14-channel "
                    "dataset is ready for Swin Transformer "
                    "analysis."
                )

        except Exception as e:

            st.error(
                f"Sentinel-2 retrieval failed: {e}"
            )


# ============================================================
# SWIN TRANSFORMER HAB DETECTION
# ============================================================

if st.session_state.sentinel_file:

    st.header(
        "🧠 Swin Transformer HAB Detection"
    )

    st.write(
        "The trained Swin Transformer analyzes the "
        "14-channel Sentinel-2 image and predicts "
        "HAB and Non-HAB pixels inside the selected "
        "waterbody."
    )

    if st.button(
        "🧠 Run Swin HAB Detection",
        type="primary"
    ):

        with st.spinner(
            "Running Swin Transformer inference..."
        ):

            try:

                result = run_swin_inference(
                    st.session_state.sentinel_file,
                    st.session_state.selected_geometry,
                )

                st.session_state.detection_result = (
                    result
                )

                st.success(
                    "Swin Transformer HAB detection completed."
                )

            except Exception as e:

                st.error(
                    f"Swin detection failed: {e}"
                )


# ============================================================
# DISPLAY HAB RESULT
# ============================================================

if st.session_state.detection_result:

    result = (
        st.session_state.detection_result
    )

    st.header(
        "🌊 HAB Detection Result"
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Waterbody Pixels",
            f"{result['water_pixels']:,}"
        )

    with c2:

        st.metric(
            "HAB Pixels",
            f"{result['hab_pixels']:,}"
        )

    with c3:

        st.metric(
            "Estimated HAB",
            f"{result['hab_percentage']:.2f}%"
        )

    # --------------------------------------------------------
    # SECOND ROW
    # --------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Non-HAB Pixels",
            f"{result['non_hab_pixels']:,}"
        )

    with c2:

        st.metric(
            "Selected Date",
            str(
                st.session_state.selected_date
            )
        )

    # --------------------------------------------------------
    # HAB MAP
    # --------------------------------------------------------

    st.subheader(
        "🗺️ Spatial HAB Detection Map"
    )

    prediction = (
        result["prediction"]
    )

    # Create a display image:
    #
    # 0 = outside waterbody
    # 1 = Non-HAB
    # 2 = HAB

    display = np.zeros(
        prediction.shape,
        dtype=np.float32
    )

    display[
        prediction == 1
    ] = 1

    display[
        prediction == 2
    ] = 2

    st.image(
        display,
        caption=(
            "Swin Transformer spatial HAB "
            "classification"
        ),
        use_container_width=True,
    )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    st.subheader(
        "📊 Interpretation"
    )

    st.write(
        f"For **{st.session_state.selected_waterbody['name']}** "
        f"on **{st.session_state.selected_date}**, "
        f"the Swin Transformer classified approximately "
        f"**{result['hab_percentage']:.2f}%** of the "
        f"mapped waterbody pixels as HAB."
    )

    st.warning(
        "This is a model-based HAB indicator. "
        "The spatial model was trained using spectral "
        "pseudo-labels and has not been field validated. "
        "Therefore this result should not be interpreted "
        "as a confirmed laboratory or field measurement."
    )


# ============================================================
# ABOUT
# ============================================================

st.header(
    "ℹ️ About the System"
)

st.subheader(
    "Spatial HAB Detection"
)

st.write(
    "The Swin Transformer is used for spatial HAB "
    "detection from 14-channel Sentinel-2 imagery."
)

st.subheader(
    "Temporal Forecasting"
)

st.write(
    "The GRU model is used for temporal NDCI "
    "forecasting using spectral and environmental features."
)

st.caption(
    "Scientific Note: HAB labels used during spatial "
    "model training are pseudo-labels derived from "
    "Sentinel-2 spectral indicators. The GRU forecasts "
    "NDCI rather than a directly verified HAB concentration. "
    "Results should therefore be interpreted as "
    "model-based indicators rather than field-validated "
    "bloom measurements."
)