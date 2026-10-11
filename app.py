from pathlib import Path

import joblib

import pandas as pd

import requests

import streamlit as st

from datetime import date, timedelta

# ============================================================

# CONFIGURATION

# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_DIR = ROOT / "models"

DATA_PATH = ROOT / "data" / "realistic_port_congestion_800.csv"

# Approximate coordinates of the ports used in this project.

PORTS = {

    "Mumbai": {"latitude": 18.94, "longitude": 72.84},

    "Mundra": {"latitude": 22.74, "longitude": 69.72},

    "Visakhapatnam": {"latitude": 17.69, "longitude": 83.30},

}

st.set_page_config(

    page_title="Aiqon | Port Risk Intelligence",

    page_icon="🚢",

    layout="wide",

)

st.title("🚢 Port Congestion & Shipment Delay Risk")

st.caption("AI-assisted operational planning | Next 24-hour assessment")

st.warning(

    "Research prototype using synthetic training data. Predictions have "

    "not been validated on real port operations and should not be used "

    "as the sole basis for operational decisions."

)

# ============================================================

# LOAD MODELS AND REFERENCE DATA

# ============================================================

@st.cache_resource

def load_model_artifact(target):

    path = MODEL_DIR / f"{target}__v2_selected.joblib"

    if not path.exists():

        return None

    artifact = joblib.load(path)

    if not isinstance(artifact, dict) or "pipeline" not in artifact:

        raise ValueError(f"Invalid model artifact: {path}")

    return artifact

congestion_artifact = load_model_artifact("congestion")

delay_artifact = load_model_artifact("delay")

if congestion_artifact is None or delay_artifact is None:

    st.error(

        "Model files not found. Expected "

        "models/congestion__v2_selected.joblib and "

        "models/delay__v2_selected.joblib."

    )

    st.stop()

congestion_model = congestion_artifact["pipeline"]

delay_model = delay_artifact["pipeline"]

congestion_threshold = float(congestion_artifact["threshold"])

delay_threshold = float(delay_artifact["threshold"])

@st.cache_data

def load_reference_data():

    if not DATA_PATH.exists():

        return None

    return pd.read_csv(DATA_PATH)

reference_data = load_reference_data()

if reference_data is None:

    st.error(

        f"Dataset not found: {DATA_PATH}\n\n"

        "Check the DATA_PATH setting near the top of app.py."

    )

    st.stop()

# ============================================================

# AUTOMATIC WEATHER FETCHING

# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)

def fetch_port_weather(port_name):

    """

    Fetch the next 24 hours of weather from Open-Meteo.

    Weather API documentation:

    https://open-meteo.com/

    Marine forecasts are fetched separately for wave height.

    If required weather data cannot be fetched, return an error

    instead of inventing current weather values.

    """

    if port_name not in PORTS:

        return {"error": "Unsupported port."}

    latitude = PORTS[port_name]["latitude"]

    longitude = PORTS[port_name]["longitude"]

    forecast_url = "https://api.open-meteo.com/v1/forecast"

    forecast_params = {

        "latitude": latitude,

        "longitude": longitude,

        "hourly": (

            "wind_speed_10m,precipitation,temperature_2m"

        ),

        "forecast_days": 2,

        "timezone": "Asia/Kolkata",

    }

    try:

        response = requests.get(

            forecast_url,

            params=forecast_params,

            timeout=15,

        )

        response.raise_for_status()

        weather = response.json()

        hourly = weather.get("hourly", {})

        times = hourly.get("time", [])

        wind = hourly.get("wind_speed_10m", [])

        rain = hourly.get("precipitation", [])

        temperature = hourly.get("temperature_2m", [])

        if not times or not wind or not rain:

            return {"error": "The weather API returned incomplete data."}

        # Keep only forecast hours from the current local date/time onward.

        now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)

        parsed_times = pd.to_datetime(pd.Series(times), errors="coerce")

        mask = parsed_times.ge(now) & parsed_times.lt(

            now + pd.Timedelta(hours=24)

        )

        indices = mask[mask].index.tolist()

        if not indices:

            return {"error": "No forecast hours available for the next 24 hours."}

        wind_values = [

            float(wind[i]) for i in indices

            if i < len(wind) and wind[i] is not None

        ]

        rain_values = [

            float(rain[i]) for i in indices

            if i < len(rain) and rain[i] is not None

        ]

        temp_values = [

            float(temperature[i]) for i in indices

            if i < len(temperature) and temperature[i] is not None

        ]

        if not wind_values or not rain_values:

            return {"error": "Forecast wind or rainfall data is unavailable."}

        # Open-Meteo wind speed is in km/h by default.

        # Convert it to knots to match the model's feature name.

        mean_wind_knots = (

            sum(wind_values) / len(wind_values)

        ) / 1.852

        rainfall_mm = sum(rain_values)

        # Wave height is retrieved from the marine API where available.

        wave_height_m = None

        marine_url = "https://marine-api.open-meteo.com/v1/marine"

        marine_params = {

            "latitude": latitude,

            "longitude": longitude,

            "hourly": "wave_height",

            "forecast_days": 2,

            "timezone": "Asia/Kolkata",

        }

        try:

            marine_response = requests.get(

                marine_url,

                params=marine_params,

                timeout=10,

            )

            marine_response.raise_for_status()

            marine_json = marine_response.json()

            marine_hourly = marine_json.get("hourly", {})

            marine_times = marine_hourly.get("time", [])

            marine_waves = marine_hourly.get("wave_height", [])

            marine_parsed = pd.to_datetime(

                pd.Series(marine_times), errors="coerce"

            )

            marine_mask = marine_parsed.ge(now) & marine_parsed.lt(

                now + pd.Timedelta(hours=24)

            )

            wave_values = [

                float(marine_waves[i])

                for i in marine_mask[marine_mask].index.tolist()

                if i < len(marine_waves)

                and marine_waves[i] is not None

            ]

            if wave_values:

                wave_height_m = sum(wave_values) / len(wave_values)

        except (requests.RequestException, ValueError, TypeError, KeyError):

            wave_height_m = None

        # Simple transparent severity indicator for this prototype.

        # This is a heuristic, not a validated maritime hazard index.

        wind_score = min(mean_wind_knots / 40.0, 1.0)

        rain_score = min(rainfall_mm / 50.0, 1.0)

        if wave_height_m is None:

            severity = 0.5 * wind_score + 0.5 * rain_score

        else:

            wave_score = min(wave_height_m / 5.0, 1.0)

            severity = (

                0.4 * wind_score

                + 0.3 * rain_score

                + 0.3 * wave_score

            )

        return {

            "wind_speed_forecast_knots": round(mean_wind_knots, 2),

            "rainfall_forecast_mm": round(rainfall_mm, 2),

            "wave_height_forecast_m": (

                round(wave_height_m, 2)

                if wave_height_m is not None else None

            ),

            "weather_severity_forecast": round(severity, 3),

            "temperature_c": (

                round(sum(temp_values) / len(temp_values), 2)

                if temp_values else None

            ),

            "forecast_hours": len(indices),

            "retrieved_at": pd.Timestamp.now(

                tz="Asia/Kolkata"

            ).strftime("%d %b %Y, %I:%M %p IST"),

        }

    except (requests.RequestException, ValueError, TypeError, KeyError) as exc:

        return {

            "error": (

                "Could not fetch live weather. Check your internet connection "

                f"and retry. Details: {exc}"

            )

        }

# ============================================================

# MODEL UTILITIES

# ============================================================

def class_probabilities(model, features):

    probabilities = model.predict_proba(features)[0]

    if hasattr(model, "named_steps") and "model" in model.named_steps:

        classes = model.named_steps["model"].classes_

    else:

        classes = model.classes_

    return {

        label: float(probability)

        for label, probability in zip(classes, probabilities)

    }

def probability_for_positive_class(model, features):

    probs = class_probabilities(model, features)

    # Support numeric 1, string "1", and common positive labels.

    for key in probs:

        if key == 1 or str(key).lower() in {

            "1", "true", "yes", "delayed", "delay"

        }:

            return probs[key]

    # Do not silently treat an unknown class as a delay.

    raise ValueError(

        "Could not identify the positive delay class. "

        f"Model classes are: {list(probs.keys())}"

    )

def reference_value(series):
    """Return a robust numeric median or categorical mode for a feature."""
    values = series.dropna()
    if values.empty:
        return None
    if pd.api.types.is_numeric_dtype(values):
        return values.median()
    modes = values.mode()
    return None if modes.empty else modes.iloc[0]


def grouped_sensitivity(model, current, reference, target_class, groups):
    """Measure local model sensitivity by replacing related feature groups.

    Each row compares the current prediction with a scenario in which all
    available features in one operational group are set to their synthetic-
    dataset median/mode. This is a grouped scenario comparison, not a causal
    effect or a formal decomposition of probability.
    """
    current_probs = class_probabilities(model, current)
    if target_class not in current_probs:
        return pd.DataFrame(), current_probs.get(target_class)

    original_probability = current_probs[target_class]
    rows = []
    for group_name, group_description, columns in groups:
        available = [
            col for col in columns
            if col in current.columns and col in reference.columns
        ]
        if not available:
            continue

        scenario = current.copy()
        changed = []
        for col in available:
            value = reference_value(reference[col])
            if value is None:
                continue
            scenario.loc[scenario.index[0], col] = value
            changed.append(col)
        if not changed:
            continue

        try:
            scenario_probability = class_probabilities(model, scenario).get(target_class)
        except (ValueError, TypeError, KeyError):
            continue
        if scenario_probability is None:
            continue

        delta = original_probability - scenario_probability
        if abs(delta) < 0.0001:
            continue
        rows.append({
            "Operational factor": group_name,
            "Relationship considered": group_description,
            "Features changed together": ", ".join(c.replace("_", " ") for c in changed),
            "Current risk": original_probability,
            "Group-reference scenario risk": scenario_probability,
            "Risk difference (percentage points)": delta * 100,
            "Model response": (
                "Current group values raise estimated risk"
                if delta > 0
                else "Current group values lower estimated risk"
            ),
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result["Risk difference (percentage points)"] = result[
            "Risk difference (percentage points)"
        ].round(1)
        result = result.sort_values(
            "Risk difference (percentage points)",
            key=lambda values: values.abs(),
            ascending=False,
        ).reset_index(drop=True)
    return result, original_probability


def local_sensitivity(model, current, reference, target_class):
    """One-feature sensitivity retained for an optional diagnostic view."""
    current_probs = class_probabilities(model, current)
    if target_class not in current_probs:
        return pd.DataFrame()
    original_probability = current_probs[target_class]
    explanations = []
    for column in current.columns:
        if column not in reference.columns:
            continue
        baseline_value = reference_value(reference[column])
        if baseline_value is None:
            continue
        modified = current.copy()
        modified.loc[modified.index[0], column] = baseline_value
        try:
            modified_probability = class_probabilities(model, modified)[target_class]
        except (ValueError, TypeError, KeyError):
            continue
        delta = original_probability - modified_probability
        if abs(delta) < 0.0001:
            continue
        explanations.append({
            "Feature": column.replace("_", " ").title(),
            "Entered value": str(current.iloc[0][column]),
            "Reference value": str(baseline_value),
            "Probability difference (percentage points)": round(delta * 100, 1),
            "Interpretation": "Raises estimated risk" if delta > 0 else "Lowers estimated risk",
        })
    explanations.sort(key=lambda item: abs(item["Probability difference (percentage points)"]), reverse=True)
    return pd.DataFrame(explanations[:10])

# ============================================================

# SIDEBAR: PORT SELECTION AND WEATHER

# ============================================================

st.sidebar.header("Port selection")

selected_port = st.sidebar.selectbox(

    "Select port",

    list(PORTS.keys()),

)

with st.sidebar:

    st.caption("Weather is fetched automatically for the selected port.")

    with st.spinner("Fetching the latest forecast..."):

        weather = fetch_port_weather(selected_port)

    if "error" in weather:

        st.warning(

            f"{weather['error']} Using clearly labelled training-data median "

            "weather values for this demonstration. These are not live conditions."

        )

        weather = {

            "wind_speed_forecast_knots": float(

                reference_data["wind_speed_forecast_knots"].median()

            ),

            "wave_height_forecast_m": float(

                reference_data["wave_height_forecast_m"].median()

            ),

            "rainfall_forecast_mm": float(

                reference_data["rainfall_forecast_mm"].median()

            ),

            "weather_severity_forecast": float(

                reference_data["weather_severity_forecast"].median()

            ),

            "retrieved_at": "Training-data median fallback (not live weather)",

            "fallback": True,

        }

    else:

        st.success("Weather forecast loaded")

    st.metric(

        "Mean forecast wind",

        f"{weather['wind_speed_forecast_knots']:.1f} knots",

    )

    st.metric(

        "Rainfall over next 24 hours",

        f"{weather['rainfall_forecast_mm']:.1f} mm",

    )

    if weather["wave_height_forecast_m"] is None:

        st.info(

            "Wave-height forecast is unavailable. The model will use the "

            "training-data median for this input."

        )

    else:

        st.metric(

            "Mean forecast wave height",

            f"{weather['wave_height_forecast_m']:.2f} m",

        )

    st.metric(

        "Derived weather severity",

        f"{weather['weather_severity_forecast']:.2f} / 1.00",

    )

    st.caption(

        f"Retrieved: {weather['retrieved_at']}"

    )

st.caption(

    "Weather source: Open-Meteo forecast and marine APIs when available. "

    "If unavailable, the app uses training-data medians and labels them as a fallback."

)

# ============================================================

# USER INPUTS: VALUES NOT AUTOMATICALLY KNOWN

# ============================================================

st.header("Operational information")

st.write(

    "Enter the planned and current operational details that are not "

    "reliably available from the weather API."

)

today = date.today()

with st.form("operational_inputs"):

    left, middle, right = st.columns(3)

    with left:

        prediction_date = st.date_input(

            "Prediction date",

            value=today + timedelta(days=1),

            min_value=today,

        )

        scheduled_arrivals = st.number_input(

            "Scheduled vessel arrivals (next 24 hours)",

            min_value=0,

            max_value=100,

            value=14,

        )

        berth_capacity = st.number_input(

            "Berth capacity (vessels per 24 hours)",

            min_value=1,

            max_value=100,

            value=20,

        )

        backlog = st.number_input(

            "Current vessel backlog",

            min_value=0,

            max_value=100,

            value=3,

        )

        previous_wait = st.number_input(

            "Mean waiting time over previous 3 days (hours)",

            min_value=0.0,

            max_value=200.0,

            value=3.0,

            step=0.5,

        )

    with middle:

        container_vessels = st.number_input(

            "Scheduled container vessels",

            min_value=0,

            max_value=100,

            value=6,

        )

        bulk_vessels = st.number_input(

            "Scheduled bulk vessels",

            min_value=0,

            max_value=100,

            value=3,

        )

        tanker_vessels = st.number_input(

            "Scheduled tanker vessels",

            min_value=0,

            max_value=100,

            value=2,

        )

        other_vessels = st.number_input(

            "Other scheduled vessels",

            min_value=0,

            max_value=100,

            value=3,

        )

        vessel_size = st.number_input(

            "Average vessel-size index",

            min_value=0.1,

            max_value=3.0,

            value=1.0,

            step=0.05,

        )

        cargo_volume = st.number_input(

            "Planned cargo volume (kton)",

            min_value=0.0,

            max_value=5000.0,

            value=400.0,

            step=10.0,

        )

    with right:

        labour_availability = st.slider(

            "Labour availability (%)",

            0.0, 100.0, 90.0, 1.0,

        )

        equipment_availability = st.slider(

            "Equipment availability (%)",

            0.0, 100.0, 90.0, 1.0,

        )

        customs_pressure = st.slider(

            "Customs processing pressure",

            0.0, 3.0, 1.0, 0.05,

        )

        arrival_variability = st.slider(

            "Expected arrival variability",

            0.0, 1.0, 0.2, 0.01,

        )

        maintenance = st.checkbox("Planned maintenance")

        public_holiday = st.checkbox(

            "Public holiday on prediction date"

        )

    submitted = st.form_submit_button(

        "Predict next-day risk",

        type="primary",

        use_container_width=True,

    )

# ============================================================

# PREPARE MODEL INPUTS

# ============================================================

if submitted:
    # Reuse the v2 artifacts loaded and validated at startup.
    # Do not reload using obsolete model target names.

    capacity = max(float(berth_capacity), 1.0)

    date_value = pd.Timestamp(prediction_date)

    row = {

        "port_name": selected_port,

        "day_of_week": int(date_value.dayofweek),

        "month": int(date_value.month),

        "is_weekend": int(date_value.dayofweek >= 5),

        "is_public_holiday": int(public_holiday),

        "scheduled_arrivals_next_24h": int(scheduled_arrivals),

        "scheduled_container_vessels": int(container_vessels),

        "scheduled_bulk_vessels": int(bulk_vessels),

        "scheduled_tanker_vessels": int(tanker_vessels),

        "scheduled_other_vessels": int(other_vessels),

        "avg_vessel_size_index": float(vessel_size),

        "planned_cargo_volume_kton": float(cargo_volume),

        "berth_capacity_vessels_next_24h": int(berth_capacity),

        "planned_berth_utilization": float(scheduled_arrivals) / capacity,

        "backlog_vessels_at_prediction_time": int(backlog),

        "mean_wait_hours_previous_3d": float(previous_wait),

        "wind_speed_forecast_knots": weather[

            "wind_speed_forecast_knots"

        ],

        "wave_height_forecast_m": weather[

            "wave_height_forecast_m"

        ],

        "rainfall_forecast_mm": weather[

            "rainfall_forecast_mm"

        ],

        "weather_severity_forecast": weather[

            "weather_severity_forecast"

        ],

        "labor_availability_pct": float(labour_availability),

        "equipment_availability_pct": float(equipment_availability),

        "customs_processing_pressure_index": float(customs_pressure),

        "planned_maintenance": int(maintenance),

        "expected_arrival_variability": float(arrival_variability),

    }

    X = pd.DataFrame([row])

    # Match the feature columns expected by the trained preprocessing pipeline.

    # If a pipeline records feature_names_in_, use those names and their order.

    def expected_features(model):

        if hasattr(model, "feature_names_in_"):

            return list(model.feature_names_in_)

        if hasattr(model, "named_steps"):

            for step in model.named_steps.values():

                if hasattr(step, "feature_names_in_"):

                    return list(step.feature_names_in_)

        return list(X.columns)

    congestion_features = expected_features(congestion_model)

    delay_features = expected_features(delay_model)

    missing_congestion = [

        col for col in congestion_features if col not in X.columns

    ]

    missing_delay = [

        col for col in delay_features if col not in X.columns

    ]

    if missing_congestion or missing_delay:

        st.error(

            "The app inputs do not match the trained model's feature schema. "

            "Please check the feature list in src/train_models.py.\n\n"

            f"Missing for congestion: {missing_congestion}\n\n"

            f"Missing for delay: {missing_delay}"

        )

        st.stop()

    X_congestion = X[congestion_features].copy()

    X_delay = X[delay_features].copy()

    # For missing wave height, use a median from the training/reference data.

    if (

        "wave_height_forecast_m" in X_congestion.columns

        and X_congestion["wave_height_forecast_m"].isna().any()

    ):

        if "wave_height_forecast_m" not in reference_data.columns:

            st.error(

                "Wave height is missing and no reference median is available."

            )

            st.stop()

        wave_median = reference_data[

            "wave_height_forecast_m"

        ].median()

        X_congestion["wave_height_forecast_m"] = (

            X_congestion["wave_height_forecast_m"].fillna(wave_median)

        )

    if (

        "wave_height_forecast_m" in X_delay.columns

        and X_delay["wave_height_forecast_m"].isna().any()

    ):

        if "wave_height_forecast_m" not in reference_data.columns:

            st.error(

                "Wave height is missing and no reference median is available."

            )

            st.stop()

        wave_median = reference_data[

            "wave_height_forecast_m"

        ].median()

        X_delay["wave_height_forecast_m"] = (

            X_delay["wave_height_forecast_m"].fillna(wave_median)

        )

    try:

        congestion_probs = class_probabilities(

            congestion_model, X_congestion

        )

        # Apply the High-congestion decision threshold saved by training.

        if congestion_probs.get("High", 0.0) >= congestion_threshold:

            congestion_prediction = "High"

        else:

            non_high = {

                label: probability

                for label, probability in congestion_probs.items()

                if label != "High"

            }

            congestion_prediction = max(non_high, key=non_high.get)

        delay_probability = probability_for_positive_class(

            delay_model, X_delay

        )

        delay_prediction = int(delay_probability >= delay_threshold)

    except Exception as exc:

        st.error(

            "Prediction failed. This commonly happens when the trained "

            "model expects different feature names, types or preprocessing. "

            "Inspect the full error below."

        )

        st.exception(exc)

        st.stop()

    # ========================================================

    # REQUIREMENT 1: PREDICTION

    # ========================================================

    st.divider()

    st.header("1. Prediction")

    st.caption(

        f"Saved decision thresholds — congestion High: {congestion_threshold:.2f}; "

        f"shipment delay: {delay_threshold:.2f}."

    )

    st.write(

        f"Port: **{selected_port}** | "

        f"Prediction date: **{prediction_date.strftime('%d %B %Y')}**"

    )

    col1, col2, col3 = st.columns(3)

    col1.metric(

        "Predicted congestion",

        str(congestion_prediction),

    )

    col2.metric(

        "Shipment delay probability",

        f"{delay_probability:.1%}",

        help=f"Predicted delayed if probability >= {delay_threshold:.2f}.",

    )

    col3.metric(

        "Forecast wind",

        f"{weather['wind_speed_forecast_knots']:.1f} knots",

    )

    if delay_prediction:

        st.warning(

            f"Shipment-delay alert: probability {delay_probability:.1%} "

            f"meets the model threshold of {delay_threshold:.0%}."

        )

    else:

        st.success(

            f"No delay alert at the current threshold ({delay_threshold:.0%}); "

            f"estimated delay probability is {delay_probability:.1%}."

        )

    st.subheader("Congestion probability by class")

    probability_table = pd.DataFrame({

        "Congestion class": [

            str(label) for label in congestion_probs.keys()

        ],

        "Predicted probability": list(congestion_probs.values()),

    }).sort_values("Predicted probability", ascending=False)

    st.dataframe(

        probability_table,

        use_container_width=True,

        hide_index=True,

    )

    st.bar_chart(

        probability_table.set_index("Congestion class")

    )

    # ========================================================

    # REQUIREMENT 2: RELIABILITY

    # ========================================================

    st.divider()

    st.header("2. Prediction reliability")

    highest_probability = max(congestion_probs.values())

    r1, r2 = st.columns(2)

    r1.metric(

        "Highest congestion-class probability",

        f"{highest_probability:.1%}",

    )

    r2.metric(

        "Estimated delay probability",

        f"{delay_probability:.1%}",

    )

    if highest_probability >= 0.75:

        st.info(

            "The model strongly favours one congestion class. This describes "

            "the model's output concentration, not a verified chance that "

            "the prediction is correct."

        )

    elif highest_probability >= 0.50:

        st.warning(

            "One congestion class leads, but the model assigns meaningful "

            "probability to alternatives."

        )

    else:

        st.warning(

            "The congestion probabilities are spread across classes. "

            "Treat the classification as uncertain."

        )

    st.warning(

        "These are model probabilities, not calibrated real-world confidence "

        "scores. Calibration and evaluation on real operational data are "

        "needed before interpreting them as reliable real-world risk."

    )

    # ========================================================
    # REQUIREMENT 3: COMPACT RELATIONSHIP-AWARE EXPLANATIONS
    # ========================================================

    st.divider()
    st.header("3. Contributing factors")
    st.caption(
        "Top grouped sensitivities for this prediction. Each comparison changes "
        "related inputs together to typical values from the synthetic dataset; "
        "it is not a causal effect or a formal feature attribution."
    )

    operational_groups = [
        (
            "Demand vs. capacity",
            "Vessel arrivals, berth capacity, utilization and planned cargo demand.",
            ["scheduled_arrivals_next_24h", "berth_capacity_vessels_next_24h", "planned_berth_utilization", "scheduled_container_vessels", "scheduled_bulk_vessels", "scheduled_tanker_vessels", "scheduled_other_vessels", "avg_vessel_size_index", "planned_cargo_volume_kton"],
        ),
        (
            "Backlog and waiting",
            "Existing queue and recent waiting time.",
            ["backlog_vessels_at_prediction_time", "mean_wait_hours_previous_3d"],
        ),
        (
            "Labour and equipment",
            "Resource availability and planned maintenance.",
            ["labor_availability_pct", "equipment_availability_pct", "planned_maintenance"],
        ),
        (
            "Weather conditions",
            "Wind, rainfall, waves and derived weather severity.",
            ["wind_speed_forecast_knots", "rainfall_forecast_mm", "wave_height_forecast_m", "weather_severity_forecast"],
        ),
        (
            "Arrival and customs pressure",
            "Expected arrival variability and customs-processing pressure.",
            ["expected_arrival_variability", "customs_processing_pressure_index"],
        ),
        (
            "Port and calendar",
            "Port identity and calendar indicators.",
            ["port_name", "day_of_week", "month", "is_weekend", "is_public_holiday"],
        ),
    ]

    reference_columns = [col for col in X.columns if col in reference_data.columns]
    reference_features = reference_data[reference_columns].copy()

    def render_grouped_explanation(title, model, current_features, target_class):
        st.subheader(title)
        grouped, current_probability = grouped_sensitivity(
            model, current_features, reference_features, target_class, operational_groups
        )
        if current_probability is None:
            st.info("The requested class was not found in the model probability output.")
            return

        st.metric("Current model probability", f"{current_probability:.1%}")
        if grouped.empty:
            st.info("No grouped scenario produced a measurable probability change.")
            return

        # Show only the three largest absolute changes in the main dashboard.
        top_factors = grouped.head(3)
        cards = st.columns(len(top_factors))
        for card, (_, item) in zip(cards, top_factors.iterrows()):
            delta = float(item["Risk difference (percentage points)"])
            direction = "higher" if delta > 0 else "lower"
            with card:
                with st.container(border=True):
                    st.markdown(f"**{item['Operational factor']}**")
                    st.metric(
                        "Difference vs. typical-value scenario",
                        f"{delta:+.1f} pp",
                        help=(
                            "Current probability minus probability after replacing "
                            "this group's available features with dataset median/mode values."
                        ),
                    )
                    st.caption(
                        f"Current inputs give a {abs(delta):.1f} percentage-point "
                        f"{direction} estimate than the group-reference scenario."
                    )
                    st.caption(str(item["Relationship considered"]))

        top = top_factors.iloc[0]
        delta = float(top["Risk difference (percentage points)"])
        direction = "higher" if delta > 0 else "lower"
        st.markdown(
            f"**Key takeaway:** {top['Operational factor']} shows the largest "
            f"grouped sensitivity; the current estimate is {abs(delta):.1f} "
            f"percentage points {direction} than its typical-value scenario."
        )

        with st.expander("View detailed factor comparisons"):
            detail = grouped.copy()
            detail["Current risk"] = detail["Current risk"].map(lambda v: f"{v:.1%}")
            detail["Group-reference scenario risk"] = detail["Group-reference scenario risk"].map(lambda v: f"{v:.1%}")
            detail = detail.rename(columns={
                "Risk difference (percentage points)": "Difference (percentage points)"
            })
            st.dataframe(
                detail[[
                    "Operational factor", "Current risk", "Group-reference scenario risk",
                    "Difference (percentage points)", "Model response"
                ]],
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "These are grouped scenario comparisons, not causal effects or a formal "
                "additive attribution. Differences should not be summed because factors "
                "may be correlated."
            )

    render_grouped_explanation(
        "Congestion risk",
        congestion_model,
        X_congestion,
        congestion_prediction,
    )

    delay_probs_for_explanation = class_probabilities(delay_model, X_delay)
    positive_delay_class = next(
        (label for label in delay_probs_for_explanation
         if label == 1 or str(label).lower() in {"1", "true", "yes", "delayed", "delay"}),
        None,
    )
    render_grouped_explanation(
        "Shipment-delay risk",
        delay_model,
        X_delay,
        positive_delay_class,
    )

    with st.expander("Technical details: individual-feature sensitivity"):
        st.caption(
            "This diagnostic changes one input at a time to its dataset median/mode. "
            "It can create unrealistic combinations when inputs are correlated."
        )
        old_congestion = local_sensitivity(
            congestion_model, X_congestion, reference_features, congestion_prediction
        )
        st.markdown("**Congestion**")
        if old_congestion.empty:
            st.info("No individual feature produced a measurable change.")
        else:
            st.dataframe(old_congestion, use_container_width=True, hide_index=True)
        if positive_delay_class is not None:
            old_delay = local_sensitivity(
                delay_model, X_delay, reference_features, positive_delay_class
            )
            st.markdown("**Shipment delay**")
            if old_delay.empty:
                st.info("No individual feature produced a measurable change.")
            else:
                st.dataframe(old_delay, use_container_width=True, hide_index=True)

    st.caption(
        "Explanations reflect patterns learned from synthetic labels and features. "
        "They describe model behaviour, not proven real-world causes; reference scenarios "
        "are not guaranteed to represent observed operations."
    )

    with st.expander("Inspect all prediction inputs"):

        st.dataframe(X, use_container_width=True, hide_index=True)
