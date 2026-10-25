"""Kiln-drying duration decision-support prototype (V1.9.0 model).

Run from the project root:
    streamlit run app.py

Required artifacts:
    models/candidate_target_mc_duration_model.joblib
    models/model_contract.json
Optional artifacts:
    outputs/audit_metrics.csv
    outputs/audit_bootstrap_samples.csv
"""

from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import altair as alt
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "candidate_target_mc_duration_model.joblib"
CONTRACT_PATH = APP_DIR / "models" / "model_contract.json"
AUDIT_METRICS_PATH = APP_DIR / "outputs" / "audit_metrics.csv"
BOOTSTRAP_PATH = APP_DIR / "outputs" / "audit_bootstrap_samples.csv"

PLANNING_REFERENCE_DAYS = 14
THICKNESSES = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 8.0]
SPECIES = ["JATI", "MAHONI"]  # category codes used during training
SPECIES_LABELS = {"JATI": "Teak", "MAHONI": "Mahogany"}
KILNS = ["1", "2", "3", "4", "5", "6"]
RAINY_MONTHS = {10, 11, 12, 1, 2, 3}

# Share of the 34 locked-audit batches whose absolute error was within each
# tolerance (V1.9.0 notebook output). Used as an empirical planning buffer,
# not as a statistical prediction interval.
AUDIT_TOLERANCE_COVERAGE = {3: 55.9, 5: 85.3, 7: 94.1}
AUDIT_N_BATCHES = 34

# Observed training ranges. These only trigger warnings; they never clip inputs.
TRAINING_RANGES = {
    "vol_total_m3": (0.191, 16.958),
    "total_lembar": (13, 1106),
    "ket_max": (3.0, 8.0),
    "n_ketebalan": (1, 7),
    "target_mc_at_max_thickness": (8.0, 15.5),
    "kelembaban_pct": (71.09, 92.65),
    "curah_hujan_mm": (0.02, 18.99),
    "suhu_maks_c": (27.49, 32.92),
    "suhu_min_c": (19.17, 23.64),
}

FEATURE_LABELS = {
    "vol_total_m3": ("Total batch volume", "m³"),
    "total_lembar": ("Total number of boards", "boards"),
    "ket_max": ("Maximum board thickness", "cm"),
    "n_ketebalan": ("Number of active thickness classes", "classes"),
    "target_mc_at_max_thickness": ("Target moisture content", "%"),
    "kelembaban_pct": ("Average relative humidity", "%"),
    "curah_hujan_mm": ("Average daily precipitation", "mm/day"),
    "suhu_maks_c": ("Average daily maximum temperature", "°C"),
    "suhu_min_c": ("Average daily minimum temperature", "°C"),
}

AUDIT_METHOD_LABELS = {
    "Model target-aware": ("Extra Trees model", "Yes"),
    "Rata-rata development": ("Development-period mean", "Yes"),
    "Baseline 14 hari": ("Fixed 14-day planning rule", "Yes"),
    "Referensi R²=0 (rata-rata audit; non-operasional)": (
        "Audit-period mean (reference only)",
        "No — uses audit outcomes",
    ),
}

GREEN_DARK = "#1f4d3f"
GREEN = "#2f6b57"
WOOD = "#b7793f"


# ---------------------------------------------------------------------------
# Page setup and styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Kiln-Drying Duration Decision Support",
    page_icon="🪵",
    layout="wide",
)

st.markdown(
    f"""
    <style>
    .block-container {{max-width: 1180px; padding-top: 1.6rem; padding-bottom: 3rem;}}
    .hero {{
        padding: 1.5rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem;
        background: linear-gradient(120deg, #173f35 0%, {GREEN} 60%, {WOOD} 100%);
        color: white;
    }}
    .hero h1 {{font-size: 2.05rem; margin: 0 0 0.35rem 0; color: white; line-height: 1.2;}}
    .hero p {{font-size: 1.0rem; margin: 0; opacity: 0.92;}}
    .step {{
        display: inline-block; background: {GREEN}; color: white; border-radius: 999px;
        width: 1.7rem; height: 1.7rem; line-height: 1.7rem; text-align: center;
        font-weight: 700; margin-right: 0.5rem; font-size: 0.95rem;
    }}
    [data-testid="stMetric"] {{
        padding: 0.9rem 1.1rem; border-radius: 14px;
        background: linear-gradient(135deg, rgba(47,107,87,0.10), rgba(183,121,63,0.08));
        border: 1px solid rgba(47,107,87,0.25);
    }}
    .small-note {{color: #5d6b64; font-size: 0.9rem;}}
    .card {{
        border: 1px solid rgba(47,107,87,0.22); border-radius: 14px;
        padding: 1rem 1.2rem; background: rgba(47,107,87,0.04); height: 100%;
    }}
    .card h4 {{margin: 0 0 0.4rem 0; color: {GREEN_DARK};}}
    </style>
    """,
    unsafe_allow_html=True,
)


def render_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def step_header(number: int, text: str) -> None:
    st.markdown(
        f'<h3><span class="step">{number}</span>{text}</h3>', unsafe_allow_html=True
    )


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


@st.cache_resource
def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_contract() -> dict[str, Any]:
    if not CONTRACT_PATH.exists():
        raise FileNotFoundError(f"Model contract not found: {CONTRACT_PATH}")
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_audit_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(AUDIT_METRICS_PATH) if AUDIT_METRICS_PATH.exists() else pd.DataFrame()
    bootstrap = pd.read_csv(BOOTSTRAP_PATH) if BOOTSTRAP_PATH.exists() else pd.DataFrame()
    return metrics, bootstrap


# ---------------------------------------------------------------------------
# Weather retrieval
# ---------------------------------------------------------------------------


def get_weather_location() -> tuple[float, float] | None:
    """Read coordinates from secrets or environment without exposing them in the UI."""
    lat = lon = None
    try:
        weather_secret = st.secrets["weather"]
        lat = weather_secret.get("latitude")
        lon = weather_secret.get("longitude")
    except Exception:
        pass
    lat = lat if lat is not None else os.getenv("WEATHER_LATITUDE")
    lon = lon if lon is not None else os.getenv("WEATHER_LONGITUDE")
    if lat is None or lon is None:
        return None
    try:
        return float(lat), float(lon)
    except (TypeError, ValueError):
        return None


def _mean_valid(values: list[Any], variable_name: str) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").replace(-999, np.nan).dropna()
    if series.empty:
        raise ValueError(f"No valid values were returned for {variable_name}.")
    return float(series.mean())


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_open_meteo_14d(
    latitude: float, longitude: float, start_date_iso: str, end_date_iso: str
) -> dict[str, Any]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "relative_humidity_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "Asia/Jakarta",
        "forecast_days": 16,
    }
    response = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=30)
    response.raise_for_status()
    daily = pd.DataFrame(response.json().get("daily", {}))
    if daily.empty or "time" not in daily:
        raise ValueError("Open-Meteo did not return usable daily data.")

    daily["time"] = pd.to_datetime(daily["time"]).dt.date
    start_day = date.fromisoformat(start_date_iso)
    end_day = date.fromisoformat(end_date_iso)
    window = daily.loc[daily["time"].between(start_day, end_day)]
    if len(window) != 14 or window["time"].nunique() != 14:
        raise ValueError(
            "The automatic forecast does not cover the full 14-day window. "
            "Use manual mode or select a start date within the forecast horizon."
        )
    return {
        "kelembaban_pct": _mean_valid(window["relative_humidity_2m_mean"].tolist(), "relative humidity"),
        "curah_hujan_mm": _mean_valid(window["precipitation_sum"].tolist(), "precipitation"),
        "suhu_maks_c": _mean_valid(window["temperature_2m_max"].tolist(), "maximum temperature"),
        "suhu_min_c": _mean_valid(window["temperature_2m_min"].tolist(), "minimum temperature"),
        "provider": "Open-Meteo (forecast)",
    }


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_nasa_power_14d(
    latitude: float, longitude: float, start_date_iso: str, end_date_iso: str
) -> dict[str, Any]:
    start_day = date.fromisoformat(start_date_iso)
    end_day = date.fromisoformat(end_date_iso)
    params = {
        "parameters": "RH2M,PRECTOTCORR,T2M_MAX,T2M_MIN",
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": start_day.strftime("%Y%m%d"),
        "end": end_day.strftime("%Y%m%d"),
        "format": "JSON",
        "time-standard": "LST",
    }
    response = requests.get(
        "https://power.larc.nasa.gov/api/temporal/daily/point", params=params, timeout=45
    )
    response.raise_for_status()
    parameters = response.json().get("properties", {}).get("parameter", {})
    required = ["RH2M", "PRECTOTCORR", "T2M_MAX", "T2M_MIN"]
    missing = [name for name in required if name not in parameters]
    if missing:
        raise ValueError(f"NASA POWER did not return these variables: {', '.join(missing)}")

    keys = [(start_day + timedelta(days=offset)).strftime("%Y%m%d") for offset in range(14)]
    for name in required:
        valid = [
            key
            for key in keys
            if key in parameters[name]
            and pd.notna(pd.to_numeric(parameters[name][key], errors="coerce"))
            and float(parameters[name][key]) != -999
        ]
        if len(valid) != 14:
            raise ValueError(
                f"NASA POWER did not provide 14 valid values for {name}. Use manual mode."
            )
    return {
        "kelembaban_pct": _mean_valid([parameters["RH2M"][k] for k in keys], "relative humidity"),
        "curah_hujan_mm": _mean_valid([parameters["PRECTOTCORR"][k] for k in keys], "precipitation"),
        "suhu_maks_c": _mean_valid([parameters["T2M_MAX"][k] for k in keys], "maximum temperature"),
        "suhu_min_c": _mean_valid([parameters["T2M_MIN"][k] for k in keys], "minimum temperature"),
        "provider": "NASA POWER (historical)",
    }


def fetch_weather_auto(start_day: date, latitude: float, longitude: float) -> dict[str, Any]:
    end_day = start_day + timedelta(days=13)
    today = date.today()
    if end_day < today:
        return fetch_nasa_power_14d(latitude, longitude, start_day.isoformat(), end_day.isoformat())
    if start_day >= today and end_day <= today + timedelta(days=15):
        return fetch_open_meteo_14d(latitude, longitude, start_day.isoformat(), end_day.isoformat())
    raise ValueError(
        "This 14-day window crosses historical and forecast periods, or lies outside the "
        "forecast horizon. Use manual mode to avoid silently mixing data sources."
    )


# ---------------------------------------------------------------------------
# Feature construction (must match the V1.9.0 training pipeline)
# ---------------------------------------------------------------------------


def weighted_thickness_stats(counts: dict[float, int]) -> dict[str, Any]:
    active = [(thickness, count) for thickness, count in counts.items() if count > 0]
    if not active:
        raise ValueError("Enter at least one board in the thickness composition.")
    values = np.array([item[0] for item in active], dtype=float)
    weights = np.array([item[1] for item in active], dtype=float)
    mean = float(np.average(values, weights=weights))
    std = float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))
    return {
        "ket_mean": mean,
        "ket_std": std,
        "ket_max": float(values.max()),
        "n_ketebalan": int(len(active)),
        "total_composition": int(weights.sum()),
    }


def build_feature_row(
    contract: dict[str, Any],
    species: str,
    kiln: str,
    volume: float,
    total_boards: int,
    start_day: date,
    counts: dict[float, int],
    target_mc: float,
    weather: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    stats = weighted_thickness_stats(counts)
    if stats["total_composition"] != total_boards:
        raise ValueError(
            f"The thickness composition total ({stats['total_composition']:,}) must equal "
            f"the total number of boards ({total_boards:,})."
        )
    proportions = {
        f"prop_ket_{str(thickness).replace('.', '_')}": counts[thickness] / total_boards
        for thickness in THICKNESSES
    }
    row = {
        "jenis_kayu": species,
        "no_kiln": str(kiln),
        "vol_total_m3": float(volume),
        "total_lembar": int(total_boards),
        "jumlah_asal": 1,  # constant in the historical records
        "vol_per_lembar": float(volume) / int(total_boards),
        "ket_mean": stats["ket_mean"],
        "ket_std": stats["ket_std"],
        "ket_max": stats["ket_max"],
        "n_ketebalan": stats["n_ketebalan"],
        "bulan_in": int(start_day.month),
        "musim_hujan": int(start_day.month in RAINY_MONTHS),
        **proportions,
        "target_mc_at_max_thickness": float(target_mc),
        "target_mc_x_max_thickness": float(target_mc) * stats["ket_max"],
        "kelembaban_pct": float(weather["kelembaban_pct"]),
        "curah_hujan_mm": float(weather["curah_hujan_mm"]),
        "suhu_maks_c": float(weather["suhu_maks_c"]),
        "suhu_min_c": float(weather["suhu_min_c"]),
        "rentang_suhu_c": float(weather["suhu_maks_c"] - weather["suhu_min_c"]),
    }
    selected = contract["selected_features"]
    missing = [feature for feature in selected if feature not in row]
    if missing:
        raise KeyError(f"The application is missing required model features: {missing}")
    return pd.DataFrame([row], columns=selected), row


def range_warnings(row: dict[str, Any]) -> list[str]:
    warnings = []
    for feature, (low, high) in TRAINING_RANGES.items():
        value = float(row[feature])
        if value < low or value > high:
            label, unit = FEATURE_LABELS[feature]
            warnings.append(
                f"{label}: {value:.2f} {unit} is outside the observed training range "
                f"({low:.2f}–{high:.2f} {unit})."
            )
    return warnings


def weather_table(weather: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Variable": [
                "Average relative humidity",
                "Average daily precipitation",
                "Average daily maximum temperature",
                "Average daily minimum temperature",
            ],
            "14-day mean": [
                f"{weather['kelembaban_pct']:.2f} %",
                f"{weather['curah_hujan_mm']:.2f} mm/day",
                f"{weather['suhu_maks_c']:.2f} °C",
                f"{weather['suhu_min_c']:.2f} °C",
            ],
        }
    )


def fmt_date(value: date) -> str:
    return value.strftime("%a, %d %b %Y")


# ---------------------------------------------------------------------------
# Evaluation summaries
# ---------------------------------------------------------------------------


def audit_table(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    model_mae = None
    for _, record in metrics.iterrows():
        label, available = AUDIT_METHOD_LABELS.get(record["metode"], (record["metode"], "—"))
        if record["metode"] == "Model target-aware":
            model_mae = float(record["mae"])
        rows.append(
            {
                "Predictor": label,
                "Available when planning?": available,
                "MAE (days)": float(record["mae"]),
                "RMSE (days)": float(record["rmse"]),
                "R²": float(record["r2"]),
                "MAPE (%)": float(record["mape_pct"]),
            }
        )
    table = pd.DataFrame(rows)
    if model_mae is not None:
        table["MAE reduction by model"] = [
            "—" if row["Predictor"] == "Extra Trees model"
            else f"{100 * (row['MAE (days)'] - model_mae) / row['MAE (days)']:.1f}%"
            for _, row in table.iterrows()
        ]
    return table


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

try:
    contract = load_contract()
    audit_metrics, bootstrap_samples = load_audit_data()
except Exception as exc:
    st.error(f"Application artifacts could not be loaded: {exc}")
    st.stop()

with st.sidebar:
    st.markdown("### Model")
    st.write(f"Version: **{contract.get('version', '—')}**")
    st.write(f"Algorithm: **{contract.get('selected_model', 'Extra Trees')}**")
    st.write("Training data: **original observed batches**")
    st.write(f"Planning reference: **{PLANNING_REFERENCE_DAYS} days**")
    st.info(
        "Candidate decision-support model. Predictions require operator review and "
        "prospective validation before routine use."
    )

render_hero(
    "Kiln-Drying Duration Decision Support",
    "Batch-level estimate of wood kiln-drying duration for loading and downstream "
    "scheduling at a wood-furniture manufacturer in Central Java Regency, Indonesia.",
)

prediction_tab, performance_tab, about_tab = st.tabs(
    ["Prediction", "Model performance", "About & limitations"]
)

# ---------------------------------------------------------------- Prediction
with prediction_tab:
    step_header(1, "Drying start date and 14-day weather conditions")
    date_col, mode_col = st.columns(2)
    with date_col:
        start_date = st.date_input("Drying start date", value=date.today())
    with mode_col:
        weather_mode = st.radio(
            "Weather data source",
            ["Automatic", "Manual"],
            horizontal=True,
            help=(
                "Automatic mode uses NASA POWER for past windows and Open-Meteo for "
                "forecast windows of up to 14 days."
            ),
        )

    end_date = start_date + timedelta(days=13)
    st.caption(
        f"Model weather window: {start_date:%d %b %Y} – {end_date:%d %b %Y} (14 calendar days)."
    )

    weather_data: dict[str, Any] | None = None
    weather_key = f"{start_date.isoformat()}::{end_date.isoformat()}"

    if weather_mode == "Automatic":
        location = get_weather_location()
        if location is None:
            st.warning(
                "Weather coordinates have not been configured. Add them to "
                "`.streamlit/secrets.toml` or select Manual mode."
            )
        elif st.button("Retrieve and summarize 14-day weather data"):
            try:
                with st.spinner("Retrieving weather data..."):
                    fetched = fetch_weather_auto(start_date, *location)
                fetched["window_key"] = weather_key
                st.session_state["weather_auto"] = fetched
            except Exception as exc:
                st.error(f"Automatic weather data could not be used: {exc}")

        stored = st.session_state.get("weather_auto")
        if stored and stored.get("window_key") == weather_key:
            weather_data = stored
            st.success(f"Weather data retrieved from {stored['provider']} for the study area.")
            st.dataframe(weather_table(stored), hide_index=True, use_container_width=True)
        elif stored:
            st.info("The start date changed. Retrieve weather data again for the new window.")
    else:
        st.markdown(
            '<p class="small-note">Enter mean values for the 14-day window shown above.</p>',
            unsafe_allow_html=True,
        )
        w1, w2, w3, w4 = st.columns(4)
        humidity = w1.number_input("Relative humidity (%)", 0.0, 100.0, 85.70, 0.10)
        rainfall = w2.number_input("Precipitation (mm/day)", 0.0, 500.0, 6.70, 0.10)
        max_temp = w3.number_input("Maximum temperature (°C)", -20.0, 60.0, 29.23, 0.10)
        min_temp = w4.number_input("Minimum temperature (°C)", -30.0, 50.0, 21.81, 0.10)
        weather_data = {
            "kelembaban_pct": humidity,
            "curah_hujan_mm": rainfall,
            "suhu_maks_c": max_temp,
            "suhu_min_c": min_temp,
            "provider": "Manual input",
        }

    st.divider()
    step_header(2, "Batch specifications")
    with st.form("prediction_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            species = st.selectbox(
                "Wood species",
                SPECIES,
                index=SPECIES.index("MAHONI"),
                format_func=lambda code: SPECIES_LABELS[code],
            )
        with c2:
            kiln = st.selectbox("Kiln number", KILNS)
        with c3:
            volume = st.number_input("Total batch volume (m³)", 0.001, 100.0, 12.0, 0.10)
        with c4:
            total_boards = st.number_input("Total number of boards", 1, 10000, 750, 1)

        st.markdown("**Number of boards in each thickness class**")
        thickness_columns = st.columns(len(THICKNESSES))
        counts: dict[float, int] = {}
        default_counts = {2.0: 0, 2.5: 500, 3.0: 150, 3.5: 50, 4.0: 50, 5.0: 0, 8.0: 0}
        for column, thickness in zip(thickness_columns, THICKNESSES):
            with column:
                counts[thickness] = st.number_input(
                    f"{thickness:g} cm",
                    min_value=0,
                    max_value=10000,
                    value=default_counts[thickness],
                    step=1,
                    key=f"count_{thickness:g}",
                )

        target_mc = st.number_input(
            "Target moisture content for the thickest boards (%)",
            min_value=1.0,
            max_value=40.0,
            value=11.0,
            step=0.5,
            help="The release specification for the thickest boards, defined before drying.",
        )
        submitted = st.form_submit_button(
            "Predict drying duration", type="primary", use_container_width=True
        )

    if submitted:
        st.session_state.pop("last_result", None)
        if weather_data is None:
            st.error("14-day weather data are unavailable. Retrieve them or use Manual mode.")
        elif weather_data["suhu_maks_c"] < weather_data["suhu_min_c"]:
            st.error("Average maximum temperature cannot be lower than average minimum temperature.")
        else:
            try:
                features, values = build_feature_row(
                    contract, species, kiln, volume, int(total_boards),
                    start_date, counts, target_mc, weather_data,
                )
                prediction = float(load_model().predict(features)[0])
                if not math.isfinite(prediction):
                    raise ValueError("The model returned a non-finite prediction.")
                st.session_state["last_result"] = {
                    "prediction": prediction,
                    "values": values,
                    "counts": dict(counts),
                    "species": species,
                    "kiln": kiln,
                    "volume": float(volume),
                    "total_boards": int(total_boards),
                    "target_mc": float(target_mc),
                    "start_date": start_date,
                    "end_date": end_date,
                    "weather_source": weather_data["provider"],
                }
            except Exception as exc:
                st.error(f"The prediction could not be generated: {exc}")

    result = st.session_state.get("last_result")
    if result and result["start_date"] == start_date:
        prediction = result["prediction"]
        planning_days = max(1, math.ceil(prediction))
        completion = result["start_date"] + timedelta(days=planning_days)
        difference = planning_days - PLANNING_REFERENCE_DAYS

        st.divider()
        step_header(3, "Planning estimate")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Model prediction", f"{prediction:.1f} days")
        r2.metric("Planning duration", f"{planning_days} days")
        r3.metric("Planned completion", completion.strftime("%d %b %Y"))
        r4.metric(
            f"Difference from {PLANNING_REFERENCE_DAYS}-day plan",
            f"{difference:+d} days",
            help="Planning duration minus the current fixed planning reference.",
        )
        st.caption(
            "Planning duration is rounded up to a whole calendar day. "
            "The final release decision remains subject to moisture inspection and operator review."
        )

        left, right = st.columns([1.15, 1])
        with left:
            st.markdown("**Planning buffer**")
            buffer_days = st.radio(
                "Buffer around the planned completion",
                list(AUDIT_TOLERANCE_COVERAGE),
                index=1,
                horizontal=True,
                format_func=lambda d: f"±{d} days",
                label_visibility="collapsed",
            )
            earliest = completion - timedelta(days=buffer_days)
            latest = completion + timedelta(days=buffer_days)
            st.info(
                f"Plan for completion between **{fmt_date(earliest)}** and "
                f"**{fmt_date(latest)}**.\n\n"
                f"In the historical audit of the {AUDIT_N_BATCHES} most recent batches, "
                f"**{AUDIT_TOLERANCE_COVERAGE[buffer_days]:.0f}%** finished within "
                f"±{buffer_days} days of the model prediction. This is an empirical "
                "buffer, not a statistical prediction interval."
            )
            warnings = range_warnings(result["values"])
            if warnings:
                st.warning(
                    "\n".join(
                        ["**Inputs outside the historical range — treat the estimate as an extrapolation:**"]
                        + [f"- {w}" for w in warnings]
                    )
                )
            else:
                st.success("All checked inputs lie within the historical training range.")
        with right:
            composition = pd.DataFrame(
                {
                    "Thickness (cm)": [f"{t:g}" for t, n in result["counts"].items() if n > 0],
                    "Boards": [n for n in result["counts"].values() if n > 0],
                }
            )
            chart = (
                alt.Chart(composition, title="Thickness composition of this batch")
                .mark_bar(color=GREEN, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Thickness (cm):N", sort=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Boards:Q"),
                    tooltip=["Thickness (cm)", "Boards"],
                )
                .properties(height=250)
            )
            labels = chart.mark_text(dy=-8, color="#333").encode(text="Boards:Q")
            st.altair_chart(chart + labels, use_container_width=True)

        record = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "model_version": contract.get("version"),
            "model": contract.get("selected_model"),
            "start_date": result["start_date"].isoformat(),
            "weather_window_end": result["end_date"].isoformat(),
            "weather_source": result["weather_source"],
            "species": SPECIES_LABELS[result["species"]],
            "kiln": result["kiln"],
            "volume_m3": result["volume"],
            "total_boards": result["total_boards"],
            "maximum_thickness_cm": result["values"]["ket_max"],
            "target_mc_pct": result["target_mc"],
            "predicted_duration_days": round(prediction, 2),
            "planning_duration_days": planning_days,
            "difference_from_14_day_plan_days": difference,
            "planned_completion_date": completion.isoformat(),
            "range_warnings": warnings,
            "review_status": "operator review required",
            "actual_exit_date": None,  # fill in later for prospective validation
        }
        st.download_button(
            "Download prediction record (JSON)",
            data=json.dumps(record, indent=2, ensure_ascii=False),
            file_name=f"kiln_prediction_{result['start_date'].isoformat()}.json",
            mime="application/json",
        )

    st.divider()
    st.caption(
        "Decision-support prototype based on historical data. It does not replace "
        "moisture-content inspection, kiln safety procedures, or decisions made by "
        "authorized operators."
    )

# --------------------------------------------------------- Model performance
with performance_tab:
    st.subheader("Historical out-of-time audit")
    st.markdown(
        f"The model was developed on earlier batches and evaluated once on the "
        f"**{AUDIT_N_BATCHES} most recent batches**, which were locked away during development."
    )
    table = audit_table(audit_metrics)
    if table.empty:
        st.info("Audit metrics are unavailable.")
    else:
        model_row = table.loc[table["Predictor"].eq("Extra Trees model")].iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Audit MAE", f"{model_row['MAE (days)']:.2f} days")
        m2.metric("Audit RMSE", f"{model_row['RMSE (days)']:.2f} days")
        m3.metric("Audit R²", f"{model_row['R²']:.3f}")
        m4.metric("Audit MAPE", f"{model_row['MAPE (%)']:.1f}%")
        if not bootstrap_samples.empty and "mae" in bootstrap_samples:
            low = bootstrap_samples["mae"].quantile(0.025)
            high = bootstrap_samples["mae"].quantile(0.975)
            st.caption(
                f"95% bootstrap interval for audit MAE: {low:.2f}–{high:.2f} days "
                "(uncertainty of the historical estimate, not a per-batch interval)."
            )
        st.markdown("**Comparison with planning baselines**")
        st.dataframe(
            table.style.format(
                {"MAE (days)": "{:.2f}", "RMSE (days)": "{:.2f}", "R²": "{:.3f}", "MAPE (%)": "{:.1f}"}
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("**Share of audit batches within an error tolerance**")
    tolerance = pd.DataFrame(
        {
            "Tolerance": [f"±{d} day{'s' if d > 1 else ''}" for d in [1, 2, 3, 5, 7]],
            "Batches within tolerance (%)": [23.5, 41.2, 55.9, 85.3, 94.1],
        }
    )
    tol_chart = (
        alt.Chart(tolerance)
        .mark_bar(color=WOOD, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Tolerance:N", sort=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Batches within tolerance (%):Q", scale=alt.Scale(domain=[0, 100])),
            tooltip=["Tolerance", "Batches within tolerance (%)"],
        )
        .properties(height=240)
    )
    st.altair_chart(
        tol_chart + tol_chart.mark_text(dy=-8, color="#333").encode(text="Batches within tolerance (%):Q"),
        use_container_width=True,
    )

    st.info(
        "**How to read these numbers.** MAE is the typical error in days and is the most "
        "useful figure for planning. R² is not an accuracy percentage: it compares the model "
        "with the audit period's own average, which is unknown when planning. Durations in the "
        "audit period were longer on average than in earlier periods, so even the planning "
        "baselines obtain strongly negative R²."
    )

# ----------------------------------------------------------------- About
with about_tab:
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Production records", "755")
    a2.metric("Kiln batches", "166")
    a3.metric("Candidate models", "5")
    a4.metric("Selected model", str(contract.get("selected_model", "Extra Trees")))

    st.subheader("Workflow")
    st.dataframe(
        pd.DataFrame(
            {
                "Stage": [
                    "1. Consolidation",
                    "2. Feature construction",
                    "3. Temporal model selection",
                    "4. Locked audit",
                    "5. Application",
                ],
                "Output": [
                    "755 thickness-level records aggregated to 166 batches",
                    "Batch, thickness composition, calendar, target MC and 14-day weather features",
                    "Five regressors compared by purged expanding-window cross-validation",
                    "One evaluation on the 34 most recent batches against planning baselines",
                    "Duration estimate, planning buffer and range warnings",
                ],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    u1, u2 = st.columns(2)
    with u1:
        st.markdown(
            '<div class="card"><h4>Intended use</h4>'
            "<ul><li>Initial duration estimate when a kiln is loaded</li>"
            "<li>Scheduling of kiln capacity and downstream machining</li>"
            "<li>Flagging batches that are likely to exceed the 14-day plan</li></ul></div>",
            unsafe_allow_html=True,
        )
    with u2:
        st.markdown(
            '<div class="card"><h4>Not supported</h4>'
            "<ul><li>Automatic kiln control or batch release</li>"
            "<li>Replacing moisture-content inspection or quality control</li>"
            "<li>Species, kilns or plants outside the historical data</li></ul></div>",
            unsafe_allow_html=True,
        )

    st.subheader("Inputs that need care")
    st.markdown(
        "- **Target moisture content.** Historical records store the moisture content measured "
        "at release for the thickest boards. The model uses it as a proxy for the release "
        "specification set before drying. The relationship is predictive, not causal.\n"
        "- **Weather.** The model was developed with realized NASA POWER weather for the first "
        "14 days after loading. In use, forecasts (Open-Meteo) or manual values are entered, "
        "so forecast error adds uncertainty to the estimate."
    )

    st.subheader("Main limitations")
    st.markdown(
        "- Developed from 166 batches at a single plant (149 mahogany, 17 teak).\n"
        "- Not yet validated prospectively on new batches.\n"
        "- Initial moisture content, wood density and in-kiln climate were not recorded and "
        "likely explain much of the remaining error.\n"
        "- Performance may change with new operating procedures, kiln condition, raw-material "
        "supply or seasons not represented in the data."
    )

    st.subheader("Data and privacy")
    st.write(
        "The application does not load the raw production records. It runs the fitted model "
        "pipeline with the inputs entered by the user. Weather coordinates are kept in the "
        "deployment secrets and are not displayed."
    )
