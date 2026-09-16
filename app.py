from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "candidate_target_mc_duration_model.joblib"
CONTRACT_PATH = APP_DIR / "models" / "model_contract.json"
AUDIT_METRICS_PATH = APP_DIR / "outputs" / "audit_metrics.csv"
BOOTSTRAP_PATH = APP_DIR / "outputs" / "audit_bootstrap_samples.csv"

THICKNESSES = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 8.0]
SPECIES = ["JATI", "MAHONI"]
KILNS = ["1", "2", "3", "4", "5", "6"]
RAINY_MONTHS = {10, 11, 12, 1, 2, 3}

# Observed training ranges. These only trigger warnings; they never clip inputs.
TRAINING_RANGES = {
    "vol_total_m3": (0.191, 17.832),
    "total_lembar": (13, 1139),
    "ket_max": (3.0, 8.0),
    "n_ketebalan": (1, 7),
    "target_mc_at_max_thickness": (8.0, 17.0),
    "kelembaban_pct": (71.09, 92.65),
    "curah_hujan_mm": (0.02, 26.16),
    "suhu_maks_c": (27.03, 32.92),
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


st.set_page_config(
    page_title="Kiln-Drying Duration Decision Support",
    page_icon="🌲",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {
        padding: 1rem 1.15rem; border-radius: 0.8rem;
        background: linear-gradient(135deg, #edf7f0, #f8fbf8);
        border: 1px solid #a8ccb2;
    }
    .small-note {color: #58645c; font-size: 0.91rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


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


def get_weather_location() -> tuple[float, float, str] | None:
    """Read coordinates from secrets or environment without exposing them in the UI."""
    lat = lon = label = None
    try:
        weather_secret = st.secrets["weather"]
        lat = weather_secret.get("latitude")
        lon = weather_secret.get("longitude")
        label = weather_secret.get("location_label")
    except Exception:
        pass

    lat = lat if lat is not None else os.getenv("WEATHER_LATITUDE")
    lon = lon if lon is not None else os.getenv("WEATHER_LONGITUDE")
    label = label or os.getenv("WEATHER_LOCATION_LABEL", "study location")

    if lat is None or lon is None:
        return None
    try:
        return float(lat), float(lon), str(label)
    except (TypeError, ValueError):
        return None


def _mean_valid(values: list[Any], variable_name: str) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").replace(-999, np.nan).dropna()
    if series.empty:
        raise ValueError(f"No valid values were returned for {variable_name}.")
    return float(series.mean())


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_open_meteo_14d(
    latitude: float,
    longitude: float,
    start_date_iso: str,
    end_date_iso: str,
) -> dict[str, Any]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ",".join(
            [
                "relative_humidity_2m_mean",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
            ]
        ),
        "timezone": "Asia/Jakarta",
        "forecast_days": 16,
    }
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast", params=params, timeout=30
    )
    response.raise_for_status()
    payload = response.json()
    daily = pd.DataFrame(payload.get("daily", {}))
    if daily.empty or "time" not in daily:
        raise ValueError("Open-Meteo did not return usable daily data.")

    daily["time"] = pd.to_datetime(daily["time"]).dt.date
    start_day = date.fromisoformat(start_date_iso)
    end_day = date.fromisoformat(end_date_iso)
    window = daily.loc[daily["time"].between(start_day, end_day)].copy()
    if len(window) != 14 or window["time"].nunique() != 14:
        raise ValueError(
            "The automatic forecast does not cover the full 14-day window. "
            "Use manual mode or select a start date within the forecast horizon."
        )

    return {
        "kelembaban_pct": _mean_valid(
            window["relative_humidity_2m_mean"].tolist(), "relative humidity"
        ),
        "curah_hujan_mm": _mean_valid(
            window["precipitation_sum"].tolist(), "precipitation"
        ),
        "suhu_maks_c": _mean_valid(
            window["temperature_2m_max"].tolist(), "maximum temperature"
        ),
        "suhu_min_c": _mean_valid(
            window["temperature_2m_min"].tolist(), "minimum temperature"
        ),
        "provider": "Open-Meteo",
        "window_start": start_date_iso,
        "window_end": end_date_iso,
    }


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_nasa_power_14d(
    latitude: float,
    longitude: float,
    start_date_iso: str,
    end_date_iso: str,
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
        "https://power.larc.nasa.gov/api/temporal/daily/point",
        params=params,
        timeout=45,
    )
    response.raise_for_status()
    payload = response.json()
    parameters = payload.get("properties", {}).get("parameter", {})
    required = ["RH2M", "PRECTOTCORR", "T2M_MAX", "T2M_MIN"]
    missing = [name for name in required if name not in parameters]
    if missing:
        raise ValueError(f"NASA POWER did not return these variables: {', '.join(missing)}")

    expected_keys = [
        (start_day + timedelta(days=offset)).strftime("%Y%m%d") for offset in range(14)
    ]
    for name in required:
        valid_days = [
            key
            for key in expected_keys
            if key in parameters[name]
            and pd.notna(pd.to_numeric(parameters[name][key], errors="coerce"))
            and float(parameters[name][key]) != -999
        ]
        if len(valid_days) != 14:
            raise ValueError(
                f"NASA POWER did not provide 14 valid values for {name}. "
                "Use manual mode for this window."
            )

    return {
        "kelembaban_pct": _mean_valid(
            [parameters["RH2M"][key] for key in expected_keys], "relative humidity"
        ),
        "curah_hujan_mm": _mean_valid(
            [parameters["PRECTOTCORR"][key] for key in expected_keys], "precipitation"
        ),
        "suhu_maks_c": _mean_valid(
            [parameters["T2M_MAX"][key] for key in expected_keys], "maximum temperature"
        ),
        "suhu_min_c": _mean_valid(
            [parameters["T2M_MIN"][key] for key in expected_keys], "minimum temperature"
        ),
        "provider": "NASA POWER",
        "window_start": start_date_iso,
        "window_end": end_date_iso,
    }


def fetch_weather_auto(start_day: date, latitude: float, longitude: float) -> dict[str, Any]:
    end_day = start_day + timedelta(days=13)
    today = date.today()
    if end_day < today:
        return fetch_nasa_power_14d(
            latitude, longitude, start_day.isoformat(), end_day.isoformat()
        )
    if start_day >= today and end_day <= today + timedelta(days=15):
        return fetch_open_meteo_14d(
            latitude, longitude, start_day.isoformat(), end_day.isoformat()
        )
    raise ValueError(
        "This 14-day window crosses historical and forecast periods, or lies outside the "
        "forecast horizon. Use manual mode to avoid silently mixing data sources."
    )


def weighted_thickness_stats(counts: dict[float, int]) -> dict[str, Any]:
    active = [(thickness, count) for thickness, count in counts.items() if count > 0]
    if not active:
        raise ValueError("Enter at least one board in the thickness composition.")
    values = np.array([item[0] for item in active], dtype=float)
    weights = np.array([item[1] for item in active], dtype=float)
    mean = float(np.average(values, weights=weights))
    std = float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))
    maximum = float(values.max())
    return {
        "ket_mean": mean,
        "ket_std": std,
        "ket_max": maximum,
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
        "jumlah_asal": 1,
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
    selected_features = contract["selected_features"]
    missing = [feature for feature in selected_features if feature not in row]
    if missing:
        raise KeyError(f"The application is missing required model features: {missing}")
    return pd.DataFrame([row], columns=selected_features), row


def range_warnings(row: dict[str, Any]) -> list[str]:
    warnings = []
    for feature, (low, high) in TRAINING_RANGES.items():
        value = float(row[feature])
        if value < low or value > high:
            label, unit = FEATURE_LABELS[feature]
            warnings.append(
                f"{label}: {value:.2f} {unit} is outside the observed training "
                f"range ({low:.2f}–{high:.2f} {unit})."
            )
    return warnings


def audit_summary(metrics: pd.DataFrame, bootstrap: pd.DataFrame) -> dict[str, float] | None:
    if metrics.empty:
        return None
    candidate = metrics.loc[metrics["metode"].eq("Model target-aware")]
    if candidate.empty:
        return None
    result = {
        "mae": float(candidate.iloc[0]["mae"]),
        "rmse": float(candidate.iloc[0]["rmse"]),
        "r2": float(candidate.iloc[0]["r2"]),
        "mape_pct": float(candidate.iloc[0]["mape_pct"]),
    }
    if not bootstrap.empty and "mae" in bootstrap:
        result["mae_ci_low"] = float(bootstrap["mae"].quantile(0.025))
        result["mae_ci_high"] = float(bootstrap["mae"].quantile(0.975))
    return result


def weather_table(weather: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Variable": [
                "Average relative humidity",
                "Average daily precipitation",
                "Average daily maximum temperature",
                "Average daily minimum temperature",
            ],
            "Value": [
                f"{weather['kelembaban_pct']:.2f} %",
                f"{weather['curah_hujan_mm']:.2f} mm/day",
                f"{weather['suhu_maks_c']:.2f} °C",
                f"{weather['suhu_min_c']:.2f} °C",
            ],
        }
    )


try:
    contract = load_contract()
    audit_metrics, bootstrap_samples = load_audit_data()
except Exception as exc:
    st.error(f"Application artifacts could not be loaded: {exc}")
    st.stop()

st.title("Kiln-Drying Duration Decision Support")
st.caption(
    "Decision-support prototype for estimating batch-level wood kiln-drying duration "
    "at an anonymized wood-furniture manufacturer in Klaten Regency, Indonesia."
)

with st.sidebar:
    st.subheader("Model")
    st.write(f"Version: **{contract.get('version', '—')}**")
    st.write(f"Algorithm: **{contract.get('selected_model', '—')}**")
    st.write("Primary training data: **original observed batches**")
    st.info(
        "This is a candidate model. Predictions require operator review and prospective "
        "validation before full operational use."
    )

prediction_tab, model_tab = st.tabs(["Prediction", "Model information"])

with prediction_tab:
    st.subheader("1. Drying start date and 14-day weather conditions")
    date_col, mode_col = st.columns([1, 1])
    with date_col:
        start_date = st.date_input("Drying start date", value=date.today())
    with mode_col:
        weather_mode = st.radio(
            "Weather data source",
            ["Automatic", "Manual"],
            horizontal=True,
            help=(
                "Automatic mode uses NASA POWER for historical windows and Open-Meteo "
                "for 14-day forecast windows."
            ),
        )

    end_date = start_date + timedelta(days=13)
    st.caption(
        f"Model weather window: {start_date.strftime('%d %b %Y')} – "
        f"{end_date.strftime('%d %b %Y')} (14 calendar days)."
    )

    weather_data: dict[str, Any] | None = None
    weather_key = f"{start_date.isoformat()}::{end_date.isoformat()}"

    if weather_mode == "Automatic":
        location = get_weather_location()
        if location is None:
            st.warning(
                "Weather coordinates have not been configured. Complete "
                "`.streamlit/secrets.toml` or select Manual mode. Coordinates are not "
                "displayed in the interface."
            )
        elif st.button("Retrieve and summarize 14-day weather data", type="secondary"):
            latitude, longitude, location_label = location
            try:
                with st.spinner("Retrieving weather data..."):
                    fetched = fetch_weather_auto(start_date, latitude, longitude)
                fetched["window_key"] = weather_key
                fetched["location_label"] = location_label
                st.session_state["weather_auto"] = fetched
            except Exception as exc:
                st.error(f"Automatic weather data could not be used: {exc}")

        stored_weather = st.session_state.get("weather_auto")
        if stored_weather and stored_weather.get("window_key") == weather_key:
            weather_data = stored_weather
            st.success(
                f"Weather data are available from {weather_data['provider']} for "
                f"{weather_data.get('location_label', 'the study location')}."
            )
            st.dataframe(weather_table(weather_data), hide_index=True, use_container_width=True)
        elif stored_weather:
            st.info("The date changed. Retrieve weather data again for the new 14-day window.")
    else:
        st.markdown(
            '<p class="small-note">Enter daily averages for the 14-day window shown above.</p>',
            unsafe_allow_html=True,
        )
        w1, w2, w3, w4 = st.columns(4)
        with w1:
            humidity = st.number_input(
                "Average relative humidity (%)", 0.0, 100.0, 85.70, 0.10
            )
        with w2:
            rainfall = st.number_input(
                "Average daily precipitation (mm/day)", 0.0, 500.0, 6.70, 0.10
            )
        with w3:
            max_temp = st.number_input("Average maximum temperature (°C)", -20.0, 60.0, 29.23, 0.10)
        with w4:
            min_temp = st.number_input("Average minimum temperature (°C)", -30.0, 50.0, 21.81, 0.10)
        weather_data = {
            "kelembaban_pct": humidity,
            "curah_hujan_mm": rainfall,
            "suhu_maks_c": max_temp,
            "suhu_min_c": min_temp,
            "provider": "Manual input",
            "window_start": start_date.isoformat(),
            "window_end": end_date.isoformat(),
        }

    st.divider()
    st.subheader("2. Batch specifications")
    with st.form("prediction_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            species = st.selectbox("Wood species", SPECIES)
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

        active_thicknesses = [value for value, count in counts.items() if count > 0]
        max_thickness_label = f"{max(active_thicknesses):g} cm" if active_thicknesses else "—"
        target_mc = st.number_input(
            f"Target moisture content for the thickest boards ({max_thickness_label}) (%)",
            min_value=1.0,
            max_value=40.0,
            value=11.0,
            step=0.5,
            help="A setpoint defined before drying, not a measurement obtained after the process.",
        )

        submitted = st.form_submit_button("Predict drying duration", type="primary", use_container_width=True)

    if submitted:
        if weather_data is None:
            st.error("The 14-day weather data are unavailable. Retrieve them automatically or use Manual mode.")
        elif weather_data["suhu_maks_c"] < weather_data["suhu_min_c"]:
            st.error("Average maximum temperature cannot be lower than average minimum temperature.")
        else:
            try:
                features, feature_values = build_feature_row(
                    contract=contract,
                    species=species,
                    kiln=kiln,
                    volume=volume,
                    total_boards=int(total_boards),
                    start_day=start_date,
                    counts=counts,
                    target_mc=target_mc,
                    weather=weather_data,
                )
                model = load_model()
                prediction = float(model.predict(features)[0])
                if not math.isfinite(prediction):
                    raise ValueError("The model returned a non-finite prediction.")

                planning_days = max(1, math.ceil(prediction))
                completion_date = start_date + timedelta(days=planning_days)
                warnings = range_warnings(feature_values)

                r1, r2, r3 = st.columns(3)
                r1.metric("Model prediction", f"{prediction:.1f} days")
                r2.metric("Planning duration", f"{planning_days} days")
                r3.metric("Planned completion date", completion_date.strftime("%d %b %Y"))
                st.caption(
                    "The planning duration is rounded up to a full calendar day. "
                    "The final decision remains subject to operator review."
                )

                if warnings:
                    st.warning("\n".join(["Inputs outside the historical range:"] + [f"- {w}" for w in warnings]))

                result_record = {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "model_version": contract.get("version"),
                    "model": contract.get("selected_model"),
                    "start_date": start_date.isoformat(),
                    "weather_window_end": end_date.isoformat(),
                    "weather_source": weather_data["provider"],
                    "species": species,
                    "kiln": kiln,
                    "volume_m3": float(volume),
                    "total_boards": int(total_boards),
                    "maximum_thickness_cm": feature_values["ket_max"],
                    "target_mc_pct": float(target_mc),
                    "predicted_duration_days": prediction,
                    "planning_duration_days": planning_days,
                    "planned_completion_date": completion_date.isoformat(),
                    "review_status": "operator review required",
                }
                st.download_button(
                    "Download prediction result (JSON)",
                    data=json.dumps(result_record, indent=2, ensure_ascii=False),
                    file_name=f"kiln_prediction_{start_date.isoformat()}.json",
                    mime="application/json",
                )
            except Exception as exc:
                st.error(f"The prediction could not be generated: {exc}")

    st.divider()
    st.caption(
        "This application is a decision-support prototype based on historical data. "
        "It does not replace moisture-content inspection, kiln safety procedures, "
        "or decisions made by authorized operators."
    )

with model_tab:
    st.subheader("Model and evaluation summary")
    summary = audit_summary(audit_metrics, bootstrap_samples)
    if summary:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Audit MAE", f"{summary['mae']:.2f} days")
        m2.metric("Audit RMSE", f"{summary['rmse']:.2f} days")
        m3.metric("Audit R²", f"{summary['r2']:.3f}")
        m4.metric("Audit MAPE", f"{summary['mape_pct']:.2f}%")
        if "mae_ci_low" in summary:
            st.caption(
                f"Historical MAE 95% bootstrap percentile interval: "
                f"{summary['mae_ci_low']:.2f}–{summary['mae_ci_high']:.2f} days. "
                "This is not a prediction interval for an individual batch."
            )
    else:
        st.info("Audit metrics are unavailable.")

    st.markdown(
        f"""
        - Target: **kiln-drying duration (`{contract.get('prediction_target', 'durasi_hari')}`)**.
        - Unit of analysis: **one batch**.
        - Training and evaluation batches: **{contract.get('n_batches', '—')}**.
        - Locked temporal audit: **the most recent {contract.get('n_audit', '—')} batches**.
        - Primary strategy: **{contract.get('primary_training_strategy', '—')}**.
        - Deployment status: **{contract.get('deployment_status', '—')}**.
        """
    )
    with st.expander("Model feature list"):
        st.code("\n".join(contract.get("selected_features", [])), language=None)
