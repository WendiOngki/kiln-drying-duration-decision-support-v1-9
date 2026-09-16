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

# Rentang observasi pelatihan. Digunakan hanya untuk memberi peringatan,
# bukan untuk mengubah input pengguna secara otomatis.
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


st.set_page_config(
    page_title="Kiln-Drying Duration Decision Support",
    page_icon="🌲",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
    .result-card {
        padding: 1.1rem 1.3rem; border-radius: 0.8rem;
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
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_contract() -> dict[str, Any]:
    if not CONTRACT_PATH.exists():
        raise FileNotFoundError(f"Kontrak model tidak ditemukan: {CONTRACT_PATH}")
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_audit_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(AUDIT_METRICS_PATH) if AUDIT_METRICS_PATH.exists() else pd.DataFrame()
    bootstrap = pd.read_csv(BOOTSTRAP_PATH) if BOOTSTRAP_PATH.exists() else pd.DataFrame()
    return metrics, bootstrap


def get_weather_location() -> tuple[float, float, str] | None:
    """Ambil koordinat dari secrets atau environment tanpa menampilkannya di UI."""
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
    label = label or os.getenv("WEATHER_LOCATION_LABEL", "Lokasi studi")

    if lat is None or lon is None:
        return None
    try:
        return float(lat), float(lon), str(label)
    except (TypeError, ValueError):
        return None


def _mean_valid(values: list[Any], variable_name: str) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").replace(-999, np.nan).dropna()
    if series.empty:
        raise ValueError(f"Tidak ada nilai valid untuk {variable_name}.")
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
        raise ValueError("Open-Meteo tidak mengembalikan data harian yang dapat digunakan.")

    daily["time"] = pd.to_datetime(daily["time"]).dt.date
    start_day = date.fromisoformat(start_date_iso)
    end_day = date.fromisoformat(end_date_iso)
    window = daily.loc[daily["time"].between(start_day, end_day)].copy()
    if len(window) != 14 or window["time"].nunique() != 14:
        raise ValueError(
            "Prakiraan otomatis belum mencakup 14 hari penuh dari tanggal mulai. "
            "Gunakan mode manual atau pilih tanggal yang berada dalam horizon prakiraan."
        )

    return {
        "kelembaban_pct": _mean_valid(
            window["relative_humidity_2m_mean"].tolist(), "kelembapan relatif"
        ),
        "curah_hujan_mm": _mean_valid(
            window["precipitation_sum"].tolist(), "curah hujan"
        ),
        "suhu_maks_c": _mean_valid(
            window["temperature_2m_max"].tolist(), "suhu maksimum"
        ),
        "suhu_min_c": _mean_valid(
            window["temperature_2m_min"].tolist(), "suhu minimum"
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
        raise ValueError(f"NASA POWER tidak mengembalikan variabel: {', '.join(missing)}")

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
                f"NASA POWER belum menyediakan 14 nilai valid untuk {name}. "
                "Gunakan mode manual untuk jendela ini."
            )

    return {
        "kelembaban_pct": _mean_valid(
            [parameters["RH2M"][key] for key in expected_keys], "kelembapan relatif"
        ),
        "curah_hujan_mm": _mean_valid(
            [parameters["PRECTOTCORR"][key] for key in expected_keys], "curah hujan"
        ),
        "suhu_maks_c": _mean_valid(
            [parameters["T2M_MAX"][key] for key in expected_keys], "suhu maksimum"
        ),
        "suhu_min_c": _mean_valid(
            [parameters["T2M_MIN"][key] for key in expected_keys], "suhu minimum"
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
        "Jendela 14 hari ini memotong periode historis dan prakiraan, atau berada di luar "
        "horizon prakiraan. Pilih mode manual agar sumber data tidak tercampur secara diam-diam."
    )


def weighted_thickness_stats(counts: dict[float, int]) -> dict[str, Any]:
    active = [(thickness, count) for thickness, count in counts.items() if count > 0]
    if not active:
        raise ValueError("Masukkan sedikitnya satu lembar pada komposisi ketebalan.")
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
            f"Jumlah komposisi ({stats['total_composition']:,}) harus sama dengan "
            f"total lembar ({total_boards:,})."
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
        raise KeyError(f"Fitur aplikasi belum lengkap: {missing}")
    return pd.DataFrame([row], columns=selected_features), row


def range_warnings(row: dict[str, Any]) -> list[str]:
    warnings = []
    for feature, (low, high) in TRAINING_RANGES.items():
        value = float(row[feature])
        if value < low or value > high:
            warnings.append(
                f"{feature} = {value:.2f} berada di luar rentang observasi "
                f"pelatihan ({low:.2f}–{high:.2f})."
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
            "Variabel": [
                "Kelembapan relatif rata-rata",
                "Curah hujan harian rata-rata",
                "Suhu maksimum harian rata-rata",
                "Suhu minimum harian rata-rata",
            ],
            "Nilai": [
                f"{weather['kelembaban_pct']:.2f} %",
                f"{weather['curah_hujan_mm']:.2f} mm/hari",
                f"{weather['suhu_maks_c']:.2f} °C",
                f"{weather['suhu_min_c']:.2f} °C",
            ],
        }
    )


try:
    contract = load_contract()
    audit_metrics, bootstrap_samples = load_audit_data()
except Exception as exc:
    st.error(f"Artefak aplikasi tidak dapat dibaca: {exc}")
    st.stop()

st.title("Kiln-Drying Duration Decision Support")
st.caption(
    "Prototipe pendukung keputusan untuk memperkirakan durasi pengeringan batch kayu "
    "pada sebuah perusahaan manufaktur furnitur kayu di Kabupaten Klaten, Indonesia."
)

with st.sidebar:
    st.subheader("Model")
    st.write(f"Versi: **{contract.get('version', '—')}**")
    st.write(f"Algoritme: **{contract.get('selected_model', '—')}**")
    st.write("Pelatihan utama: **batch observasi asli**")
    st.info(
        "Model berstatus kandidat. Hasil perlu ditinjau operator dan divalidasi "
        "secara prospektif sebelum penggunaan operasional penuh."
    )

prediction_tab, model_tab = st.tabs(["Prediksi", "Informasi model"])

with prediction_tab:
    st.subheader("1. Tanggal mulai dan cuaca 14 hari")
    date_col, mode_col = st.columns([1, 1])
    with date_col:
        start_date = st.date_input("Tanggal mulai pengeringan", value=date.today())
    with mode_col:
        weather_mode = st.radio(
            "Sumber data cuaca",
            ["Otomatis", "Manual"],
            horizontal=True,
            help=(
                "Otomatis memakai NASA POWER untuk jendela historis dan Open-Meteo "
                "untuk jendela prakiraan 14 hari."
            ),
        )

    end_date = start_date + timedelta(days=13)
    st.caption(
        f"Jendela cuaca model: {start_date.strftime('%d %b %Y')} – "
        f"{end_date.strftime('%d %b %Y')} (14 hari kalender)."
    )

    weather_data: dict[str, Any] | None = None
    weather_key = f"{start_date.isoformat()}::{end_date.isoformat()}"

    if weather_mode == "Otomatis":
        location = get_weather_location()
        if location is None:
            st.warning(
                "Koordinat cuaca belum dikonfigurasi. Isi `.streamlit/secrets.toml` "
                "atau pilih mode Manual. Koordinat tidak ditampilkan di antarmuka."
            )
        elif st.button("Ambil dan ringkas cuaca 14 hari", type="secondary"):
            latitude, longitude, location_label = location
            try:
                with st.spinner("Mengambil data cuaca..."):
                    fetched = fetch_weather_auto(start_date, latitude, longitude)
                fetched["window_key"] = weather_key
                fetched["location_label"] = location_label
                st.session_state["weather_auto"] = fetched
            except Exception as exc:
                st.error(f"Data cuaca otomatis tidak dapat digunakan: {exc}")

        stored_weather = st.session_state.get("weather_auto")
        if stored_weather and stored_weather.get("window_key") == weather_key:
            weather_data = stored_weather
            st.success(
                f"Cuaca tersedia dari {weather_data['provider']} untuk "
                f"{weather_data.get('location_label', 'lokasi studi')}."
            )
            st.dataframe(weather_table(weather_data), hide_index=True, use_container_width=True)
        elif stored_weather:
            st.info("Tanggal berubah. Ambil kembali cuaca agar jendela 14 hari sesuai.")
    else:
        st.markdown(
            '<p class="small-note">Masukkan rata-rata harian untuk jendela 14 hari di atas.</p>',
            unsafe_allow_html=True,
        )
        w1, w2, w3, w4 = st.columns(4)
        with w1:
            humidity = st.number_input(
                "Kelembapan rata-rata (%)", 0.0, 100.0, 85.70, 0.10
            )
        with w2:
            rainfall = st.number_input(
                "Curah hujan (mm/hari)", 0.0, 500.0, 6.70, 0.10
            )
        with w3:
            max_temp = st.number_input("Suhu maksimum rata-rata (°C)", -20.0, 60.0, 29.23, 0.10)
        with w4:
            min_temp = st.number_input("Suhu minimum rata-rata (°C)", -30.0, 50.0, 21.81, 0.10)
        weather_data = {
            "kelembaban_pct": humidity,
            "curah_hujan_mm": rainfall,
            "suhu_maks_c": max_temp,
            "suhu_min_c": min_temp,
            "provider": "Input manual",
            "window_start": start_date.isoformat(),
            "window_end": end_date.isoformat(),
        }

    st.divider()
    st.subheader("2. Spesifikasi batch")
    with st.form("prediction_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            species = st.selectbox("Jenis kayu", SPECIES)
        with c2:
            kiln = st.selectbox("Nomor kiln", KILNS)
        with c3:
            volume = st.number_input("Volume total (m³)", 0.001, 100.0, 12.0, 0.10)
        with c4:
            total_boards = st.number_input("Total lembar", 1, 10000, 750, 1)

        st.markdown("**Komposisi jumlah lembar per ketebalan**")
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
            f"Target MC papan paling tebal ({max_thickness_label}) (%)",
            min_value=1.0,
            max_value=40.0,
            value=11.0,
            step=0.5,
            help="Setpoint yang sudah ditentukan sebelum proses, bukan hasil pengukuran setelah proses.",
        )

        submitted = st.form_submit_button("Prediksi durasi", type="primary", use_container_width=True)

    if submitted:
        if weather_data is None:
            st.error("Data cuaca 14 hari belum tersedia. Ambil data otomatis atau pilih mode Manual.")
        elif weather_data["suhu_maks_c"] < weather_data["suhu_min_c"]:
            st.error("Suhu maksimum rata-rata tidak boleh lebih rendah daripada suhu minimum rata-rata.")
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
                    raise ValueError("Model menghasilkan nilai prediksi yang tidak valid.")

                planning_days = max(1, math.ceil(prediction))
                completion_date = start_date + timedelta(days=planning_days)
                warnings = range_warnings(feature_values)

                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                r1, r2, r3 = st.columns(3)
                r1.metric("Prediksi model", f"{prediction:.1f} hari")
                r2.metric("Durasi rencana", f"{planning_days} hari")
                r3.metric("Tanggal selesai rencana", completion_date.strftime("%d %b %Y"))
                st.caption(
                    "Durasi rencana dibulatkan ke atas ke hari kalender penuh. "
                    "Keputusan akhir tetap memerlukan penilaian operator."
                )
                st.markdown("</div>", unsafe_allow_html=True)

                if warnings:
                    st.warning("\n".join(["Input di luar cakupan historis:"] + [f"- {w}" for w in warnings]))

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
                    "Unduh hasil prediksi (JSON)",
                    data=json.dumps(result_record, indent=2, ensure_ascii=False),
                    file_name=f"prediksi_kiln_{start_date.isoformat()}.json",
                    mime="application/json",
                )
            except Exception as exc:
                st.error(f"Prediksi tidak dapat dibuat: {exc}")

    st.divider()
    st.caption(
        "Aplikasi ini adalah prototipe pendukung keputusan berbasis data historis. "
        "Aplikasi tidak menggantikan inspeksi kadar air, prosedur keselamatan kiln, "
        "atau keputusan operator yang berwenang."
    )

with model_tab:
    st.subheader("Ringkasan model dan evaluasi")
    summary = audit_summary(audit_metrics, bootstrap_samples)
    if summary:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Audit MAE", f"{summary['mae']:.2f} hari")
        m2.metric("Audit RMSE", f"{summary['rmse']:.2f} hari")
        m3.metric("Audit R²", f"{summary['r2']:.3f}")
        m4.metric("Audit MAPE", f"{summary['mape_pct']:.2f}%")
        if "mae_ci_low" in summary:
            st.caption(
                f"Interval persentil bootstrap 95% untuk MAE historis: "
                f"{summary['mae_ci_low']:.2f}–{summary['mae_ci_high']:.2f} hari. "
                "Ini bukan interval prediksi untuk satu batch."
            )
    else:
        st.info("Ringkasan metrik audit tidak tersedia.")

    st.markdown(
        f"""
        - Target: **{contract.get('prediction_target', 'durasi_hari')}**.
        - Unit analisis: **satu batch**.
        - Batch pelatihan dan evaluasi: **{contract.get('n_batches', '—')}**.
        - Audit temporal terkunci: **{contract.get('n_audit', '—')} batch terbaru**.
        - Strategi utama: **{contract.get('primary_training_strategy', '—')}**.
        - Status penerapan: **{contract.get('deployment_status', '—')}**.
        """
    )
    with st.expander("Daftar fitur model"):
        st.code("\n".join(contract.get("selected_features", [])), language=None)

