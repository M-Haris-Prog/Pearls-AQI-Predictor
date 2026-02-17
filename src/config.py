"""
Central configuration for the AQI Predictor project.
Loads environment variables and defines constants.
Supports both .env files (local) and Streamlit secrets (cloud deployment).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env (local development)
load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    """Get a secret from environment variables or Streamlit secrets."""
    value = os.getenv(key, "")
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


# ─── Project Paths ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── API Keys ────────────────────────────────────────────────────
OPENWEATHER_API_KEY = _get_secret("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = _get_secret("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = _get_secret("HOPSWORKS_PROJECT_NAME")

# ─── City Configuration ─────────────────────────────────────────
CITY_NAME = os.getenv("CITY_NAME", "Lahore")
CITY_LAT = float(os.getenv("CITY_LAT", "31.5204"))
CITY_LON = float(os.getenv("CITY_LON", "74.3587"))

# ─── OpenWeatherMap API Endpoints ────────────────────────────────
OWM_BASE_URL = "http://api.openweathermap.org/data/2.5"
OWM_AIR_POLLUTION_URL = f"{OWM_BASE_URL}/air_pollution"
OWM_AIR_POLLUTION_HISTORY_URL = f"{OWM_BASE_URL}/air_pollution/history"
OWM_AIR_POLLUTION_FORECAST_URL = f"{OWM_BASE_URL}/air_pollution/forecast"
OWM_WEATHER_URL = f"{OWM_BASE_URL}/weather"
OWM_FORECAST_URL = f"{OWM_BASE_URL}/forecast"

# ─── Feature Store Configuration ─────────────────────────────────
FEATURE_GROUP_NAME = "lahore_aqi_features"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "lahore_aqi_fv"
FEATURE_VIEW_VERSION = 1
MODEL_REGISTRY_NAME = "aqi_predictor"

# ─── AQI Categories ──────────────────────────────────────────────
AQI_CATEGORIES = {
    "Good": {"min": 0, "max": 50, "color": "#00E400", "health": "Air quality is satisfactory."},
    "Moderate": {"min": 51, "max": 100, "color": "#FFFF00", "health": "Acceptable; moderate health concern for sensitive individuals."},
    "Unhealthy for Sensitive Groups": {"min": 101, "max": 150, "color": "#FF7E00", "health": "Members of sensitive groups may experience health effects."},
    "Unhealthy": {"min": 151, "max": 200, "color": "#FF0000", "health": "Everyone may begin to experience health effects."},
    "Very Unhealthy": {"min": 201, "max": 300, "color": "#8F3F97", "health": "Health alert: everyone may experience serious health effects."},
    "Hazardous": {"min": 301, "max": 500, "color": "#7E0023", "health": "Health warnings of emergency conditions."},
}

# ─── Model Configuration ─────────────────────────────────────────
FORECAST_DAYS = 3
SEQUENCE_LENGTH = 24  # hours of history for LSTM input
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ─── Feature Lists ───────────────────────────────────────────────
POLLUTANT_FEATURES = ["pm2_5", "pm10", "no2", "so2", "co", "o3"]
WEATHER_FEATURES = ["temperature", "humidity", "wind_speed", "pressure"]
TIME_FEATURES = ["hour", "day_of_week", "month", "is_weekend", "season"]
DERIVED_FEATURES = [
    "aqi_change_rate", "pm25_pm10_ratio",
    "rolling_aqi_6h", "rolling_aqi_12h", "rolling_aqi_24h",
]
LAG_FEATURES = ["aqi_lag_1", "aqi_lag_3", "aqi_lag_6", "aqi_lag_12", "aqi_lag_24"]

ALL_FEATURES = (POLLUTANT_FEATURES + WEATHER_FEATURES + TIME_FEATURES
                + DERIVED_FEATURES + LAG_FEATURES)
TARGET = "aqi"
