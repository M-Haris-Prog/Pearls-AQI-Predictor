"""
Fetch AQI and weather data from OpenWeatherMap APIs.
Supports current, forecast, and historical air pollution data.
"""
import time
import logging
from datetime import datetime
from typing import Optional

import requests
import pandas as pd

from src.config import (
    OPENWEATHER_API_KEY, CITY_LAT, CITY_LON,
    OWM_AIR_POLLUTION_URL, OWM_AIR_POLLUTION_HISTORY_URL,
    OWM_AIR_POLLUTION_FORECAST_URL, OWM_WEATHER_URL, OWM_FORECAST_URL,
)

logger = logging.getLogger(__name__)


def _make_request(url: str, params: dict, max_retries: int = 3) -> dict:
    """Make an API request with retry logic and exponential backoff."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            wait_time = 2 ** attempt
            logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}. "
                           f"Retrying in {wait_time}s...")
            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} attempts failed for {url}")
                raise


def fetch_current_air_pollution() -> pd.DataFrame:
    """Fetch current air pollution data for the configured city."""
    params = {
        "lat": CITY_LAT,
        "lon": CITY_LON,
        "appid": OPENWEATHER_API_KEY,
    }
    data = _make_request(OWM_AIR_POLLUTION_URL, params)
    return _parse_air_pollution_response(data)


def fetch_air_pollution_forecast() -> pd.DataFrame:
    """Fetch 5-day air pollution forecast (hourly)."""
    params = {
        "lat": CITY_LAT,
        "lon": CITY_LON,
        "appid": OPENWEATHER_API_KEY,
    }
    data = _make_request(OWM_AIR_POLLUTION_FORECAST_URL, params)
    return _parse_air_pollution_response(data)


def fetch_air_pollution_history(start_ts: int, end_ts: int) -> pd.DataFrame:
    """
    Fetch historical air pollution data between two Unix timestamps.

    Args:
        start_ts: Start Unix timestamp
        end_ts: End Unix timestamp
    """
    params = {
        "lat": CITY_LAT,
        "lon": CITY_LON,
        "start": start_ts,
        "end": end_ts,
        "appid": OPENWEATHER_API_KEY,
    }
    data = _make_request(OWM_AIR_POLLUTION_HISTORY_URL, params)
    return _parse_air_pollution_response(data)


def fetch_current_weather() -> pd.DataFrame:
    """Fetch current weather data for the configured city."""
    params = {
        "lat": CITY_LAT,
        "lon": CITY_LON,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }
    data = _make_request(OWM_WEATHER_URL, params)
    return _parse_weather_response(data)


def fetch_weather_forecast() -> pd.DataFrame:
    """Fetch 5-day / 3-hour weather forecast."""
    params = {
        "lat": CITY_LAT,
        "lon": CITY_LON,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }
    data = _make_request(OWM_FORECAST_URL, params)
    return _parse_weather_forecast_response(data)


def fetch_combined_data(timestamp: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch and combine air pollution + weather data into a single DataFrame.
    If timestamp is None, fetches current data. Otherwise used for historical.
    """
    if timestamp is None:
        air_df = fetch_current_air_pollution()
        weather_df = fetch_current_weather()
    else:
        # For historical, fetch air pollution in a small window
        air_df = fetch_air_pollution_history(timestamp, timestamp + 3600)
        weather_df = fetch_current_weather()  # Historical weather not available in free tier

    if air_df.empty:
        logger.warning("No air pollution data returned")
        return pd.DataFrame()

    # Merge on nearest timestamp
    if not weather_df.empty:
        # Use the first weather record for current data
        for col in weather_df.columns:
            if col != "datetime":
                air_df[col] = weather_df[col].iloc[0]

    return air_df


# ─── Response Parsers ─────────────────────────────────────────────

def _parse_air_pollution_response(data: dict) -> pd.DataFrame:
    """Parse OpenWeatherMap air pollution API response into DataFrame."""
    if "list" not in data or len(data["list"]) == 0:
        logger.warning("Empty air pollution response")
        return pd.DataFrame()

    records = []
    for item in data["list"]:
        record = {
            "datetime": datetime.utcfromtimestamp(item["dt"]),
            "unix_timestamp": item["dt"],
            "aqi_index": item["main"]["aqi"],  # OWM's 1-5 scale
            "co": item["components"].get("co", 0),
            "no": item["components"].get("no", 0),
            "no2": item["components"].get("no2", 0),
            "o3": item["components"].get("o3", 0),
            "so2": item["components"].get("so2", 0),
            "pm2_5": item["components"].get("pm2_5", 0),
            "pm10": item["components"].get("pm10", 0),
            "nh3": item["components"].get("nh3", 0),
        }
        records.append(record)

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def _parse_weather_response(data: dict) -> pd.DataFrame:
    """Parse OpenWeatherMap current weather API response."""
    record = {
        "datetime": datetime.utcfromtimestamp(data["dt"]),
        "temperature": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "wind_speed": data["wind"]["speed"],
        "wind_deg": data["wind"].get("deg", 0),
        "visibility": data.get("visibility", 10000) / 1000,  # Convert m to km
        "clouds": data["clouds"]["all"],
    }
    return pd.DataFrame([record])


def _parse_weather_forecast_response(data: dict) -> pd.DataFrame:
    """Parse OpenWeatherMap 5-day forecast API response."""
    records = []
    for item in data.get("list", []):
        record = {
            "datetime": datetime.utcfromtimestamp(item["dt"]),
            "unix_timestamp": item["dt"],
            "temperature": item["main"]["temp"],
            "feels_like": item["main"]["feels_like"],
            "humidity": item["main"]["humidity"],
            "pressure": item["main"]["pressure"],
            "wind_speed": item["wind"]["speed"],
            "wind_deg": item["wind"].get("deg", 0),
            "visibility": item.get("visibility", 10000) / 1000,
            "clouds": item["clouds"]["all"],
        }
        records.append(record)

    return pd.DataFrame(records)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Fetching current air pollution data...")
    df = fetch_current_air_pollution()
    print(df)
    print("\nFetching current weather...")
    wdf = fetch_current_weather()
    print(wdf)
