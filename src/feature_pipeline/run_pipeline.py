"""
Feature pipeline orchestrator.
Fetches data from APIs, engineers features, and stores in Feature Store.
Entry point for both manual runs and CI/CD automation.
"""
import sys
import logging
import argparse

from src.feature_pipeline.fetch_data import (
    fetch_current_air_pollution,
    fetch_current_weather,
    fetch_air_pollution_forecast,
)
from src.feature_pipeline.feature_engineering import engineer_features
from src.feature_pipeline.feature_store import insert_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_current_pipeline(use_hopsworks: bool = True) -> bool:
    """
    Run the feature pipeline for current data.
    Fetches current air pollution + weather, engineers features, stores them.
    """
    logger.info("=" * 60)
    logger.info("Starting Feature Pipeline (current data)")
    logger.info("=" * 60)

    # Step 1: Fetch air pollution data
    logger.info("Step 1: Fetching current air pollution data...")
    air_df = fetch_current_air_pollution()
    if air_df.empty:
        logger.error("No air pollution data returned. Aborting.")
        return False
    logger.info(f"  Got {len(air_df)} air pollution records")

    # Step 2: Fetch weather data
    logger.info("Step 2: Fetching current weather data...")
    try:
        weather_df = fetch_current_weather()
        if not weather_df.empty:
            for col in weather_df.columns:
                if col != "datetime":
                    air_df[col] = weather_df[col].iloc[0]
            logger.info(f"  Added weather features: {list(weather_df.columns)}")
        else:
            logger.warning("  No weather data returned, proceeding without it")
    except Exception as e:
        logger.warning(f"  Weather fetch failed: {e}, proceeding without it")

    # Step 3: Engineer features
    logger.info("Step 3: Engineering features...")
    features_df = engineer_features(air_df)
    logger.info(f"  Feature matrix: {features_df.shape[0]} rows × {features_df.shape[1]} columns")

    # Step 4: Store features
    logger.info("Step 4: Storing features...")
    success = insert_features(features_df, use_hopsworks=use_hopsworks)

    if success:
        logger.info("Feature pipeline completed successfully!")
    else:
        logger.error("Feature pipeline failed at storage step")

    return success


def run_forecast_pipeline(use_hopsworks: bool = True) -> bool:
    """
    Run the feature pipeline for forecast data (next 5 days hourly).
    Used by the inference pipeline to get future weather/pollution forecasts.
    """
    logger.info("=" * 60)
    logger.info("Starting Feature Pipeline (forecast data)")
    logger.info("=" * 60)

    # Step 1: Fetch forecast air pollution data
    logger.info("Step 1: Fetching air pollution forecast...")
    forecast_df = fetch_air_pollution_forecast()
    if forecast_df.empty:
        logger.error("No forecast data returned. Aborting.")
        return False
    logger.info(f"  Got {len(forecast_df)} forecast records")

    # Step 2: Engineer features (limited — no lag/rolling for future)
    logger.info("Step 2: Engineering forecast features...")
    from src.feature_pipeline.feature_engineering import compute_aqi, add_time_features
    forecast_df["aqi"] = forecast_df.apply(compute_aqi, axis=1)
    forecast_df = add_time_features(forecast_df)

    # Set weather defaults for forecast (if not available)
    for col in ["temperature", "humidity", "wind_speed", "pressure"]:
        if col not in forecast_df.columns:
            forecast_df[col] = 0

    logger.info(f"  Forecast features: {forecast_df.shape}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the AQI Feature Pipeline")
    parser.add_argument(
        "--mode",
        choices=["current", "forecast", "both"],
        default="current",
        help="Pipeline mode: current data, forecast, or both",
    )
    parser.add_argument(
        "--no-hopsworks",
        action="store_true",
        help="Skip Hopsworks and use local storage only",
    )
    args = parser.parse_args()

    use_hopsworks = not args.no_hopsworks

    success = True

    if args.mode in ("current", "both"):
        if not run_current_pipeline(use_hopsworks=use_hopsworks):
            success = False

    if args.mode in ("forecast", "both"):
        if not run_forecast_pipeline(use_hopsworks=use_hopsworks):
            success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
