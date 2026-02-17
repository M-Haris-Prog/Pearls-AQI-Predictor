"""
Inference pipeline for 3-day AQI prediction.
Loads the best model and generates forecasts.
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import joblib

from src.config import (
    MODELS_DIR, ALL_FEATURES, TARGET, FORECAST_DAYS,
    SEQUENCE_LENGTH, CITY_NAME, AQI_CATEGORIES,
)
from src.feature_pipeline.feature_store import get_features
from src.feature_pipeline.fetch_data import (
    fetch_current_air_pollution, fetch_current_weather,
    fetch_weather_forecast, fetch_air_pollution_forecast,
)
from src.feature_pipeline.feature_engineering import (
    engineer_features, compute_aqi, add_time_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_aqi_category(aqi_value: float) -> Dict:
    """Get the AQI category info for a given AQI value."""
    for category, info in AQI_CATEGORIES.items():
        if info["min"] <= aqi_value <= info["max"]:
            return {"category": category, **info}
    return {"category": "Hazardous", **AQI_CATEGORIES["Hazardous"]}


def load_best_model():
    """
    Load the best model from local storage.
    Returns (model, metadata, model_type).
    """
    best_dir = MODELS_DIR / "best"

    if not best_dir.exists():
        # Try to find any available model
        for model_type in ["xgboost", "random_forest", "ridge", "lstm"]:
            model_dir = MODELS_DIR / model_type
            if model_dir.exists():
                best_dir = model_dir
                break
        else:
            raise FileNotFoundError("No trained model found. Run the training pipeline first.")

    # Load metadata
    metadata_path = best_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    model_name = metadata.get("model_name", "unknown")

    # Load model
    if model_name == "lstm" or (best_dir / "model.h5").exists():
        from src.training_pipeline.models.lstm_model import load_model
        model = load_model(str(best_dir / "model.h5"))
        return model, metadata, "lstm"
    else:
        model = joblib.load(best_dir / "model.pkl")
        return model, metadata, model_name


def load_model_by_name(model_name: str):
    """Load a specific model by name."""
    model_dir = MODELS_DIR / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"Model '{model_name}' not found at {model_dir}")

    metadata_path = model_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    if model_name == "lstm" or (model_dir / "model.h5").exists():
        from src.training_pipeline.models.lstm_model import load_model
        model = load_model(str(model_dir / "model.h5"))
    else:
        model = joblib.load(model_dir / "model.pkl")

    return model, metadata


def prepare_forecast_features() -> pd.DataFrame:
    """
    Prepare feature vectors for the next 3 days.
    Uses weather forecast API + air pollution forecast from OpenWeatherMap.
    Falls back to last known values if API calls fail.
    """
    logger.info("Preparing forecast features...")

    # Try to get forecast data from APIs
    forecast_features = []

    try:
        air_forecast = fetch_air_pollution_forecast()
        weather_forecast = fetch_weather_forecast()
    except Exception as e:
        logger.warning(f"Could not fetch forecasts: {e}. Using last known values.")
        air_forecast = pd.DataFrame()
        weather_forecast = pd.DataFrame()

    # Get historical features for lag/rolling computations
    historical = get_features(use_hopsworks=False)

    now = datetime.utcnow()

    for day in range(1, FORECAST_DAYS + 1):
        target_date = now + timedelta(days=day)

        # Try to find forecast data for this day
        day_features = {}

        if not air_forecast.empty:
            # Get average predictions for the target day
            day_mask = air_forecast["datetime"].dt.date == target_date.date()
            day_air = air_forecast[day_mask]
            if not day_air.empty:
                for col in ["pm2_5", "pm10", "no2", "so2", "co", "o3"]:
                    if col in day_air.columns:
                        day_features[col] = day_air[col].mean()

        if not weather_forecast.empty:
            day_mask = weather_forecast["datetime"].dt.date == target_date.date()
            day_weather = weather_forecast[day_mask]
            if not day_weather.empty:
                for col in ["temperature", "humidity", "wind_speed", "pressure"]:
                    if col in day_weather.columns:
                        day_features[col] = day_weather[col].mean()

        # Fill missing with last known values
        if not historical.empty:
            last_row = historical.iloc[-1]
            for col in ALL_FEATURES:
                if col not in day_features and col in last_row.index:
                    day_features[col] = last_row[col]

        # Add time features
        day_features["hour"] = 12  # Midday average
        day_features["day_of_week"] = target_date.weekday()
        day_features["month"] = target_date.month
        day_features["is_weekend"] = 1 if target_date.weekday() >= 5 else 0
        month = target_date.month
        day_features["season"] = (
            1 if month in [12, 1, 2] else
            2 if month in [3, 4, 5] else
            3 if month in [6, 7, 8] else 4
        )

        # Default values for derived/lag features
        for col in ALL_FEATURES:
            if col not in day_features:
                day_features[col] = 0

        day_features["forecast_date"] = target_date.strftime("%Y-%m-%d")
        day_features["forecast_day"] = day

        forecast_features.append(day_features)

    return pd.DataFrame(forecast_features)


def predict_next_3_days(model_name: str = None) -> Dict:
    """
    Generate AQI predictions for the next 3 days.

    Args:
        model_name: Specific model to use (None = best model)

    Returns:
        Dictionary with predictions, categories, and metadata
    """
    logger.info("=" * 60)
    logger.info(f"Generating 3-day AQI Forecast for {CITY_NAME}")
    logger.info("=" * 60)

    # Load model
    if model_name:
        model, metadata = load_model_by_name(model_name)
        model_type = model_name
    else:
        model, metadata, model_type = load_best_model()

    logger.info(f"Using model: {model_type}")

    # Prepare features
    forecast_df = prepare_forecast_features()

    # Select features the model was trained on
    feature_cols = [c for c in ALL_FEATURES if c in forecast_df.columns]
    X_forecast = forecast_df[feature_cols].values

    # Generate predictions
    if model_type == "lstm":
        # LSTM needs sequence input
        historical = get_features(use_hopsworks=False)
        if not historical.empty:
            hist_features = historical[feature_cols].values
            scaler_path = MODELS_DIR / "lstm_scaler.pkl"
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                hist_features = scaler.transform(hist_features)

            # Use last N hours as input sequence
            seq_len = min(SEQUENCE_LENGTH, len(hist_features))
            X_input = hist_features[-seq_len:].reshape(1, seq_len, len(feature_cols))
            predictions = model.predict(X_input, verbose=0)[0]
        else:
            predictions = model.predict(X_forecast.reshape(1, -1, len(feature_cols)), verbose=0)[0]
    else:
        predictions = model.predict(X_forecast)

    # Build result
    now = datetime.utcnow()
    result = {
        "city": CITY_NAME,
        "generated_at": now.isoformat(),
        "model": model_type,
        "model_metrics": metadata.get("metrics", {}),
        "predictions": [],
    }

    for i, pred_value in enumerate(predictions[:FORECAST_DAYS]):
        pred_aqi = max(0, float(pred_value))
        target_date = now + timedelta(days=i + 1)
        category_info = get_aqi_category(pred_aqi)

        prediction = {
            "day": i + 1,
            "date": target_date.strftime("%Y-%m-%d"),
            "day_name": target_date.strftime("%A"),
            "predicted_aqi": round(pred_aqi),
            "category": category_info["category"],
            "color": category_info["color"],
            "health_advisory": category_info["health"],
            "is_hazardous": pred_aqi > 150,
        }
        result["predictions"].append(prediction)

        logger.info(
            f"  Day {i + 1} ({prediction['day_name']}): "
            f"AQI {prediction['predicted_aqi']} — {prediction['category']}"
        )

    # Check for alerts
    hazardous_days = [p for p in result["predictions"] if p["is_hazardous"]]
    result["alert"] = len(hazardous_days) > 0
    result["alert_message"] = (
        f"⚠️ Air quality is predicted to be unhealthy on "
        f"{', '.join(p['day_name'] for p in hazardous_days)}. "
        f"Sensitive groups should limit outdoor activity."
        if hazardous_days else ""
    )

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate AQI predictions")
    parser.add_argument("--model", type=str, default=None, help="Model name to use")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    args = parser.parse_args()

    result = predict_next_3_days(model_name=args.model)

    # Print results
    print(f"\n{'='*50}")
    print(f"  3-Day AQI Forecast — {result['city']}")
    print(f"{'='*50}")

    for pred in result["predictions"]:
        emoji = "🟢" if pred["predicted_aqi"] <= 50 else \
                "🟡" if pred["predicted_aqi"] <= 100 else \
                "🟠" if pred["predicted_aqi"] <= 150 else \
                "🔴" if pred["predicted_aqi"] <= 200 else "🟣"
        print(f"  {emoji} {pred['day_name']:10s} ({pred['date']}): "
              f"AQI {pred['predicted_aqi']:>3d} — {pred['category']}")

    if result["alert"]:
        print(f"\n  {result['alert_message']}")

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nPredictions saved to {args.output}")

    return result


if __name__ == "__main__":
    main()
