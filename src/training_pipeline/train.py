"""
Training pipeline orchestrator.
Fetches features, trains all models, evaluates, and registers the best model.
"""
import logging
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from src.config import (
    MODELS_DIR, PROCESSED_DATA_DIR, ALL_FEATURES, TARGET,
    RANDOM_STATE, TEST_SIZE, SEQUENCE_LENGTH, FORECAST_DAYS,
)
from src.feature_pipeline.feature_store import get_features, get_training_data
from src.training_pipeline.evaluate import (
    evaluate_model, compare_models, select_best_model,
    plot_comparison, plot_predictions_vs_actual,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def time_based_split(X, y, test_size: float = TEST_SIZE):
    """
    Time-based train/test split (no shuffling — preserves temporal order).

    Args:
        X: Feature matrix
        y: Target vector
        test_size: Fraction of data for testing

    Returns:
        X_train, X_test, y_train, y_test
    """
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    logger.info(f"Train/Test split: {len(X_train)}/{len(X_test)} samples (test_size={test_size})")
    return X_train, X_test, y_train, y_test


def train_ridge(X_train, y_train, X_test, y_test):
    """Train and evaluate Ridge Regression."""
    from src.training_pipeline.models.ridge_model import train, predict
    logger.info("Training Ridge Regression...")
    model = train(X_train.values, y_train.values, tune=True)
    y_pred = predict(model, X_test.values)
    metrics = evaluate_model("Ridge Regression", y_test.values, y_pred)
    return model, metrics, y_pred


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate Random Forest."""
    from src.training_pipeline.models.random_forest import train, predict
    logger.info("Training Random Forest...")
    model = train(X_train.values, y_train.values, tune=True, random_state=RANDOM_STATE)
    y_pred = predict(model, X_test.values)
    metrics = evaluate_model("Random Forest", y_test.values, y_pred)
    return model, metrics, y_pred


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train and evaluate XGBoost."""
    from src.training_pipeline.models.xgboost_model import train, predict
    logger.info("Training XGBoost...")
    model = train(
        X_train.values, y_train.values,
        X_val=X_test.values, y_val=y_test.values,
        tune=False,  # Use early stopping instead of grid search for speed
        random_state=RANDOM_STATE,
    )
    y_pred = predict(model, X_test.values)
    metrics = evaluate_model("XGBoost", y_test.values, y_pred)
    return model, metrics, y_pred


def train_lstm(X_train, y_train, X_test, y_test, all_features_df):
    """Train and evaluate LSTM model."""
    from src.training_pipeline.models.lstm_model import (
        create_sequences, train, predict,
    )
    logger.info("Training LSTM...")

    # Get all data for sequence creation
    feature_cols = [c for c in ALL_FEATURES if c in all_features_df.columns]
    data = all_features_df[feature_cols].values
    target = all_features_df[TARGET].values

    # Normalize data for LSTM
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    data_scaled = scaler_X.fit_transform(data)

    # Create sequences
    X_seq, y_seq = create_sequences(
        data_scaled, target,
        sequence_length=min(SEQUENCE_LENGTH, len(data) // 4),
        forecast_horizon=FORECAST_DAYS,
        step_size=max(1, len(data) // (len(data) // 24) if len(data) >= 24 else 1),
    )

    if len(X_seq) < 10:
        logger.warning("Not enough data for LSTM sequences. Skipping LSTM.")
        return None, {"rmse": float("inf"), "mae": float("inf"), "r2": -float("inf")}, None

    # Split sequences
    split_idx = int(len(X_seq) * (1 - TEST_SIZE))
    X_train_seq, X_test_seq = X_seq[:split_idx], X_seq[split_idx:]
    y_train_seq, y_test_seq = y_seq[:split_idx], y_seq[split_idx:]

    # Train
    model, history = train(
        X_train_seq, y_train_seq,
        X_val=X_test_seq, y_val=y_test_seq,
        epochs=50,
        batch_size=min(32, len(X_train_seq)),
    )

    # Evaluate (use mean of 3-day predictions vs mean of actual)
    y_pred_seq = predict(model, X_test_seq)
    y_pred_mean = y_pred_seq.mean(axis=1)
    y_test_mean = y_test_seq.mean(axis=1)
    metrics = evaluate_model("LSTM", y_test_mean, y_pred_mean)

    # Save scaler for inference
    joblib.dump(scaler_X, MODELS_DIR / "lstm_scaler.pkl")

    return model, metrics, y_pred_mean


def save_model_artifact(model, model_name: str, metrics: dict):
    """Save model artifact and metadata to disk."""
    model_dir = MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    if model_name == "lstm":
        from src.training_pipeline.models.lstm_model import save_model
        save_model(model, str(model_dir / "model.h5"))
    else:
        joblib.dump(model, model_dir / "model.pkl")

    # Save metadata
    metadata = {
        "model_name": model_name,
        "metrics": metrics,
        "trained_at": datetime.utcnow().isoformat(),
        "features": ALL_FEATURES,
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model '{model_name}' saved to {model_dir}")


def register_to_hopsworks(model, model_name: str, metrics: dict):
    """Register the best model in Hopsworks Model Registry."""
    from src.config import HOPSWORKS_API_KEY, MODEL_REGISTRY_NAME

    if not HOPSWORKS_API_KEY:
        logger.info("Hopsworks not configured — skipping model registration")
        return

    try:
        import hopsworks
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        mr = project.get_model_registry()

        # Save model locally first
        model_dir = MODELS_DIR / model_name
        model_path = str(model_dir)

        hw_model = mr.python.create_model(
            name=MODEL_REGISTRY_NAME,
            description=f"AQI Predictor ({model_name}) — RMSE: {metrics['rmse']:.2f}",
            metrics=metrics,
        )
        hw_model.save(model_path)
        logger.info(f"Model registered in Hopsworks: {MODEL_REGISTRY_NAME}")
    except Exception as e:
        logger.error(f"Failed to register model in Hopsworks: {e}")


def run_training_pipeline(use_hopsworks: bool = True):
    """
    Main training pipeline.
    Fetches data, trains all models, evaluates, and saves the best one.
    """
    logger.info("=" * 60)
    logger.info("Starting Training Pipeline")
    logger.info("=" * 60)

    # Step 1: Get training data
    logger.info("\nStep 1: Fetching training data...")
    try:
        X, y = get_training_data(use_hopsworks=use_hopsworks)
    except ValueError as e:
        logger.error(f"Cannot train: {e}")
        logger.info("Run the backfill pipeline first: python -m src.feature_pipeline.backfill --days 30")
        return

    # Get full features DataFrame for LSTM
    all_features_df = get_features(use_hopsworks=use_hopsworks)

    # Step 2: Time-based split
    logger.info("\nStep 2: Splitting data...")
    X_train, X_test, y_train, y_test = time_based_split(X, y)

    # Step 3: Train all models
    logger.info("\nStep 3: Training models...")
    results = {}
    models = {}

    # 3a: Ridge Regression
    try:
        model, metrics, y_pred = train_ridge(X_train, y_train, X_test, y_test)
        results["Ridge Regression"] = metrics
        models["ridge"] = model
        plot_predictions_vs_actual(
            y_test.values, y_pred, "Ridge Regression",
            str(MODELS_DIR / "ridge_predictions.png"),
        )
    except Exception as e:
        logger.error(f"Ridge training failed: {e}")

    # 3b: Random Forest
    try:
        model, metrics, y_pred = train_random_forest(X_train, y_train, X_test, y_test)
        results["Random Forest"] = metrics
        models["random_forest"] = model
        plot_predictions_vs_actual(
            y_test.values, y_pred, "Random Forest",
            str(MODELS_DIR / "rf_predictions.png"),
        )
    except Exception as e:
        logger.error(f"Random Forest training failed: {e}")

    # 3c: XGBoost
    try:
        model, metrics, y_pred = train_xgboost(X_train, y_train, X_test, y_test)
        results["XGBoost"] = metrics
        models["xgboost"] = model
        plot_predictions_vs_actual(
            y_test.values, y_pred, "XGBoost",
            str(MODELS_DIR / "xgb_predictions.png"),
        )
    except Exception as e:
        logger.error(f"XGBoost training failed: {e}")

    # 3d: LSTM
    try:
        model, metrics, y_pred = train_lstm(X_train, y_train, X_test, y_test, all_features_df)
        if model is not None:
            results["LSTM"] = metrics
            models["lstm"] = model
    except Exception as e:
        logger.error(f"LSTM training failed: {e}")

    if not results:
        logger.error("No models trained successfully!")
        return

    # Step 4: Compare and select best
    logger.info("\nStep 4: Comparing models...")
    comparison_df = compare_models(results)
    comparison_df.to_csv(MODELS_DIR / "model_comparison.csv")
    plot_comparison(results, str(MODELS_DIR / "model_comparison.png"))

    best_name = select_best_model(results)

    # Map display name to model key
    name_map = {
        "Ridge Regression": "ridge",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
        "LSTM": "lstm",
    }
    best_key = name_map.get(best_name, best_name.lower().replace(" ", "_"))

    # Step 5: Save best model
    logger.info("\nStep 5: Saving best model...")
    if best_key in models:
        save_model_artifact(models[best_key], best_key, results[best_name])

        # Also save as "best" for easy loading
        save_model_artifact(models[best_key], "best", results[best_name])

        if use_hopsworks:
            register_to_hopsworks(models[best_key], best_key, results[best_name])

    # Save all models
    for display_name, key in name_map.items():
        if key in models and key != best_key:
            save_model_artifact(models[key], key, results.get(display_name, {}))

    logger.info("\n" + "=" * 60)
    logger.info("Training Pipeline Complete!")
    logger.info(f"Best Model: {best_name}")
    logger.info(f"RMSE: {results[best_name]['rmse']:.2f}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run the AQI Training Pipeline")
    parser.add_argument("--no-hopsworks", action="store_true", help="Use local storage only")
    args = parser.parse_args()
    run_training_pipeline(use_hopsworks=not args.no_hopsworks)


if __name__ == "__main__":
    main()
