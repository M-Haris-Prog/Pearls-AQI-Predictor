"""
Flask REST API for AQI predictions.
Provides endpoints for predictions, features, and explanations.
"""
import json
import logging
from datetime import datetime

from flask import Flask, jsonify, request
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "aqi-predictor-api",
    })


@app.route("/predict", methods=["GET"])
def predict():
    """
    Get 3-day AQI predictions.
    Query params:
        - model: model name (optional, default: best)
    """
    model_name = request.args.get("model", None)

    try:
        from src.inference.predict import predict_next_3_days
        result = predict_next_3_days(model_name=model_name)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e), "hint": "Run training pipeline first"}), 404
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/features", methods=["GET"])
def features():
    """
    Get latest feature values from the Feature Store.
    Query params:
        - limit: number of recent records (default: 24)
    """
    limit = request.args.get("limit", 24, type=int)

    try:
        from src.feature_pipeline.feature_store import get_features
        df = get_features(use_hopsworks=False)

        if df.empty:
            return jsonify({"error": "No features available"}), 404

        # Return most recent records
        recent = df.tail(limit)
        records = recent.to_dict(orient="records")

        # Convert datetime to string for JSON serialization
        for record in records:
            if "datetime" in record:
                record["datetime"] = str(record["datetime"])

        return jsonify({
            "count": len(records),
            "features": records,
            "latest_datetime": str(df["datetime"].max()),
        })
    except Exception as e:
        logger.error(f"Feature retrieval failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/explain", methods=["GET"])
def explain():
    """
    Get SHAP feature importance for the best model.
    """
    try:
        from src.config import MODELS_DIR  # noqa: F401

        # Try to load pre-computed feature importance
        from src.inference.predict import load_best_model
        from src.feature_pipeline.feature_store import get_training_data
        from src.explainability.shap_explain import compute_shap_values
        from src.config import ALL_FEATURES

        model, metadata, model_type = load_best_model()
        X, y = get_training_data(use_hopsworks=False)

        feature_cols = [c for c in ALL_FEATURES if c in X.columns]
        results = compute_shap_values(model, model_type, X[feature_cols].values[:100], feature_cols)

        return jsonify({
            "model": model_type,
            "feature_importance": {k: float(v) for k, v in results["feature_importance"].items()},
            "top_features": dict(list(results["feature_importance"].items())[:10]),
        })
    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/models", methods=["GET"])
def list_models():
    """List all available trained models."""
    from src.config import MODELS_DIR

    models_info = []
    for model_dir in MODELS_DIR.iterdir():
        if model_dir.is_dir():
            metadata_path = model_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                models_info.append({
                    "name": model_dir.name,
                    "metrics": metadata.get("metrics", {}),
                    "trained_at": metadata.get("trained_at", ""),
                })

    return jsonify({"models": models_info})


@app.route("/alerts", methods=["GET"])
def alerts():
    """Get current alert status based on latest predictions."""
    try:
        from src.inference.predict import predict_next_3_days
        from app.alerts import check_alerts

        result = predict_next_3_days()
        alert_list = check_alerts(result["predictions"])

        return jsonify({
            "has_alerts": len(alert_list) > 0,
            "alert_count": len(alert_list),
            "alerts": alert_list,
        })
    except Exception as e:
        logger.error(f"Alert check failed: {e}")
        return jsonify({"error": str(e)}), 500


def main():
    """Run the Flask API server."""
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()
