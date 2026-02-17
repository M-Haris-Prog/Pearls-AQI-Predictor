"""
Tests for the Training Pipeline.
Uses synthetic data to verify model training and evaluation.
"""
import pytest
import numpy as np
import pandas as pd


class TestModels:
    """Test individual model modules."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic training data."""
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 10)
        # Simple linear relationship + noise
        y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(n) * 0.5 + 100
        return X, y

    def test_ridge_train_predict(self, synthetic_data):
        """Test Ridge model training and prediction."""
        from src.training_pipeline.models.ridge_model import train, predict

        X, y = synthetic_data
        model = train(X, y, tune=False)
        preds = predict(model, X)

        assert len(preds) == len(y)
        assert preds.dtype in [np.float64, np.float32]

    def test_ridge_tuning(self, synthetic_data):
        """Test Ridge model with hyperparameter tuning."""
        from src.training_pipeline.models.ridge_model import train

        X, y = synthetic_data
        model = train(X, y, tune=True)
        assert model is not None

    def test_random_forest_train_predict(self, synthetic_data):
        """Test Random Forest training and prediction."""
        from src.training_pipeline.models.random_forest import train, predict

        X, y = synthetic_data
        model = train(X, y, tune=False)
        preds = predict(model, X)

        assert len(preds) == len(y)
        # RF should fit well on training data
        from sklearn.metrics import r2_score
        r2 = r2_score(y, preds)
        assert r2 > 0.5, f"R² too low: {r2}"

    def test_random_forest_feature_importance(self, synthetic_data):
        """Test feature importance extraction."""
        from src.training_pipeline.models.random_forest import train, get_feature_importance

        X, y = synthetic_data
        model = train(X, y, tune=False)
        features = [f"feature_{i}" for i in range(X.shape[1])]
        importance = get_feature_importance(model, features)

        assert len(importance) == X.shape[1]
        assert all(v >= 0 for v in importance.values())

    def test_xgboost_train_predict(self, synthetic_data):
        """Test XGBoost training and prediction."""
        from src.training_pipeline.models.xgboost_model import train, predict

        X, y = synthetic_data
        model = train(X, y, tune=False)
        preds = predict(model, X)

        assert len(preds) == len(y)

    def test_xgboost_early_stopping(self, synthetic_data):
        """Test XGBoost with early stopping."""
        from src.training_pipeline.models.xgboost_model import train

        X, y = synthetic_data
        split = int(len(X) * 0.8)
        model = train(
            X[:split], y[:split],
            X_val=X[split:], y_val=y[split:],
            tune=False,
        )
        assert model is not None


class TestEvaluation:
    """Test model evaluation utilities."""

    def test_compute_metrics(self):
        """Test metric computation."""
        from src.training_pipeline.evaluate import compute_metrics

        y_true = np.array([100, 150, 200, 250, 300])
        y_pred = np.array([105, 145, 210, 245, 290])

        metrics = compute_metrics(y_true, y_pred)

        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0
        assert metrics["r2"] <= 1.0

    def test_compare_models(self):
        """Test model comparison table generation."""
        from src.training_pipeline.evaluate import compare_models

        results = {
            "Model A": {"rmse": 15.0, "mae": 10.0, "r2": 0.85},
            "Model B": {"rmse": 12.0, "mae": 8.0, "r2": 0.90},
        }

        df = compare_models(results)

        assert len(df) == 2
        # Should be sorted by RMSE (ascending)
        assert df.index[0] == "Model B"

    def test_select_best_model(self):
        """Test best model selection."""
        from src.training_pipeline.evaluate import select_best_model

        results = {
            "Model A": {"rmse": 15.0, "mae": 10.0, "r2": 0.85},
            "Model B": {"rmse": 12.0, "mae": 8.0, "r2": 0.90},
            "Model C": {"rmse": 20.0, "mae": 15.0, "r2": 0.75},
        }

        best = select_best_model(results)
        assert best == "Model B"


class TestLSTMModel:
    """Test LSTM model components."""

    def test_create_sequences(self):
        """Test LSTM sequence creation."""
        from src.training_pipeline.models.lstm_model import create_sequences

        n_samples = 100
        n_features = 5
        data = np.random.randn(n_samples, n_features)
        target = np.random.uniform(50, 200, n_samples)

        seq_length = 10
        forecast_horizon = 3
        step_size = 1  # For testing with small data

        X, y = create_sequences(data, target, seq_length, forecast_horizon, step_size)

        assert len(X.shape) == 3  # (samples, seq_length, features)
        assert X.shape[1] == seq_length
        assert X.shape[2] == n_features
        assert y.shape[1] == forecast_horizon
        assert len(X) == len(y)
