"""
Ridge Regression model for AQI prediction.
Good baseline model that handles multicollinearity well.
"""
import logging
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

MODEL_NAME = "ridge_regression"


def build_model(alpha: float = 1.0) -> Pipeline:
    """
    Build a Ridge Regression pipeline with standard scaling.

    Args:
        alpha: Regularization strength

    Returns:
        sklearn Pipeline with scaler + Ridge
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])
    return pipeline


def train(X: np.ndarray, y: np.ndarray, tune: bool = True) -> Pipeline:
    """
    Train the Ridge Regression model with optional hyperparameter tuning.

    Args:
        X: Feature matrix
        y: Target vector
        tune: Whether to perform grid search

    Returns:
        Trained pipeline
    """
    if tune:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge()),
        ])
        param_grid = {
            "ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        }
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X, y)
        logger.info(f"Best Ridge alpha: {grid_search.best_params_['ridge__alpha']}")
        logger.info(f"Best CV RMSE: {np.sqrt(-grid_search.best_score_):.2f}")
        return grid_search.best_estimator_
    else:
        model = build_model()
        model.fit(X, y)
        return model


def predict(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Generate predictions using the trained Ridge model."""
    return model.predict(X)
