"""
Random Forest Regressor for AQI prediction.
Ensemble model that captures non-linear relationships.
"""
import logging
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)

MODEL_NAME = "random_forest"


def build_model(n_estimators: int = 100, max_depth: int = None, random_state: int = 42):
    """
    Build a Random Forest Regressor.

    Args:
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        random_state: Random seed

    Returns:
        RandomForestRegressor instance
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )


def train(X: np.ndarray, y: np.ndarray, tune: bool = True, random_state: int = 42):
    """
    Train the Random Forest model with optional hyperparameter tuning.

    Args:
        X: Feature matrix
        y: Target vector
        tune: Whether to perform grid search

    Returns:
        Trained RandomForestRegressor
    """
    if tune:
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
        model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X, y)
        best = grid_search.best_params_
        logger.info(f"Best RF params: {best}")
        logger.info(f"Best CV RMSE: {np.sqrt(-grid_search.best_score_):.2f}")
        return grid_search.best_estimator_
    else:
        model = build_model(random_state=random_state)
        model.fit(X, y)
        return model


def predict(model, X: np.ndarray) -> np.ndarray:
    """Generate predictions using the trained Random Forest model."""
    return model.predict(X)


def get_feature_importance(model, feature_names: list) -> dict:
    """Get feature importance from the trained model."""
    importances = model.feature_importances_
    return dict(sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    ))
