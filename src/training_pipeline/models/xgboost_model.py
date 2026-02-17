"""
XGBoost Regressor for AQI prediction.
Gradient boosting model with early stopping support.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "xgboost"


def build_model(
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
):
    """
    Build an XGBoost Regressor.

    Args:
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Step size shrinkage
        random_state: Random seed

    Returns:
        XGBRegressor instance
    """
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    tune: bool = True,
    random_state: int = 42,
):
    """
    Train the XGBoost model with optional early stopping and tuning.

    Args:
        X_train: Training feature matrix
        y_train: Training target vector
        X_val: Validation features (for early stopping)
        y_val: Validation target
        tune: Whether to perform grid search

    Returns:
        Trained XGBRegressor
    """
    from xgboost import XGBRegressor
    from sklearn.model_selection import GridSearchCV

    if tune and X_val is None:
        # Grid search without early stopping
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
        model = XGBRegressor(random_state=random_state, n_jobs=-1, verbosity=0)
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)
        logger.info(f"Best XGB params: {grid_search.best_params_}")
        logger.info(f"Best CV RMSE: {np.sqrt(-grid_search.best_score_):.2f}")
        return grid_search.best_estimator_
    else:
        model = build_model(random_state=random_state)
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            model.set_params(early_stopping_rounds=20)

        model.fit(X_train, y_train, **fit_params)
        return model


def predict(model, X: np.ndarray) -> np.ndarray:
    """Generate predictions using the trained XGBoost model."""
    return model.predict(X)


def get_feature_importance(model, feature_names: list) -> dict:
    """Get feature importance from the trained model."""
    importances = model.feature_importances_
    return dict(sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    ))
