"""
Model evaluation utilities.
Computes RMSE, MAE, R² and generates comparison visualizations.
"""
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Dictionary with RMSE, MAE, R² metrics
    """
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_model(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate a model and log results.

    Args:
        model_name: Name of the model
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Dictionary with metrics
    """
    metrics = compute_metrics(y_true, y_pred)
    logger.info(
        f"{model_name:20s} | RMSE: {metrics['rmse']:8.2f} | "
        f"MAE: {metrics['mae']:8.2f} | R²: {metrics['r2']:6.3f}"
    )
    return metrics


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create a comparison table of all models.

    Args:
        results: {model_name: {metric_name: value}}

    Returns:
        DataFrame with model comparison
    """
    df = pd.DataFrame(results).T
    df.index.name = "Model"
    df = df.sort_values("rmse")

    logger.info("\n" + "=" * 60)
    logger.info("Model Comparison (sorted by RMSE)")
    logger.info("=" * 60)
    logger.info(f"\n{df.to_string()}")

    return df


def select_best_model(results: Dict[str, Dict[str, float]]) -> str:
    """
    Select the best model based on RMSE.

    Args:
        results: {model_name: {metric_name: value}}

    Returns:
        Name of the best model
    """
    best = min(results, key=lambda k: results[k]["rmse"])
    logger.info(f"\nBest model: {best} (RMSE: {results[best]['rmse']:.2f})")
    return best


def plot_comparison(results: Dict[str, Dict[str, float]], save_path: str = None):
    """
    Create a bar chart comparing model metrics.

    Args:
        results: {model_name: {metric_name: value}}
        save_path: Optional path to save the plot
    """
    df = pd.DataFrame(results).T

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # RMSE
    colors = ["#2ecc71" if v == df["rmse"].min() else "#3498db" for v in df["rmse"]]
    axes[0].barh(df.index, df["rmse"], color=colors)
    axes[0].set_xlabel("RMSE")
    axes[0].set_title("RMSE (lower is better)")

    # MAE
    colors = ["#2ecc71" if v == df["mae"].min() else "#e74c3c" for v in df["mae"]]
    axes[1].barh(df.index, df["mae"], color=colors)
    axes[1].set_xlabel("MAE")
    axes[1].set_title("MAE (lower is better)")

    # R²
    colors = ["#2ecc71" if v == df["r2"].max() else "#f39c12" for v in df["r2"]]
    axes[2].barh(df.index, df["r2"], color=colors)
    axes[2].set_xlabel("R²")
    axes[2].set_title("R² (higher is better)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Comparison plot saved to {save_path}")
    plt.close()
    return fig


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: str = None,
):
    """
    Create prediction vs actual scatter plot.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of the model
        save_path: Optional path to save
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=10, color="steelblue")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    axes[0].set_xlabel("Actual AQI")
    axes[0].set_ylabel("Predicted AQI")
    axes[0].set_title(f"{model_name}: Predicted vs Actual")

    # Residuals
    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Residual (Actual - Predicted)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"{model_name}: Residual Distribution")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig
