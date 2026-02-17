"""
SHAP-based model explainability for AQI predictions.
Generates feature importance plots and per-prediction explanations.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import MODELS_DIR, ALL_FEATURES

logger = logging.getLogger(__name__)

PLOTS_DIR = MODELS_DIR / "shap_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def get_shap_explainer(model, model_type: str, X_background: np.ndarray = None):
    """
    Create the appropriate SHAP explainer for a given model type.

    Args:
        model: Trained model
        model_type: Type of model ('random_forest', 'xgboost', 'ridge', 'lstm')
        X_background: Background data for KernelExplainer

    Returns:
        SHAP Explainer object
    """
    import shap

    if model_type == "xgboost":
        # Use TreeExplainer with the booster directly for XGBoost 3.x compatibility
        try:
            booster = model.get_booster()
            explainer = shap.TreeExplainer(booster)
        except Exception:
            # Fallback: use model.predict with output_margin for KernelExplainer
            if X_background is not None:
                background = shap.sample(X_background, min(100, len(X_background)))
            else:
                background = X_background
            explainer = shap.KernelExplainer(model.predict, background)
        logger.info(f"Created SHAP explainer for {model_type}")
    elif model_type == "random_forest":
        explainer = shap.TreeExplainer(model)
        logger.info(f"Created TreeExplainer for {model_type}")
    elif model_type == "ridge":
        # For sklearn pipelines, use the pipeline's predict
        if X_background is not None:
            background = shap.sample(X_background, min(100, len(X_background)))
        else:
            background = X_background
        explainer = shap.KernelExplainer(model.predict, background)
        logger.info("Created KernelExplainer for Ridge")
    else:
        # KernelExplainer for any model (slow but universal)
        if X_background is not None:
            background = shap.sample(X_background, min(50, len(X_background)))
        else:
            background = X_background
        explainer = shap.KernelExplainer(
            lambda x: model.predict(x.reshape(-1, x.shape[-1])) if len(x.shape) > 2 else model.predict(x),
            background,
        )
        logger.info(f"Created KernelExplainer for {model_type}")

    return explainer


def compute_shap_values(
    model,
    model_type: str,
    X: np.ndarray,
    feature_names: list = None,
) -> dict:
    """
    Compute SHAP values for the given dataset.

    Args:
        model: Trained model
        model_type: Model type string
        X: Feature matrix
        feature_names: List of feature names

    Returns:
        Dictionary with SHAP values and feature importance
    """
    import shap

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    explainer = get_shap_explainer(model, model_type, X)

    # Compute SHAP values
    if model_type in ("random_forest", "xgboost"):
        shap_values = explainer.shap_values(X)
    else:
        # For KernelExplainer, use a subset for speed
        subset_size = min(100, len(X))
        shap_values = explainer.shap_values(X[:subset_size])

    # Compute mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = dict(sorted(
        zip(feature_names, mean_abs_shap),
        key=lambda x: x[1],
        reverse=True,
    ))

    return {
        "shap_values": shap_values,
        "feature_importance": feature_importance,
        "explainer": explainer,
        "feature_names": feature_names,
    }


def plot_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list,
    save_path: Optional[str] = None,
    max_display: int = 15,
):
    """
    Generate SHAP summary plot (beeswarm).

    Args:
        shap_values: SHAP values array
        X: Feature matrix
        feature_names: Feature names
        save_path: Path to save the plot
        max_display: Maximum features to display
    """
    import shap

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X if len(X) == len(shap_values) else X[:len(shap_values)],
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.title("SHAP Feature Importance — AQI Predictor")
    plt.tight_layout()

    if save_path is None:
        save_path = str(PLOTS_DIR / "shap_summary.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP summary plot saved to {save_path}")


def plot_feature_importance_bar(
    feature_importance: dict,
    save_path: Optional[str] = None,
    top_n: int = 10,
):
    """
    Generate a bar chart of top SHAP feature importances.

    Args:
        feature_importance: {feature_name: importance_value}
        save_path: Path to save the plot
        top_n: Number of top features to show
    """
    top_features = dict(list(feature_importance.items())[:top_n])

    fig, ax = plt.subplots(figsize=(10, 6))
    features = list(top_features.keys())
    values = list(top_features.values())

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(features)))
    ax.barh(range(len(features)), values, color=colors)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(f"Top {top_n} Feature Importance (SHAP)")
    ax.invert_yaxis()

    plt.tight_layout()

    if save_path is None:
        save_path = str(PLOTS_DIR / "shap_bar.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Feature importance bar plot saved to {save_path}")
    return fig


def plot_force(
    explainer,
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list,
    sample_idx: int = 0,
    save_path: Optional[str] = None,
):
    """
    Generate SHAP force plot for a single prediction.

    Args:
        explainer: SHAP explainer
        shap_values: SHAP values
        X: Feature matrix
        feature_names: Feature names
        sample_idx: Index of sample to explain
        save_path: Path to save
    """
    import shap

    fig = shap.force_plot(
        explainer.expected_value,
        shap_values[sample_idx],
        X[sample_idx],
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )

    if save_path is None:
        save_path = str(PLOTS_DIR / f"shap_force_{sample_idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Force plot saved to {save_path}")


def explain_model(
    model,
    model_type: str,
    X: np.ndarray,
    feature_names: list = None,
) -> dict:
    """
    Full SHAP explanation pipeline: compute values + generate all plots.

    Args:
        model: Trained model
        model_type: Model type string
        X: Feature matrix for explanation
        feature_names: Feature names

    Returns:
        Dictionary with SHAP results and plot paths
    """
    logger.info(f"Generating SHAP explanations for {model_type}...")

    results = compute_shap_values(model, model_type, X, feature_names)

    # Generate plots
    plot_summary(
        results["shap_values"], X, results["feature_names"],
        save_path=str(PLOTS_DIR / "shap_summary.png"),
    )
    plot_feature_importance_bar(
        results["feature_importance"],
        save_path=str(PLOTS_DIR / "shap_bar.png"),
    )

    # Force plot for first sample
    try:
        plot_force(
            results["explainer"],
            results["shap_values"],
            X if len(X) == len(results["shap_values"]) else X[:len(results["shap_values"])],
            results["feature_names"],
            sample_idx=0,
            save_path=str(PLOTS_DIR / "shap_force_latest.png"),
        )
    except Exception as e:
        logger.warning(f"Could not generate force plot: {e}")

    results["plot_paths"] = {
        "summary": str(PLOTS_DIR / "shap_summary.png"),
        "bar": str(PLOTS_DIR / "shap_bar.png"),
        "force": str(PLOTS_DIR / "shap_force_latest.png"),
    }

    logger.info("SHAP explanation complete!")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("SHAP explainability module ready.")
    print(f"Plots will be saved to: {PLOTS_DIR}")
