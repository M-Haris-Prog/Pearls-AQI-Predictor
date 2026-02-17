"""
LIME-based model explainability for AQI predictions.
Provides local interpretable explanations for individual predictions.
"""
import logging
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from src.config import MODELS_DIR

logger = logging.getLogger(__name__)

PLOTS_DIR = MODELS_DIR / "lime_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def create_explainer(
    X_train: np.ndarray,
    feature_names: list,
    mode: str = "regression",
):
    """
    Create a LIME tabular explainer.

    Args:
        X_train: Training data for determining feature statistics
        feature_names: List of feature names
        mode: 'regression' or 'classification'

    Returns:
        LimeTabularExplainer instance
    """
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        mode=mode,
        verbose=False,
    )
    logger.info("LIME TabularExplainer created")
    return explainer


def explain_prediction(
    explainer,
    model,
    instance: np.ndarray,
    num_features: int = 10,
    save_path: Optional[str] = None,
) -> dict:
    """
    Generate LIME explanation for a single prediction.

    Args:
        explainer: LIME explainer
        model: Trained model (must have predict method)
        instance: Single instance to explain (1D array)
        num_features: Number of features to show
        save_path: Path to save explanation plot

    Returns:
        Dictionary with explanation details
    """
    # Get prediction function
    if hasattr(model, "predict"):
        predict_fn = model.predict
    else:
        def predict_fn(x):
            return model.predict(x)

    explanation = explainer.explain_instance(
        data_row=instance,
        predict_fn=predict_fn,
        num_features=num_features,
    )

    # Extract feature contributions
    feature_contributions = {}
    for feature, weight in explanation.as_list():
        feature_contributions[feature] = weight

    # Generate and save plot
    if save_path is None:
        save_path = str(PLOTS_DIR / "lime_explanation.png")

    fig = explanation.as_pyplot_figure()
    fig.set_size_inches(10, 6)
    plt.title("LIME Feature Contributions — AQI Prediction")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"LIME explanation plot saved to {save_path}")

    return {
        "prediction": float(explanation.predicted_value) if hasattr(explanation, "predicted_value") else None,
        "feature_contributions": feature_contributions,
        "intercept": float(explanation.intercept[0]) if hasattr(explanation, "intercept") else None,
        "plot_path": save_path,
    }


def explain_multiple(
    model,
    X_train: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list,
    num_samples: int = 5,
    num_features: int = 10,
) -> list:
    """
    Generate LIME explanations for multiple predictions.

    Args:
        model: Trained model
        X_train: Training data
        X_explain: Instances to explain
        feature_names: Feature names
        num_samples: Number of samples to explain
        num_features: Features per explanation

    Returns:
        List of explanation dictionaries
    """
    explainer = create_explainer(X_train, feature_names)
    explanations = []

    for i in range(min(num_samples, len(X_explain))):
        try:
            exp = explain_prediction(
                explainer, model, X_explain[i],
                num_features=num_features,
                save_path=str(PLOTS_DIR / f"lime_explanation_{i}.png"),
            )
            explanations.append(exp)
        except Exception as e:
            logger.warning(f"LIME explanation failed for sample {i}: {e}")

    logger.info(f"Generated {len(explanations)} LIME explanations")
    return explanations


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("LIME explainability module ready.")
    print(f"Plots will be saved to: {PLOTS_DIR}")
