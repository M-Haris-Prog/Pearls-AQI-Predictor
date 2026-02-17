"""
LSTM (Long Short-Term Memory) model for AQI prediction using TensorFlow/Keras.
Captures temporal patterns in the time-series data.
Outputs 3 values: AQI predictions for Day+1, Day+2, Day+3.
"""
import logging
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)

MODEL_NAME = "lstm"


def build_model(
    input_shape: Tuple[int, int],
    output_steps: int = 3,
    lstm_units: int = 64,
    dropout: float = 0.2,
) -> "tf.keras.Model":
    """
    Build an LSTM model for multi-step AQI forecasting.

    Args:
        input_shape: (sequence_length, n_features)
        output_steps: Number of future steps to predict (default: 3 days)
        lstm_units: Number of LSTM units per layer
        dropout: Dropout rate

    Returns:
        Compiled Keras model
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        BatchNormalization(),
        LSTM(lstm_units // 2, return_sequences=False),
        Dropout(dropout),
        BatchNormalization(),
        Dense(32, activation="relu"),
        Dense(output_steps),  # Predict 3 days
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )

    logger.info(f"LSTM model built: input_shape={input_shape}, output_steps={output_steps}")
    return model


def create_sequences(
    data: np.ndarray,
    target: np.ndarray,
    sequence_length: int = 24,
    forecast_horizon: int = 3,
    step_size: int = 24,  # daily aggregation step
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input sequences and targets for LSTM training.

    For 3-day forecasting, we take 'sequence_length' hours of history
    and predict the average AQI for the next 3 days (72 hours, aggregated daily).

    Args:
        data: Feature matrix (n_samples, n_features)
        target: Target vector (n_samples,)
        sequence_length: Number of time steps for input
        forecast_horizon: Number of days to forecast
        step_size: Hours per day for aggregation

    Returns:
        Tuple of (X_sequences, y_targets)
    """
    X, y = [], []
    total_future = forecast_horizon * step_size  # 3 days × 24 hours = 72

    for i in range(len(data) - sequence_length - total_future + 1):
        # Input: 'sequence_length' hours of features
        X.append(data[i:i + sequence_length])

        # Target: average AQI for each of the next 3 days
        future_targets = []
        for day in range(forecast_horizon):
            start = i + sequence_length + day * step_size
            end = min(start + step_size, len(target))
            if end > start:
                future_targets.append(np.mean(target[start:end]))
            else:
                future_targets.append(target[start - 1] if start > 0 else 0)
        y.append(future_targets)

    return np.array(X), np.array(y)


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 10,
):
    """
    Train the LSTM model.

    Args:
        X_train: Training sequences (n_samples, sequence_length, n_features)
        y_train: Training targets (n_samples, forecast_horizon)
        X_val: Validation sequences
        y_val: Validation targets
        epochs: Training epochs
        batch_size: Batch size
        patience: Early stopping patience

    Returns:
        Tuple of (trained model, training history)
    """
    import tensorflow as tf

    input_shape = (X_train.shape[1], X_train.shape[2])
    output_steps = y_train.shape[1]

    model = build_model(input_shape, output_steps)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss" if X_val is not None else "loss",
            patience=patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if X_val is not None else "loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        ),
    ]

    validation_data = (X_val, y_val) if X_val is not None else None

    history = model.fit(
        X_train, y_train,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    logger.info(f"LSTM training complete. Final loss: {history.history['loss'][-1]:.4f}")
    return model, history


def predict(model, X: np.ndarray) -> np.ndarray:
    """
    Generate predictions using the trained LSTM model.

    Args:
        model: Trained Keras model
        X: Input sequences (n_samples, sequence_length, n_features)

    Returns:
        Predictions (n_samples, forecast_horizon)
    """
    return model.predict(X, verbose=0)


def save_model(model, path: str):
    """Save the LSTM model to disk."""
    model.save(path)
    logger.info(f"LSTM model saved to {path}")


def load_model(path: str):
    """Load an LSTM model from disk."""
    import tensorflow as tf
    model = tf.keras.models.load_model(path)
    logger.info(f"LSTM model loaded from {path}")
    return model
