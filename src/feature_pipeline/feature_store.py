"""
Feature Store interface using Hopsworks.
Handles reading/writing features and managing Feature Groups & Views.
Falls back to local Parquet storage if Hopsworks is unavailable.
"""
import logging
from typing import Optional, Tuple

import pandas as pd

from src.config import (
    HOPSWORKS_API_KEY, HOPSWORKS_PROJECT_NAME,
    FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION,
    FEATURE_VIEW_NAME, FEATURE_VIEW_VERSION,
    PROCESSED_DATA_DIR, TARGET, ALL_FEATURES,
)

logger = logging.getLogger(__name__)

# Local fallback path
LOCAL_FEATURE_FILE = PROCESSED_DATA_DIR / "features.parquet"


def _get_hopsworks_connection():
    """Connect to Hopsworks and return the project."""
    try:
        import hopsworks
        project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT_NAME,
        )
        return project
    except Exception as e:
        logger.warning(f"Could not connect to Hopsworks: {e}")
        return None


def _get_feature_store(project):
    """Get the feature store from a Hopsworks project."""
    return project.get_feature_store()


def insert_features(df: pd.DataFrame, use_hopsworks: bool = True) -> bool:
    """
    Insert features into the Feature Store.
    Falls back to local Parquet if Hopsworks is unavailable.

    Args:
        df: DataFrame with engineered features
        use_hopsworks: Whether to attempt Hopsworks storage

    Returns:
        True if insertion succeeded
    """
    # Always save locally as backup
    _save_local(df)

    if not use_hopsworks or not HOPSWORKS_API_KEY:
        logger.info("Saved features locally (Hopsworks not configured)")
        return True

    try:
        project = _get_hopsworks_connection()
        if project is None:
            return True  # Already saved locally

        fs = _get_feature_store(project)

        # Create or get Feature Group
        fg = fs.get_or_create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            description="Lahore AQI features with weather and pollutant data",
            primary_key=["unix_timestamp"],
            event_time="datetime",
        )

        # Insert data
        fg.insert(df, write_options={"wait_for_job": True})
        logger.info(f"Inserted {len(df)} rows into Hopsworks Feature Group '{FEATURE_GROUP_NAME}'")
        return True

    except Exception as e:
        logger.error(f"Failed to insert features into Hopsworks: {e}")
        logger.info("Features were saved locally as fallback")
        return True


def get_features(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_hopsworks: bool = True,
) -> pd.DataFrame:
    """
    Retrieve features from the Feature Store.

    Args:
        start_date: ISO format start date (e.g., '2025-01-01')
        end_date: ISO format end date
        use_hopsworks: Whether to attempt Hopsworks retrieval

    Returns:
        DataFrame with features
    """
    if use_hopsworks and HOPSWORKS_API_KEY:
        try:
            project = _get_hopsworks_connection()
            if project:
                fs = _get_feature_store(project)
                fg = fs.get_feature_group(
                    name=FEATURE_GROUP_NAME,
                    version=FEATURE_GROUP_VERSION,
                )
                df = fg.read()

                if start_date:
                    df = df[df["datetime"] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df["datetime"] <= pd.to_datetime(end_date)]

                logger.info(f"Retrieved {len(df)} rows from Hopsworks")
                return df.sort_values("datetime").reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Could not read from Hopsworks: {e}. Falling back to local.")

    # Fallback: read from local Parquet
    return _load_local(start_date, end_date)


def get_training_data(use_hopsworks: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Get feature matrix X and target vector y for model training.

    Returns:
        Tuple of (X, y) where X is feature DataFrame and y is target Series
    """
    df = get_features(use_hopsworks=use_hopsworks)

    if df.empty:
        raise ValueError("No features available for training. Run the feature pipeline first.")

    # Select only the features we need
    available_features = [f for f in ALL_FEATURES if f in df.columns]
    X = df[available_features]
    y = df[TARGET]

    # Remove rows with NaN in target
    mask = y.notna()
    X = X[mask]
    y = y[mask]

    logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def create_feature_view(use_hopsworks: bool = True):
    """Create a Feature View in Hopsworks for training data extraction."""
    if not use_hopsworks or not HOPSWORKS_API_KEY:
        logger.info("Skipping Feature View creation (Hopsworks not configured)")
        return None

    try:
        project = _get_hopsworks_connection()
        if project is None:
            return None

        fs = _get_feature_store(project)
        fg = fs.get_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
        )

        query = fg.select_all()
        fv = fs.get_or_create_feature_view(
            name=FEATURE_VIEW_NAME,
            version=FEATURE_VIEW_VERSION,
            description="Feature View for AQI prediction training",
            query=query,
        )
        logger.info(f"Feature View '{FEATURE_VIEW_NAME}' created/retrieved")
        return fv

    except Exception as e:
        logger.error(f"Failed to create Feature View: {e}")
        return None


# ─── Local Storage Helpers ────────────────────────────────────────

def _save_local(df: pd.DataFrame):
    """Append features to local Parquet file."""
    if LOCAL_FEATURE_FILE.exists():
        existing = pd.read_parquet(LOCAL_FEATURE_FILE)
        df = pd.concat([existing, df], ignore_index=True)
        # Remove duplicates based on timestamp
        if "unix_timestamp" in df.columns:
            df = df.drop_duplicates(subset=["unix_timestamp"], keep="last")
        df = df.sort_values("datetime").reset_index(drop=True)

    df.to_parquet(LOCAL_FEATURE_FILE, index=False)
    logger.info(f"Saved {len(df)} total rows to {LOCAL_FEATURE_FILE}")


def _load_local(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load features from local Parquet file."""
    if not LOCAL_FEATURE_FILE.exists():
        logger.warning(f"No local feature file found at {LOCAL_FEATURE_FILE}")
        return pd.DataFrame()

    df = pd.read_parquet(LOCAL_FEATURE_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"])

    if start_date:
        df = df[df["datetime"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["datetime"] <= pd.to_datetime(end_date)]

    logger.info(f"Loaded {len(df)} rows from local storage")
    return df.sort_values("datetime").reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Feature Store status:")
    print(f"  Local file exists: {LOCAL_FEATURE_FILE.exists()}")
    if LOCAL_FEATURE_FILE.exists():
        df = _load_local()
        print(f"  Local rows: {len(df)}")
        print(f"  Date range: {df['datetime'].min()} → {df['datetime'].max()}")
