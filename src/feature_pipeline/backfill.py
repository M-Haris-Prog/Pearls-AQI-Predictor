"""
Historical data backfill script.
Fetches past air pollution data from OpenWeatherMap and stores in Feature Store.
Supports resume-from-checkpoint via local progress tracking.
"""
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.feature_pipeline.fetch_data import fetch_air_pollution_history, fetch_current_weather
from src.feature_pipeline.feature_engineering import engineer_features
from src.feature_pipeline.feature_store import insert_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CHECKPOINT_FILE = PROCESSED_DATA_DIR / "backfill_checkpoint.json"


def load_checkpoint() -> dict:
    """Load the last backfill checkpoint."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(last_date: str, total_rows: int):
    """Save backfill progress checkpoint."""
    checkpoint = {
        "last_date": last_date,
        "total_rows": total_rows,
        "updated_at": datetime.utcnow().isoformat(),
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)
    logger.info(f"Checkpoint saved: {checkpoint}")


def backfill(
    days: int = 30,
    chunk_days: int = 7,
    use_hopsworks: bool = True,
    resume: bool = True,
):
    """
    Backfill historical AQI data.

    Args:
        days: Number of days to backfill from today
        chunk_days: Process data in chunks of this many days
        use_hopsworks: Whether to store in Hopsworks
        resume: Whether to resume from last checkpoint
    """
    logger.info("=" * 60)
    logger.info(f"Starting Historical Backfill: {days} days, {chunk_days}-day chunks")
    logger.info("=" * 60)

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    # Resume from checkpoint if available
    if resume:
        checkpoint = load_checkpoint()
        if checkpoint.get("last_date"):
            resume_date = datetime.fromisoformat(checkpoint["last_date"])
            if resume_date > start_date:
                start_date = resume_date
                logger.info(f"Resuming from checkpoint: {start_date.isoformat()}")

    total_rows = 0
    current_start = start_date

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk_days), end_date)

        start_ts = int(current_start.timestamp())
        end_ts = int(current_end.timestamp())

        logger.info(f"Fetching: {current_start.date()} → {current_end.date()}")

        try:
            # Fetch historical air pollution data
            air_df = fetch_air_pollution_history(start_ts, end_ts)

            if air_df.empty:
                logger.warning(f"  No data for {current_start.date()} → {current_end.date()}")
                current_start = current_end
                continue

            logger.info(f"  Fetched {len(air_df)} records")

            # Add placeholder weather data (historical weather not in free tier)
            # In production, you'd use a weather history API
            for col in ["temperature", "humidity", "wind_speed", "pressure"]:
                if col not in air_df.columns:
                    air_df[col] = 0

            # Engineer features
            features_df = engineer_features(air_df)
            logger.info(f"  Engineered {features_df.shape[0]} feature rows")

            # Store features
            insert_features(features_df, use_hopsworks=use_hopsworks)
            total_rows += len(features_df)

            # Save checkpoint
            save_checkpoint(current_end.isoformat(), total_rows)

        except Exception as e:
            logger.error(f"  Error processing chunk: {e}")
            save_checkpoint(current_start.isoformat(), total_rows)
            raise

        current_start = current_end

    logger.info("=" * 60)
    logger.info(f"Backfill complete! Total rows: {total_rows}")
    logger.info("=" * 60)
    return total_rows


def main():
    parser = argparse.ArgumentParser(description="Backfill historical AQI data")
    parser.add_argument("--days", type=int, default=30, help="Days to backfill")
    parser.add_argument("--chunk-days", type=int, default=7, help="Chunk size in days")
    parser.add_argument("--no-hopsworks", action="store_true", help="Use local storage only")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh (ignore checkpoint)")
    args = parser.parse_args()

    backfill(
        days=args.days,
        chunk_days=args.chunk_days,
        use_hopsworks=not args.no_hopsworks,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
