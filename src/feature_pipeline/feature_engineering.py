"""
Feature engineering for AQI prediction.
Computes time-based, derived, lag, and rolling features.
Converts raw pollutant concentrations to EPA AQI values.
"""
import numpy as np
import pandas as pd

from src.config import (
    WEATHER_FEATURES, TARGET,
)


# ─── EPA AQI Breakpoints ─────────────────────────────────────────
# Format: (C_low, C_high, I_low, I_high)
EPA_BREAKPOINTS = {
    "pm2_5": [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ],
    "pm10": [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 504, 301, 400),
        (505, 604, 401, 500),
    ],
    "o3": [  # 8-hour average in ppb → we use µg/m³, approximate conversion
        (0, 54, 0, 50),
        (55, 70, 51, 100),
        (71, 85, 101, 150),
        (86, 105, 151, 200),
        (106, 200, 201, 300),
    ],
    "no2": [  # ppb
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 1649, 301, 400),
        (1650, 2049, 401, 500),
    ],
    "so2": [  # ppb
        (0, 35, 0, 50),
        (36, 75, 51, 100),
        (76, 185, 101, 150),
        (186, 304, 151, 200),
        (305, 604, 201, 300),
        (605, 804, 301, 400),
        (805, 1004, 401, 500),
    ],
    "co": [  # ppm → OWM gives µg/m³, convert: 1 ppm ≈ 1145 µg/m³
        (0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 40.4, 301, 400),
        (40.5, 50.4, 401, 500),
    ],
}


def _compute_sub_aqi(concentration: float, breakpoints: list) -> float:
    """Compute AQI for a single pollutant using EPA breakpoint formula."""
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= concentration <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
            return round(aqi)
    # If concentration exceeds all breakpoints, return max AQI
    if concentration > breakpoints[-1][1]:
        return 500
    return 0


def compute_aqi(row: pd.Series) -> float:
    """
    Compute overall AQI from pollutant concentrations.
    AQI = max of all sub-AQIs (EPA standard).
    CO is converted from µg/m³ to ppm for breakpoint lookup.
    """
    sub_aqis = []

    for pollutant in ["pm2_5", "pm10", "no2", "so2", "o3"]:
        if pollutant in row and pd.notna(row[pollutant]):
            bp = EPA_BREAKPOINTS.get(pollutant)
            if bp:
                sub_aqis.append(_compute_sub_aqi(row[pollutant], bp))

    # CO: convert µg/m³ to ppm (1 ppm ≈ 1145 µg/m³)
    if "co" in row and pd.notna(row["co"]):
        co_ppm = row["co"] / 1145.0
        sub_aqis.append(_compute_sub_aqi(co_ppm, EPA_BREAKPOINTS["co"]))

    return max(sub_aqis) if sub_aqis else 0


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features from the datetime column."""
    df = df.copy()
    dt = df["datetime"]

    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["day_of_year"] = dt.dt.dayofyear

    # Season: 1=Winter, 2=Spring, 3=Summer, 4=Fall (Northern Hemisphere)
    df["season"] = df["month"].map(
        lambda m: 1 if m in [12, 1, 2] else 2 if m in [3, 4, 5]
        else 3 if m in [6, 7, 8] else 4
    )

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features: change rates, ratios, rolling averages."""
    df = df.copy()

    # AQI change rate (from PAST values only — shift(1) vs shift(2) to avoid leaking target)
    df["aqi_change_rate"] = (df[TARGET].shift(1) - df[TARGET].shift(2)).fillna(0)

    # PM2.5 / PM10 ratio
    df["pm25_pm10_ratio"] = np.where(
        df["pm10"] > 0,
        df["pm2_5"] / df["pm10"],
        0
    )

    # Rolling averages for AQI — SHIFTED by 1 to exclude current value (prevent leakage)
    for window in [6, 12, 24]:
        col_name = f"rolling_aqi_{window}h"
        df[col_name] = df[TARGET].rolling(window=window, min_periods=1).mean().shift(1).fillna(0)

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag features for AQI at various time steps."""
    df = df.copy()
    for lag in [1, 3, 6, 12, 24]:
        df[f"aqi_lag_{lag}"] = df[TARGET].shift(lag).bfill()
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline:
    1. Compute AQI from pollutant concentrations
    2. Add time-based features
    3. Add derived features (ratios, rolling averages)
    4. Add lag features
    5. Clean and fill missing values
    """
    df = df.copy()

    # Ensure datetime is parsed
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

    # Step 1: Compute AQI
    df[TARGET] = df.apply(compute_aqi, axis=1)

    # Step 2: Time features
    df = add_time_features(df)

    # Step 3: Derived features
    df = add_derived_features(df)

    # Step 4: Lag features
    df = add_lag_features(df)

    # Step 5: Fill remaining NaN
    df = df.fillna(0)

    # Ensure default weather columns exist (may be missing in air-only data)
    for col in WEATHER_FEATURES:
        if col not in df.columns:
            df[col] = 0

    return df


if __name__ == "__main__":
    # Quick test with synthetic data
    test_data = pd.DataFrame({
        "datetime": pd.date_range("2025-01-01", periods=48, freq="h"),
        "pm2_5": np.random.uniform(10, 200, 48),
        "pm10": np.random.uniform(20, 300, 48),
        "no2": np.random.uniform(5, 100, 48),
        "so2": np.random.uniform(2, 50, 48),
        "co": np.random.uniform(200, 5000, 48),
        "o3": np.random.uniform(10, 80, 48),
        "temperature": np.random.uniform(10, 35, 48),
        "humidity": np.random.uniform(30, 90, 48),
        "wind_speed": np.random.uniform(0, 15, 48),
        "pressure": np.random.uniform(1000, 1025, 48),
    })
    result = engineer_features(test_data)
    print(f"Features shape: {result.shape}")
    print(f"Columns: {list(result.columns)}")
    print(result.head())
