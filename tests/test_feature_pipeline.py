"""
Tests for the Feature Pipeline.
Covers data fetching (mocked), feature engineering, and feature store operations.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock


# ─── Test Feature Engineering ──────────────────────────────────────

class TestFeatureEngineering:
    """Tests for feature_engineering.py"""

    @pytest.fixture
    def sample_data(self):
        """Create a sample DataFrame mimicking API data."""
        np.random.seed(42)
        n = 48  # 2 days of hourly data
        return pd.DataFrame({
            "datetime": pd.date_range("2025-01-01", periods=n, freq="h"),
            "pm2_5": np.random.uniform(10, 200, n),
            "pm10": np.random.uniform(20, 300, n),
            "no2": np.random.uniform(5, 100, n),
            "so2": np.random.uniform(2, 50, n),
            "co": np.random.uniform(200, 5000, n),
            "o3": np.random.uniform(10, 80, n),
            "temperature": np.random.uniform(10, 35, n),
            "humidity": np.random.uniform(30, 90, n),
            "wind_speed": np.random.uniform(0, 15, n),
            "pressure": np.random.uniform(1000, 1025, n),
        })

    def test_compute_aqi_good(self):
        """Test AQI computation for Good range."""
        from src.feature_pipeline.feature_engineering import compute_aqi
        row = pd.Series({"pm2_5": 5.0, "pm10": 20, "no2": 10, "so2": 5, "co": 500, "o3": 20})
        aqi = compute_aqi(row)
        assert 0 <= aqi <= 50, f"Expected Good AQI (0-50), got {aqi}"

    def test_compute_aqi_unhealthy(self):
        """Test AQI computation for Unhealthy range."""
        from src.feature_pipeline.feature_engineering import compute_aqi
        row = pd.Series({"pm2_5": 100.0, "pm10": 200, "no2": 50, "so2": 20, "co": 2000, "o3": 40})
        aqi = compute_aqi(row)
        assert aqi > 100, f"Expected Unhealthy AQI (>100), got {aqi}"

    def test_engineer_features_output_shape(self, sample_data):
        """Test that engineer_features produces expected columns."""
        from src.feature_pipeline.feature_engineering import engineer_features
        result = engineer_features(sample_data)

        # Should have all input columns plus engineered ones
        assert "aqi" in result.columns
        assert "hour" in result.columns
        assert "day_of_week" in result.columns
        assert "month" in result.columns
        assert "is_weekend" in result.columns
        assert "season" in result.columns
        assert "aqi_change_rate" in result.columns
        assert "pm25_pm10_ratio" in result.columns
        assert "rolling_aqi_6h" in result.columns
        assert "aqi_lag_1" in result.columns

    def test_engineer_features_no_nan(self, sample_data):
        """Test that engineer_features fills all NaN values."""
        from src.feature_pipeline.feature_engineering import engineer_features
        result = engineer_features(sample_data)
        assert result.isnull().sum().sum() == 0, "Found NaN values in engineered features"

    def test_time_features_correct(self, sample_data):
        """Test time features are computed correctly."""
        from src.feature_pipeline.feature_engineering import add_time_features
        result = add_time_features(sample_data)

        # First row is 2025-01-01 00:00 (Wednesday)
        assert result.iloc[0]["hour"] == 0
        assert result.iloc[0]["day_of_week"] == 2  # Wednesday
        assert result.iloc[0]["month"] == 1
        assert result.iloc[0]["season"] == 1  # Winter

    def test_lag_features(self, sample_data):
        """Test lag features are properly shifted."""
        from src.feature_pipeline.feature_engineering import engineer_features
        result = engineer_features(sample_data)

        # At index 1, lag_1 should equal aqi at index 0
        assert result.iloc[1]["aqi_lag_1"] == result.iloc[0]["aqi"]

    def test_rolling_features(self, sample_data):
        """Test rolling average features."""
        from src.feature_pipeline.feature_engineering import engineer_features
        result = engineer_features(sample_data)

        # Rolling averages should be between min and max AQI
        assert result["rolling_aqi_6h"].min() >= 0
        assert result["rolling_aqi_24h"].min() >= 0


# ─── Test Data Fetching (Mocked) ──────────────────────────────────

class TestFetchData:
    """Tests for fetch_data.py with mocked API responses."""

    @pytest.fixture
    def mock_air_pollution_response(self):
        """Mock OpenWeatherMap air pollution API response."""
        return {
            "list": [
                {
                    "dt": 1704067200,
                    "main": {"aqi": 4},
                    "components": {
                        "co": 2000.5,
                        "no": 5.2,
                        "no2": 45.3,
                        "o3": 30.1,
                        "so2": 12.4,
                        "pm2_5": 85.2,
                        "pm10": 120.5,
                        "nh3": 3.1,
                    },
                }
            ]
        }

    @pytest.fixture
    def mock_weather_response(self):
        """Mock OpenWeatherMap weather API response."""
        return {
            "dt": 1704067200,
            "main": {
                "temp": 22.5,
                "feels_like": 21.0,
                "humidity": 65,
                "pressure": 1013,
            },
            "wind": {"speed": 3.5, "deg": 180},
            "visibility": 8000,
            "clouds": {"all": 40},
        }

    @patch("src.feature_pipeline.fetch_data.requests.get")
    def test_fetch_current_air_pollution(self, mock_get, mock_air_pollution_response):
        """Test fetching current air pollution data."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: mock_air_pollution_response,
        )
        mock_get.return_value.raise_for_status = MagicMock()

        from src.feature_pipeline.fetch_data import fetch_current_air_pollution
        df = fetch_current_air_pollution()

        assert not df.empty
        assert "pm2_5" in df.columns
        assert "pm10" in df.columns
        assert "datetime" in df.columns
        assert df.iloc[0]["pm2_5"] == 85.2

    @patch("src.feature_pipeline.fetch_data.requests.get")
    def test_fetch_current_weather(self, mock_get, mock_weather_response):
        """Test fetching current weather data."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: mock_weather_response,
        )
        mock_get.return_value.raise_for_status = MagicMock()

        from src.feature_pipeline.fetch_data import fetch_current_weather
        df = fetch_current_weather()

        assert not df.empty
        assert "temperature" in df.columns
        assert "humidity" in df.columns
        assert df.iloc[0]["temperature"] == 22.5

    @patch("src.feature_pipeline.fetch_data.requests.get")
    def test_api_retry_on_failure(self, mock_get):
        """Test retry logic on API failure."""
        mock_get.side_effect = [
            Exception("Connection error"),
            Exception("Timeout"),
            MagicMock(
                status_code=200,
                json=lambda: {"list": []},
                raise_for_status=MagicMock(),
            ),
        ]

        from src.feature_pipeline.fetch_data import fetch_current_air_pollution
        df = fetch_current_air_pollution()
        assert df.empty  # Empty response but no crash


# ─── Test Feature Store (Local) ──────────────────────────────────

class TestFeatureStore:
    """Tests for feature_store.py local storage."""

    @pytest.fixture
    def sample_features(self):
        """Create sample engineered features."""
        return pd.DataFrame({
            "datetime": pd.date_range("2025-01-01", periods=24, freq="h"),
            "unix_timestamp": range(1704067200, 1704067200 + 24 * 3600, 3600),
            "aqi": np.random.uniform(50, 200, 24),
            "pm2_5": np.random.uniform(10, 100, 24),
            "temperature": np.random.uniform(10, 30, 24),
        })

    def test_save_and_load_local(self, sample_features, tmp_path):
        """Test saving and loading features locally."""
        from src.feature_pipeline.feature_store import _save_local, _load_local

        # Temporarily change the file path
        import src.feature_pipeline.feature_store as fs_module
        original_path = fs_module.LOCAL_FEATURE_FILE
        fs_module.LOCAL_FEATURE_FILE = tmp_path / "test_features.parquet"

        try:
            _save_local(sample_features)
            loaded = _load_local()

            assert len(loaded) == len(sample_features)
            assert "aqi" in loaded.columns
        finally:
            fs_module.LOCAL_FEATURE_FILE = original_path
