"""
Tests for the Inference Pipeline.
Verifies prediction output format and alert system.
"""


class TestInference:
    """Test inference pipeline components."""

    def test_get_aqi_category_good(self):
        """Test AQI category lookup for Good range."""
        from src.inference.predict import get_aqi_category

        result = get_aqi_category(25)
        assert result["category"] == "Good"
        assert result["color"] == "#00E400"

    def test_get_aqi_category_unhealthy(self):
        """Test AQI category for Unhealthy range."""
        from src.inference.predict import get_aqi_category

        result = get_aqi_category(175)
        assert result["category"] == "Unhealthy"
        assert result["color"] == "#FF0000"

    def test_get_aqi_category_hazardous(self):
        """Test AQI category for Hazardous range."""
        from src.inference.predict import get_aqi_category

        result = get_aqi_category(350)
        assert result["category"] == "Hazardous"

    def test_get_aqi_category_beyond_scale(self):
        """Test AQI category for values beyond 500."""
        from src.inference.predict import get_aqi_category

        result = get_aqi_category(550)
        assert result["category"] == "Hazardous"


class TestAlerts:
    """Test the alert system."""

    def test_check_alerts_no_alerts(self):
        """Test that no alerts are generated for safe AQI."""
        from app.alerts import check_alerts

        predictions = [
            {"predicted_aqi": 45, "day": 1, "date": "2025-01-01", "day_name": "Monday"},
            {"predicted_aqi": 80, "day": 2, "date": "2025-01-02", "day_name": "Tuesday"},
            {"predicted_aqi": 30, "day": 3, "date": "2025-01-03", "day_name": "Wednesday"},
        ]
        alerts = check_alerts(predictions)
        assert len(alerts) == 0

    def test_check_alerts_unhealthy(self):
        """Test alert generation for unhealthy AQI."""
        from app.alerts import check_alerts

        predictions = [
            {"predicted_aqi": 45, "day": 1, "date": "2025-01-01", "day_name": "Monday"},
            {"predicted_aqi": 180, "day": 2, "date": "2025-01-02", "day_name": "Tuesday"},
            {"predicted_aqi": 220, "day": 3, "date": "2025-01-03", "day_name": "Wednesday"},
        ]
        alerts = check_alerts(predictions)

        assert len(alerts) == 2
        assert alerts[0]["aqi"] == 180
        assert alerts[0]["category"] == "Unhealthy"
        assert alerts[1]["aqi"] == 220
        assert alerts[1]["category"] == "Very Unhealthy"

    def test_check_alerts_hazardous(self):
        """Test alert generation for hazardous AQI."""
        from app.alerts import check_alerts

        predictions = [
            {"predicted_aqi": 350, "day": 1, "date": "2025-01-01", "day_name": "Monday"},
        ]
        alerts = check_alerts(predictions)

        assert len(alerts) == 1
        assert alerts[0]["severity"] == 5

    def test_alert_recommendations(self):
        """Test that recommendations are provided."""
        from app.alerts import check_alerts

        predictions = [
            {"predicted_aqi": 200, "day": 1, "date": "2025-01-01", "day_name": "Monday"},
        ]
        alerts = check_alerts(predictions)

        assert len(alerts) == 1
        assert "recommendations" in alerts[0]
        assert len(alerts[0]["recommendations"]) > 0

    def test_format_alert_banner(self):
        """Test alert banner formatting."""
        from app.alerts import format_alert_banner, check_alerts

        predictions = [
            {"predicted_aqi": 250, "day": 1, "date": "2025-01-01", "day_name": "Monday"},
        ]
        alerts = check_alerts(predictions)
        banner = format_alert_banner(alerts)

        assert "ALERT" in banner
        assert "Monday" in banner

    def test_format_alert_banner_empty(self):
        """Test alert banner with no alerts."""
        from app.alerts import format_alert_banner

        banner = format_alert_banner([])
        assert banner == ""


class TestAQICategories:
    """Test AQI category definitions."""

    def test_all_categories_covered(self):
        """Test that all AQI values 0-500 map to a category."""
        from app.alerts import get_aqi_category

        for aqi in range(0, 501):
            result = get_aqi_category(aqi)
            assert "category" in result
            assert "color" in result
            assert "health" in result

    def test_category_boundaries(self):
        """Test exact boundary values."""
        from app.alerts import get_aqi_category

        assert get_aqi_category(0)["category"] == "Good"
        assert get_aqi_category(50)["category"] == "Good"
        assert get_aqi_category(51)["category"] == "Moderate"
        assert get_aqi_category(100)["category"] == "Moderate"
        assert get_aqi_category(101)["category"] == "Unhealthy for Sensitive Groups"
        assert get_aqi_category(150)["category"] == "Unhealthy for Sensitive Groups"
        assert get_aqi_category(151)["category"] == "Unhealthy"
        assert get_aqi_category(200)["category"] == "Unhealthy"
        assert get_aqi_category(201)["category"] == "Very Unhealthy"
        assert get_aqi_category(300)["category"] == "Very Unhealthy"
        assert get_aqi_category(301)["category"] == "Hazardous"
        assert get_aqi_category(500)["category"] == "Hazardous"
