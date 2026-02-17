"""
AQI alert system.
Defines AQI categories, color coding, health advisories, and alert triggers.
"""
from typing import Dict, List
from src.config import AQI_CATEGORIES


def get_aqi_category(aqi_value: float) -> Dict:
    """
    Get the AQI category, color, and health advisory for a given AQI value.

    Args:
        aqi_value: Numeric AQI value (0-500+)

    Returns:
        Dictionary with category name, color, health message, and severity level
    """
    for i, (category, info) in enumerate(AQI_CATEGORIES.items()):
        if info["min"] <= aqi_value <= info["max"]:
            return {
                "category": category,
                "color": info["color"],
                "health": info["health"],
                "severity": i,  # 0=Good, 5=Hazardous
                "aqi": round(aqi_value),
            }

    # Beyond 500
    return {
        "category": "Hazardous",
        "color": "#7E0023",
        "health": "Health warnings of emergency conditions. Entire population is affected.",
        "severity": 5,
        "aqi": round(aqi_value),
    }


def check_alerts(predictions: List[Dict]) -> List[Dict]:
    """
    Check predictions for alert conditions.

    Args:
        predictions: List of prediction dicts with 'predicted_aqi' key

    Returns:
        List of alert dictionaries
    """
    alerts = []

    for pred in predictions:
        aqi = pred.get("predicted_aqi", 0)
        category_info = get_aqi_category(aqi)

        if aqi > 150:  # Unhealthy and above
            alert = {
                "day": pred.get("day", ""),
                "date": pred.get("date", ""),
                "day_name": pred.get("day_name", ""),
                "aqi": aqi,
                "category": category_info["category"],
                "color": category_info["color"],
                "severity": category_info["severity"],
                "message": _get_alert_message(aqi, category_info["category"], pred.get("day_name", "")),
                "recommendations": _get_recommendations(category_info["severity"]),
            }
            alerts.append(alert)

    return alerts


def _get_alert_message(aqi: float, category: str, day_name: str) -> str:
    """Generate a human-readable alert message."""
    if aqi > 300:
        return (f"🚨 HAZARDOUS: AQI is predicted to reach {aqi} on {day_name}. "
                f"Health emergency conditions. Avoid ALL outdoor activity.")
    elif aqi > 200:
        return (f"⚠️ VERY UNHEALTHY: AQI is predicted to reach {aqi} on {day_name}. "
                f"Everyone may experience serious health effects.")
    elif aqi > 150:
        return (f"⚠️ UNHEALTHY: AQI is predicted to reach {aqi} on {day_name}. "
                f"Everyone may begin to experience health effects.")
    else:
        return (f"⚠️ SENSITIVE GROUPS: AQI is predicted to reach {aqi} on {day_name}. "
                f"Sensitive individuals should reduce prolonged outdoor exertion.")


def _get_recommendations(severity: int) -> List[str]:
    """Get health recommendations based on severity level."""
    base = ["Monitor air quality updates regularly"]

    if severity >= 5:  # Hazardous
        return base + [
            "Stay indoors with windows and doors closed",
            "Use air purifiers if available",
            "Wear N95 mask if you must go outside",
            "Avoid all outdoor physical activity",
            "Keep emergency medications accessible",
        ]
    elif severity >= 4:  # Very Unhealthy
        return base + [
            "Avoid prolonged outdoor exertion",
            "Keep windows closed",
            "Use air purifiers",
            "Wear a mask outdoors",
            "Check on elderly and children regularly",
        ]
    elif severity >= 3:  # Unhealthy
        return base + [
            "Reduce prolonged outdoor exertion",
            "Take more breaks during outdoor activities",
            "People with respiratory conditions should stay indoors",
            "Consider wearing a mask outdoors",
        ]
    elif severity >= 2:  # Unhealthy for Sensitive
        return base + [
            "Sensitive groups should limit prolonged outdoor exertion",
            "People with asthma should keep medications handy",
            "Consider reducing intense outdoor activity",
        ]
    elif severity >= 1:  # Moderate
        return base + [
            "Unusually sensitive people should consider reducing prolonged outdoor exertion",
        ]
    else:  # Good
        return ["Air quality is satisfactory — enjoy outdoor activities!"]


def format_alert_banner(alerts: List[Dict]) -> str:
    """Format alerts into a display-ready banner string."""
    if not alerts:
        return ""

    max_severity = max(a["severity"] for a in alerts)
    if max_severity >= 5:
        icon = "🚨"
        level = "HAZARDOUS ALERT"
    elif max_severity >= 4:
        icon = "⚠️"
        level = "VERY UNHEALTHY ALERT"
    else:
        icon = "⚠️"
        level = "UNHEALTHY AIR QUALITY ALERT"

    days_affected = ", ".join(a["day_name"] for a in alerts)
    max_aqi = max(a["aqi"] for a in alerts)

    return (
        f"{icon} {level}\n"
        f"Predicted unhealthy air quality on: {days_affected}\n"
        f"Peak predicted AQI: {max_aqi}"
    )
