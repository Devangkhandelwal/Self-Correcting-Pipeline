import requests
import json
from typing import Dict, Any, List

EXPECTED_SCHEMA = {
    "latitude": float,
    "longitude": float,
    "temperature_2m": float,
    "relative_humidity_2m": float,
    "precipitation": float,
    "surface_pressure": float,
    "wind_speed_10m": float,
    "time": str
}

class WeatherAPIClient:
    """Fetches weather data from Open-Meteo API."""
    def __init__(self, lat: float = 40.7128, lon: float = -74.0060):
        self.url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m"

    def fetch_live_sample(self) -> Dict[str, Any]:
        """Fetches a single clean weather payload record."""
        try:
            resp = requests.get(self.url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            current = data.get("current", {})
            payload = {
                "latitude": data.get("latitude"),
                "longitude": data.get("longitude"),
                "temperature_2m": current.get("temperature_2m"),
                "relative_humidity_2m": current.get("relative_humidity_2m"),
                "precipitation": current.get("precipitation"),
                "surface_pressure": current.get("surface_pressure"),
                "wind_speed_10m": current.get("wind_speed_10m"),
                "time": current.get("time")
            }
            return payload
        except Exception as e:
            # Fallback mock clean data if offline or API error
            return {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "temperature_2m": 22.4,
                "relative_humidity_2m": 65.0,
                "precipitation": 0.0,
                "surface_pressure": 1013.2,
                "wind_speed_10m": 12.5,
                "time": "2026-08-22T12:00"
            }

    @staticmethod
    def get_expected_schema_spec() -> Dict[str, str]:
        return {k: v.__name__ for k, v in EXPECTED_SCHEMA.items()}
