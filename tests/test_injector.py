import pytest
from src.ingestion.api_client import WeatherAPIClient
from src.ingestion.injector import SyntheticAnomalyInjector, ANOMALY_CLASSES

def test_anomaly_injector():
    client = WeatherAPIClient()
    clean = client.fetch_live_sample()
    injector = SyntheticAnomalyInjector()

    for anomaly_class in ANOMALY_CLASSES:
        corrupted, label, meta = injector.inject_anomaly(clean, target_anomaly=anomaly_class)
        assert label == anomaly_class
        assert isinstance(corrupted, dict)
