import pytest
from src.ingestion.api_client import WeatherAPIClient
from src.ingestion.injector import SyntheticAnomalyInjector
from src.layer1_detector.feature_extractor import FeatureExtractor
from src.layer1_detector.model import MLAnomalyDetector

def test_feature_extractor():
    client = WeatherAPIClient()
    payload = client.fetch_live_sample()
    extractor = FeatureExtractor()

    features = extractor.extract_features(payload)
    assert len(features) == len(FeatureExtractor.FEATURE_NAMES)
    assert features[1] == 0.0  # missing_expected_count for clean payload

def test_ml_anomaly_detector():
    client = WeatherAPIClient()
    injector = SyntheticAnomalyInjector()
    detector = MLAnomalyDetector()

    clean = client.fetch_live_sample()
    res_clean = detector.detect(clean)
    assert res_clean["anomaly_type"] == "normal" or not res_clean["is_anomaly"]

    corrupted, _, _ = injector.inject_anomaly(clean, target_anomaly="schema_drift")
    res_corrupted = detector.detect(corrupted)
    assert res_corrupted["is_anomaly"] == True
