"""Tests for content-based duplicate detection.

Replaces a circular arrangement: the injector used to add
`_is_duplicate_flag` / `duplicate_uuid` and the feature extractor set its
`duplicate_flag` feature from the presence of those exact keys, so for the
duplicate class the feature *was* the label. Detection is now derived from a
content fingerprint.
"""
import random

import pytest

from src.ingestion.dedup import DEFAULT_CAPACITY, DuplicateTracker, fingerprint
from src.ingestion.injector import SyntheticAnomalyInjector
from src.ingestion.pipeline_registry import registry
from src.layer1_detector.feature_extractor import FeatureExtractor
from src.layer1_detector.model import MLAnomalyDetector

RECORD = {"amount": 12.5, "currency": "USD", "user_id": "USR-1"}


# -- fingerprinting ---------------------------------------------------------

def test_fingerprint_is_order_independent():
    reordered = dict(reversed(list(RECORD.items())))
    assert fingerprint(RECORD) == fingerprint(reordered)


def test_fingerprint_changes_with_content():
    altered = dict(RECORD, amount=12.6)
    assert fingerprint(RECORD) != fingerprint(altered)


def test_fingerprint_ignores_annotation_keys():
    """Annotating a record must not change its identity."""
    annotated = dict(RECORD, _healing={"amount": {"actions": ["defaulted"]}})
    annotated["_is_duplicate"] = True
    assert fingerprint(annotated) == fingerprint(RECORD)


# -- tracker ----------------------------------------------------------------

def test_first_sighting_is_not_a_duplicate():
    assert DuplicateTracker().observe(dict(RECORD)) is False


def test_second_sighting_is_a_duplicate():
    tracker = DuplicateTracker()
    tracker.observe(dict(RECORD))
    assert tracker.observe(dict(RECORD)) is True


def test_seen_does_not_record():
    tracker = DuplicateTracker()
    assert tracker.seen(RECORD) is False
    assert tracker.seen(RECORD) is False
    assert tracker.observe(dict(RECORD)) is False


def test_capacity_is_bounded_and_evicts_oldest():
    tracker = DuplicateTracker(capacity=3)
    for index in range(10):
        tracker.observe({"n": index})
    assert len(tracker) == 3
    # The oldest entries were evicted, so they read as new again.
    assert tracker.observe({"n": 0}) is False
    # The most recent is still remembered.
    assert tracker.observe({"n": 9}) is True


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        DuplicateTracker(capacity=0)


def test_reset_clears_memory():
    tracker = DuplicateTracker()
    tracker.observe(dict(RECORD))
    tracker.reset()
    assert tracker.observe(dict(RECORD)) is False
    assert tracker.stats()["observed_count"] == 1


def test_default_capacity_is_sane():
    assert DEFAULT_CAPACITY >= 256


# -- injector ---------------------------------------------------------------

def test_injector_no_longer_plants_duplicate_markers():
    """The marker keys that made the benchmark circular must be gone."""
    pipeline = registry.require_pipeline("weather")
    injector = SyntheticAnomalyInjector(rng=random.Random(3))

    for _ in range(4):
        injector.inject_anomaly(pipeline.cached_baseline(), target_anomaly="normal")

    record, label, meta = injector.inject_anomaly(
        pipeline.cached_baseline(), target_anomaly="duplicate_record"
    )
    assert label == "duplicate_record"
    assert "_is_duplicate_flag" not in record
    assert "duplicate_uuid" not in record
    assert meta["replayed"] is True


def test_replayed_record_is_detected_by_content():
    pipeline = registry.require_pipeline("finance")
    injector = SyntheticAnomalyInjector(rng=random.Random(5))
    tracker = DuplicateTracker()

    for _ in range(5):
        record, _, _ = injector.inject_anomaly(
            pipeline.fetch_clean_sample(), target_anomaly="normal"
        )
        tracker.observe(record)

    replay, _, meta = injector.inject_anomaly(
        pipeline.fetch_clean_sample(), target_anomaly="duplicate_record"
    )
    assert meta["replayed"] is True
    assert tracker.observe(replay) is True


def test_replay_without_history_is_reported_honestly():
    """With nothing to replay, the injector must not claim a duplicate."""
    pipeline = registry.require_pipeline("iot")
    injector = SyntheticAnomalyInjector(rng=random.Random(11))

    _, label, meta = injector.inject_anomaly(
        pipeline.fetch_clean_sample(), target_anomaly="duplicate_record"
    )
    assert label == "duplicate_record"
    assert meta["replayed"] is False
    assert meta["action"] == "no_history_to_replay"


# -- feature extraction and the detector ------------------------------------

def test_duplicate_flag_is_zero_without_a_tracker():
    """A single record in isolation cannot be known to be a duplicate."""
    features = FeatureExtractor().extract_feature_dict(dict(RECORD))
    assert features["duplicate_flag"] == 0.0


def test_duplicate_flag_comes_from_the_tracker():
    extractor = FeatureExtractor()
    tracker = DuplicateTracker()

    first = extractor.extract_feature_dict(dict(RECORD), duplicate_tracker=tracker)
    second = extractor.extract_feature_dict(dict(RECORD), duplicate_tracker=tracker)

    assert first["duplicate_flag"] == 0.0
    assert second["duplicate_flag"] == 1.0


def test_detector_flags_a_repeat_of_an_observed_record():
    pipeline = registry.require_pipeline("weather")
    detector = MLAnomalyDetector(duplicate_tracker=DuplicateTracker())
    record = pipeline.cached_baseline()

    detector.observe(record)
    result = detector.detect(
        dict(record),
        expected_fields=pipeline.expected_schema,
        valid_ranges=pipeline.valid_ranges,
    )
    assert result["anomaly_type"] == "duplicate_record"
    assert result["is_anomaly"] is True


def test_track_duplicates_false_does_not_record():
    detector = MLAnomalyDetector(duplicate_tracker=DuplicateTracker())
    record = registry.require_pipeline("weather").cached_baseline()

    detector.detect(dict(record), track_duplicates=False)
    detector.detect(dict(record), track_duplicates=False)
    assert len(detector.duplicate_tracker) == 0


def test_classify_scores_a_precomputed_vector():
    """Lets a caller extract features once and score both engines."""
    detector = MLAnomalyDetector()
    features = detector.extractor.extract_feature_dict(
        registry.require_pipeline("weather").cached_baseline()
    )
    assert detector.classify(features)["anomaly_type"] == "normal"
    assert detector.rule_based_predict(features)["anomaly_type"] == "normal"


# -- regressions found by end-to-end testing --------------------------------

def test_replay_history_is_partitioned_by_record_shape():
    """A replay must come from the same stream, not a different pipeline.

    One shared history let a finance duplicate_record request replay a
    weather record. The result was classified schema_drift -- every finance
    field missing, every weather field extra -- and "healed" into a 16-field
    weather/finance hybrid.
    """
    injector = SyntheticAnomalyInjector(rng=random.Random(17))
    weather = registry.require_pipeline("weather")
    finance = registry.require_pipeline("finance")

    # Build history for weather only.
    for _ in range(5):
        injector.inject_anomaly(weather.cached_baseline(), target_anomaly="normal")

    # A finance replay must not borrow a weather record.
    record, _, meta = injector.inject_anomaly(
        finance.fetch_clean_sample(), target_anomaly="duplicate_record"
    )
    assert meta["replayed"] is False, "replayed a record from another pipeline"
    assert set(record) == set(finance.expected_schema)

    # Once finance has its own history, replay works within that stream.
    for _ in range(3):
        injector.inject_anomaly(finance.fetch_clean_sample(), target_anomaly="normal")
    record, _, meta = injector.inject_anomaly(
        finance.fetch_clean_sample(), target_anomaly="duplicate_record"
    )
    assert meta["replayed"] is True
    assert set(record) == set(finance.expected_schema)


def test_detect_accepts_a_tracker_override():
    """Lets a caller score against one stream's state, not the detector's."""
    pipeline = registry.require_pipeline("weather")
    detector = MLAnomalyDetector()
    record = pipeline.cached_baseline()

    isolated = DuplicateTracker()
    isolated.observe(record)

    result = detector.detect(
        dict(record),
        expected_fields=pipeline.expected_schema,
        valid_ranges=pipeline.valid_ranges,
        duplicate_tracker=isolated,
    )
    assert result["anomaly_type"] == "duplicate_record"
    # The detector's own tracker must be untouched by the override.
    assert len(detector.duplicate_tracker) == 0
