"""Tests for the value-compatibility feature.

String similarity cannot separate a real rename from a missing field that
merely arrived beside a lookalike key: both give missing=1, extra=1 and a low
similarity ratio, so the two cases were feature-identical and no classifier
could tell them apart. Layer 1 consequently reported
`reorder_level` -> `min_stock` (similarity 0.09) as `null_field`.

This feature scores whether an unexpected key's *value* fits the declared type
and range of a missing field, which is signal the other ten features do not
carry.
"""
import difflib

import pytest

from src.ingestion.pipeline_registry import registry
from src.layer1_detector.feature_extractor import FeatureExtractor
from src.layer1_detector.model import MLAnomalyDetector
from src.layer2_diagnostician.agent import LLMDiagnosticAgent

FEATURE = "best_alias_value_compatibility"


def _all_declared_renames():
    cases = []
    for pipeline_id in registry.pipeline_ids():
        pipeline = registry.require_pipeline(pipeline_id)
        for canonical, alias in pipeline.field_renames.items():
            cases.append(pytest.param(
                pipeline_id, canonical, alias,
                id=f"{pipeline_id}-{canonical}-to-{alias}",
            ))
    return cases


def test_feature_is_part_of_the_vector():
    assert FEATURE in FeatureExtractor.FEATURE_NAMES
    assert len(FeatureExtractor.FEATURE_NAMES) == 11


def test_a_real_rename_scores_full_compatibility():
    pipeline = registry.require_pipeline("ecommerce")
    record = dict(pipeline.fetch_clean_sample())
    record["min_stock"] = record.pop("reorder_level")

    features = FeatureExtractor().extract_feature_dict(
        record, pipeline.expected_schema, pipeline.valid_ranges
    )
    assert features[FEATURE] == 1.0


def test_a_coincidental_lookalike_key_scores_zero():
    """The case similarity gets wrong: a high ratio but an unusable value."""
    pipeline = registry.require_pipeline("ecommerce")
    record = dict(pipeline.fetch_clean_sample())
    record.pop("reorder_level")
    record["reorder_mode"] = "auto"

    features = FeatureExtractor().extract_feature_dict(
        record, pipeline.expected_schema, pipeline.valid_ranges
    )
    assert features[FEATURE] == 0.0
    # Similarity is actively misleading here: it rates the fake above the real.
    assert features["max_key_similarity"] > 0.4


def test_clean_record_scores_zero():
    """No missing fields means there is no alias candidate to score."""
    pipeline = registry.require_pipeline("finance")
    features = FeatureExtractor().extract_feature_dict(
        pipeline.fetch_clean_sample(), pipeline.expected_schema, pipeline.valid_ranges
    )
    assert features[FEATURE] == 0.0


def test_out_of_range_value_scores_partial():
    """Right type, wrong magnitude: a weaker match than an in-range value."""
    pipeline = registry.require_pipeline("finance")
    record = dict(pipeline.fetch_clean_sample())
    record.pop("risk_score")
    record["fraud_score"] = 870.5  # risk_score's range is (0.0, 1.0)

    features = FeatureExtractor().extract_feature_dict(
        record, pipeline.expected_schema, pipeline.valid_ranges
    )
    assert features[FEATURE] == 0.5


def test_boolean_is_never_compatible_with_a_numeric_field():
    """bool is an int subclass but never a real measurement."""
    pipeline = registry.require_pipeline("iot")
    record = dict(pipeline.fetch_clean_sample())
    record.pop("motor_rpm")
    record["rpm"] = True

    features = FeatureExtractor().extract_feature_dict(
        record, pipeline.expected_schema, pipeline.valid_ranges
    )
    assert features[FEATURE] == 0.0


def test_numeric_string_scores_partial():
    pipeline = registry.require_pipeline("iot")
    record = dict(pipeline.fetch_clean_sample())
    record.pop("motor_rpm")
    record["rpm"] = "3100"

    features = FeatureExtractor().extract_feature_dict(
        record, pipeline.expected_schema, pipeline.valid_ranges
    )
    assert features[FEATURE] == 0.5


# -- the behaviour the feature exists to fix --------------------------------

@pytest.mark.parametrize("pipeline_id,canonical,alias", _all_declared_renames())
def test_every_declared_rename_is_labelled_schema_drift(pipeline_id, canonical, alias):
    """Layer 1 must recognise a rename regardless of name similarity.

    Before this feature, `reorder_level` -> `min_stock` (0.09) and
    `category` -> `item_cat` (0.38) were reported as `null_field`.
    """
    pipeline = registry.require_pipeline(pipeline_id)
    record = dict(pipeline.fetch_clean_sample())
    record[alias] = record.pop(canonical)

    detection = MLAnomalyDetector().detect(
        record,
        expected_fields=pipeline.expected_schema,
        valid_ranges=pipeline.valid_ranges,
    )
    similarity = difflib.SequenceMatcher(None, canonical, alias).ratio()
    assert detection["anomaly_type"] == "schema_drift", (
        f"{canonical} -> {alias} (similarity {similarity:.2f}) "
        f"labelled {detection['anomaly_type']}"
    )


@pytest.mark.parametrize("pipeline_id,canonical,alias", _all_declared_renames())
def test_rule_baseline_also_recognises_every_declared_rename(
    pipeline_id, canonical, alias
):
    """The feature must help the rules too, not just feed the model.

    A feature that only the model can exploit is suspect; a real one improves
    the explicit baseline as well. Adding this lifted the rule baseline on the
    confounded benchmark from 89.09% to 98.33% F1.
    """
    pipeline = registry.require_pipeline(pipeline_id)
    record = dict(pipeline.fetch_clean_sample())
    record[alias] = record.pop(canonical)

    detector = MLAnomalyDetector()
    features = detector.extractor.extract_feature_dict(
        record, pipeline.expected_schema, pipeline.valid_ranges
    )
    verdict = detector.rule_based_predict(features)
    assert verdict["anomaly_type"] == "schema_drift"


@pytest.mark.parametrize("pipeline_id,canonical,alias", _all_declared_renames())
def test_layer2_names_the_renamed_field(pipeline_id, canonical, alias):
    pipeline = registry.require_pipeline(pipeline_id)
    record = dict(pipeline.fetch_clean_sample())
    record[alias] = record.pop(canonical)
    aliases = {a: c for c, a in pipeline.field_renames.items()}

    detection = MLAnomalyDetector().detect(
        record,
        expected_fields=pipeline.expected_schema,
        valid_ranges=pipeline.valid_ranges,
    )
    report = LLMDiagnosticAgent().diagnose(
        failing_payload=record,
        expected_schema=pipeline.get_schema_spec(),
        layer1_result=detection,
        valid_ranges=pipeline.valid_ranges,
        known_aliases=aliases,
    )
    assert report.anomaly_category == "schema_drift"
    assert report.affected_field == canonical
    assert report.found_replacement_key == alias
