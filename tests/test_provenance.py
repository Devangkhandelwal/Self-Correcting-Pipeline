"""Tests that healed records disclose what the parser changed.

Without this, healing is indistinguishable from real data downstream: a
backfilled finance `amount` of 0.01 looked exactly like a genuine 0.01, and a
clamped sensor reading looked like a real measurement.
"""
import pytest

from src.ingestion.pipeline_registry import registry
from src.layer1_detector.model import MLAnomalyDetector
from src.layer2_diagnostician.agent import DiagnosticReport, LLMDiagnosticAgent
from src.layer3_patcher.auto_healer import SelfHealingOrchestrator
from src.layer3_patcher.generator import PROVENANCE_KEY


def _active_parser(pipeline_id: str):
    """Heal a pipeline so an active parser is bound, and return it."""
    pipeline = registry.require_pipeline(pipeline_id)
    aliases = {a: c for c, a in pipeline.field_renames.items()}
    report = DiagnosticReport(
        cause="test fixture",
        affected_field=next(iter(pipeline.expected_schema)),
        anomaly_category="schema_drift",
        expected_type="float",
        actual_type="missing_renamed",
        suggested_fix="n/a",
    )
    orchestrator = SelfHealingOrchestrator()
    success, _, verification = orchestrator.heal_pipeline(
        diagnostic=report,
        failing_payload=pipeline.fetch_clean_sample(),
        pipeline_id=pipeline_id,
        expected_schema=pipeline.expected_schema,
        valid_ranges=pipeline.valid_ranges,
        known_aliases=aliases,
    )
    assert success, verification.get("error_trace", "")[:300]
    return pipeline, orchestrator


def test_clean_record_carries_no_provenance():
    """A record that needed no healing must not be annotated."""
    pipeline, orchestrator = _active_parser("finance")
    healed = orchestrator.parse(pipeline.fetch_clean_sample(), pipeline_id="finance")
    assert PROVENANCE_KEY not in healed


def test_backfilled_field_is_marked_defaulted():
    pipeline, orchestrator = _active_parser("finance")
    record = pipeline.fetch_clean_sample()
    record.pop("amount")

    healed = orchestrator.parse(record, pipeline_id="finance")

    assert PROVENANCE_KEY in healed
    entry = healed[PROVENANCE_KEY]["amount"]
    assert "defaulted" in entry["actions"]
    # The default must itself satisfy the declared range.
    low, high = pipeline.valid_ranges["amount"]
    assert low <= healed["amount"] <= high


def test_clamped_field_records_the_original_value():
    pipeline, orchestrator = _active_parser("iot")
    record = pipeline.fetch_clean_sample()
    record["motor_rpm"] = 999999.0

    healed = orchestrator.parse(record, pipeline_id="iot")

    entry = healed[PROVENANCE_KEY]["motor_rpm"]
    assert "clamped_to_max" in entry["actions"]
    assert entry["original"] == pytest.approx(999999.0)
    assert healed["motor_rpm"] == pytest.approx(pipeline.valid_ranges["motor_rpm"][1])


def test_coerced_field_records_the_original_value():
    pipeline, orchestrator = _active_parser("finance")
    record = pipeline.fetch_clean_sample()
    record["amount"] = "42.5"

    healed = orchestrator.parse(record, pipeline_id="finance")

    entry = healed[PROVENANCE_KEY]["amount"]
    assert "coerced_to_number" in entry["actions"]
    assert entry["original"] == "42.5"
    assert healed["amount"] == pytest.approx(42.5)


def test_aliased_field_names_the_key_it_came_from():
    pipeline, orchestrator = _active_parser("ecommerce")
    record = pipeline.fetch_clean_sample()
    original = record["reorder_level"]
    record["min_stock"] = record.pop("reorder_level")

    healed = orchestrator.parse(record, pipeline_id="ecommerce")

    entry = healed[PROVENANCE_KEY]["reorder_level"]
    assert any("aliased_from:min_stock" == action for action in entry["actions"])
    assert healed["reorder_level"] == pytest.approx(original)


def test_untouched_fields_are_absent_from_provenance():
    """Only altered fields appear, so the block stays a useful signal."""
    pipeline, orchestrator = _active_parser("finance")
    record = pipeline.fetch_clean_sample()
    record.pop("amount")

    healed = orchestrator.parse(record, pipeline_id="finance")

    touched = set(healed[PROVENANCE_KEY])
    assert touched == {"amount"}, f"unexpected fields annotated: {touched - {'amount'}}"


def test_duplicate_verdict_is_recorded_on_the_output():
    """The parser cannot know this itself; detection supplies it."""
    pipeline, orchestrator = _active_parser("weather")
    healed = orchestrator.parse(
        pipeline.cached_baseline(), pipeline_id="weather", is_duplicate=True
    )
    assert healed["_is_duplicate"] is True


def test_provenance_survives_the_sandbox_schema_check():
    """The annotation must not make a verified patch fail verification."""
    pipeline, orchestrator = _active_parser("weather")
    record = pipeline.cached_baseline()
    record["temperature_2m"] = "not a number"

    healed = orchestrator.parse(record, pipeline_id="weather")

    for field, declared in pipeline.get_schema_spec().items():
        assert field in healed
        if declared == "float":
            assert isinstance(healed[field], (int, float))
            assert not isinstance(healed[field], bool)


def test_detector_integration_marks_a_real_duplicate_end_to_end():
    pipeline, orchestrator = _active_parser("weather")
    detector = MLAnomalyDetector()
    diagnostician = LLMDiagnosticAgent()
    record = pipeline.cached_baseline()

    detector.observe(record)  # the record arrived earlier in the stream
    detection = detector.detect(
        dict(record),
        expected_fields=pipeline.expected_schema,
        valid_ranges=pipeline.valid_ranges,
    )
    assert detection["anomaly_type"] == "duplicate_record"

    report = diagnostician.diagnose(
        failing_payload=dict(record),
        expected_schema=pipeline.get_schema_spec(),
        layer1_result=detection,
        valid_ranges=pipeline.valid_ranges,
    )
    assert report.anomaly_category == "duplicate_record"
    assert "fingerprint" in report.cause
