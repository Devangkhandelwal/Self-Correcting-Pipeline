"""Regression tests for outlier diagnosis and clamping.

Before the fix, the Layer 2 fallback engine only recognised outliers in the
hardcoded field `temperature_2m`. Every other field fell through to a generic
report with affected_field="general", which the generator turned into
`if 'general' in parsed:` -- a branch that can never be true. The outlier
passed through untouched while heal_success reported True.
"""
import pytest

from src.ingestion.pipeline_registry import registry
from src.layer1_detector.model import MLAnomalyDetector
from src.layer2_diagnostician.agent import LLMDiagnosticAgent
from src.layer3_patcher.auto_healer import SelfHealingOrchestrator

# (pipeline_id, field, injected out-of-range value)
OUTLIER_CASES = [
    ("weather", "surface_pressure", 12031.0),     # valid range (800, 1200)
    ("weather", "relative_humidity_2m", 9999.0),  # valid range (0, 100)
    ("weather", "wind_speed_10m", 30456.0),       # valid range (0, 300)
    ("iot", "motor_rpm", 999999.0),               # valid range (0, 6000)
    ("iot", "vibration_rms", 4210.0),             # valid range (0, 15)
    ("finance", "risk_score", 870.5),             # valid range (0, 1)
]


def _diagnose(pipeline_id, field, bad_value):
    pipeline = registry.get_pipeline(pipeline_id)
    corrupted = dict(pipeline.fetch_clean_sample())
    corrupted[field] = bad_value

    detection = MLAnomalyDetector().detect(
        corrupted,
        expected_fields=pipeline.expected_schema,
        valid_ranges=pipeline.valid_ranges,
    )
    report = LLMDiagnosticAgent().diagnose(
        failing_payload=corrupted,
        expected_schema=pipeline.get_schema_spec(),
        layer1_result=detection,
        valid_ranges=pipeline.valid_ranges,
    )
    return pipeline, corrupted, report


@pytest.mark.parametrize("pipeline_id,field,bad_value", OUTLIER_CASES)
def test_outlier_is_diagnosed_on_the_correct_field(pipeline_id, field, bad_value):
    """Layer 2 must name the offending field, not fall back to a placeholder."""
    _, _, report = _diagnose(pipeline_id, field, bad_value)

    assert report.anomaly_category == "outlier"
    assert report.affected_field == field, (
        "diagnosed '%s' instead of '%s'" % (report.affected_field, field)
    )


@pytest.mark.parametrize("pipeline_id,field,bad_value", OUTLIER_CASES)
def test_outlier_is_clamped_into_range_by_the_patch(pipeline_id, field, bad_value):
    """The healed value must land inside the declared valid range."""
    pipeline, corrupted, report = _diagnose(pipeline_id, field, bad_value)

    orch = SelfHealingOrchestrator()
    success, _, _ = orch.heal_pipeline(
        diagnostic=report, failing_payload=corrupted, pipeline_id=pipeline_id
    )
    healed = orch.parse(dict(corrupted), pipeline_id=pipeline_id)

    assert success is True
    low, high = pipeline.valid_ranges[field]
    assert low <= healed[field] <= high, (
        "%s was %s, healed to %s, outside declared range (%s, %s)"
        % (field, bad_value, healed[field], low, high)
    )
