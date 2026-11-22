"""Regression tests for cross-domain self-healing.

These pin the contract that Layer 3 must honour the *selected pipeline's*
schema rather than a hardcoded weather schema. Before the schema-threading
fix, every one of these failed: the generated parser emitted eight
zero-filled weather fields and discarded the real payload entirely, while
still reporting heal_success=True at a 100% verification rate.

Drift cases are enumerated per (pipeline, field) rather than left to the
injector's RNG. An earlier version of this file let the injector choose, and
it masked a real defect: `category` -> `item_cat` and
`reorder_level` -> `min_stock` were never recovered, but the test only hit
them when the random draw happened to pick those fields.
"""
import pytest

from src.ingestion.injector import SyntheticAnomalyInjector
from src.ingestion.pipeline_registry import registry
from src.layer1_detector.model import MLAnomalyDetector
from src.layer2_diagnostician.agent import LLMDiagnosticAgent
from src.layer3_patcher.auto_healer import SelfHealingOrchestrator

NON_WEATHER = ["finance", "ecommerce", "iot"]

WEATHER_ONLY_FIELDS = {
    "latitude", "longitude", "temperature_2m", "relative_humidity_2m",
    "precipitation", "surface_pressure", "wind_speed_10m",
}


def _all_drift_cases():
    """Every (pipeline_id, canonical_field, alias) the registry declares."""
    cases = []
    for pipeline_id in registry.pipeline_ids():
        pipeline = registry.require_pipeline(pipeline_id)
        for canonical, alias in pipeline.field_renames.items():
            cases.append(pytest.param(
                pipeline_id, canonical, alias,
                id=f"{pipeline_id}-{canonical}-to-{alias}",
            ))
    return cases


def _heal(pipeline_id, corrupted):
    """Run detect -> diagnose -> heal for an already-corrupted payload."""
    pipeline = registry.require_pipeline(pipeline_id)
    known_aliases = {
        alias: canonical for canonical, alias in pipeline.field_renames.items()
    }

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
        known_aliases=known_aliases,
    )
    orchestrator = SelfHealingOrchestrator()
    success, _, verification = orchestrator.heal_pipeline(
        diagnostic=report,
        failing_payload=corrupted,
        pipeline_id=pipeline_id,
        expected_schema=pipeline.expected_schema,
        valid_ranges=pipeline.valid_ranges,
        known_aliases=known_aliases,
    )
    healed = orchestrator.parse(dict(corrupted), pipeline_id=pipeline_id)
    return pipeline, healed, success, verification


def _assert_same_value(field, expected, actual):
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert actual == pytest.approx(expected), (
            f"field '{field}' was {expected!r}, healed to {actual!r}"
        )
    else:
        assert str(actual) == str(expected), (
            f"field '{field}' was {expected!r}, healed to {actual!r}"
        )


@pytest.mark.parametrize("pipeline_id,canonical,alias", _all_drift_cases())
def test_declared_rename_is_recovered_with_its_value(pipeline_id, canonical, alias):
    """Every rename the registry declares must be recoverable, value intact.

    This is the test that catches semantic renames invisible to string
    similarity, e.g. reorder_level -> min_stock (0.09 similarity).
    """
    pipeline = registry.require_pipeline(pipeline_id)
    clean = pipeline.fetch_clean_sample()
    corrupted = dict(clean)
    corrupted[alias] = corrupted.pop(canonical)

    _, healed, success, _ = _heal(pipeline_id, corrupted)

    assert success is True
    assert canonical in healed, f"'{canonical}' absent from healed record"
    _assert_same_value(canonical, clean[canonical], healed[canonical])


@pytest.mark.parametrize("pipeline_id", NON_WEATHER)
def test_healed_output_matches_pipeline_schema(pipeline_id):
    """The healed record must carry the pipeline's own fields, not weather's."""
    pipeline = registry.require_pipeline(pipeline_id)
    clean = pipeline.fetch_clean_sample()
    corrupted, _, _ = SyntheticAnomalyInjector(
        rng=__import__("random").Random(1234)
    ).inject_anomaly(
        clean,
        target_anomaly="schema_drift",
        field_renames=pipeline.field_renames,
        valid_ranges=pipeline.valid_ranges,
    )

    _, healed, success, _ = _heal(pipeline_id, corrupted)

    assert success is True
    for field in pipeline.expected_schema:
        assert field in healed, f"{pipeline_id}: healed output lost field '{field}'"

    leaked = WEATHER_ONLY_FIELDS & set(healed)
    assert not leaked, f"{pipeline_id}: weather fields leaked into output: {leaked}"


@pytest.mark.parametrize("pipeline_id,canonical,alias", _all_drift_cases())
def test_healing_preserves_every_uncorrupted_field(pipeline_id, canonical, alias):
    """Healing one field must not blank out or alter any other field."""
    pipeline = registry.require_pipeline(pipeline_id)
    clean = pipeline.fetch_clean_sample()
    corrupted = dict(clean)
    corrupted[alias] = corrupted.pop(canonical)

    _, healed, _, _ = _heal(pipeline_id, corrupted)

    for field, expected in clean.items():
        if field == canonical:
            continue  # recovered via alias; asserted separately
        _assert_same_value(field, expected, healed[field])


@pytest.mark.parametrize("pipeline_id", registry.pipeline_ids())
@pytest.mark.parametrize(
    "anomaly",
    ["type_mismatch", "null_field", "outlier", "duplicate_record"],
)
def test_every_anomaly_class_heals_on_every_pipeline(pipeline_id, anomaly):
    """The full anomaly taxonomy must heal across all four domains."""
    import random

    pipeline = registry.require_pipeline(pipeline_id)
    clean = pipeline.fetch_clean_sample()
    corrupted, _, _ = SyntheticAnomalyInjector(rng=random.Random(7)).inject_anomaly(
        clean,
        target_anomaly=anomaly,
        field_renames=pipeline.field_renames,
        valid_ranges=pipeline.valid_ranges,
    )

    _, healed, success, verification = _heal(pipeline_id, corrupted)

    assert success is True, (
        f"{pipeline_id}/{anomaly} failed to heal: "
        f"{verification.get('error_trace', '')[:400]}"
    )
    for field, declared in pipeline.get_schema_spec().items():
        assert field in healed
        if declared == "float":
            assert isinstance(healed[field], (int, float))
            assert not isinstance(healed[field], bool)
        elif declared == "str":
            assert isinstance(healed[field], str)
