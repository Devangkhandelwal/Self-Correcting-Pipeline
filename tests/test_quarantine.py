"""Tests for the per-field missing-value policy.

Backfilling is safe for a measurement but not for an identity or a monetary
amount. A transaction with an empty `transaction_id` and an `amount` of 0.01
passes schema validation, reaches the warehouse, and enters `SUM(amount)` and
`COUNT(*) GROUP BY merchant_id` -- so it is not a repaired record, it is a
corrupted one that now looks valid. That is the failure this project exists to
prevent, so healing must not commit a milder version of it.

A quarantined record keeps a complete, correctly typed shape -- it can still
flow through typed systems -- but carries `_quarantine` naming the fields whose
values were invented or altered.
"""
import pytest

from src.ingestion.pipeline_registry import (
    POLICY_BACKFILL,
    POLICY_QUARANTINE,
    BasePipeline,
    registry,
)
from src.layer2_diagnostician.agent import DiagnosticReport
from src.layer3_patcher.auto_healer import SelfHealingOrchestrator
from src.layer3_patcher.generator import QUARANTINE_KEY


@pytest.fixture(scope="module")
def orchestrator():
    return SelfHealingOrchestrator()


def _activate(orchestrator, pipeline_id):
    """Heal a pipeline so an active parser is bound, and return the pipeline."""
    pipeline = registry.require_pipeline(pipeline_id)
    report = DiagnosticReport(
        cause="test fixture",
        affected_field=next(iter(pipeline.expected_schema)),
        anomaly_category="schema_drift",
        expected_type="float",
        actual_type="missing_renamed",
        suggested_fix="n/a",
    )
    success, _, verification = orchestrator.heal_pipeline(
        diagnostic=report,
        failing_payload=pipeline.fetch_clean_sample(),
        pipeline_id=pipeline_id,
        expected_schema=pipeline.expected_schema,
        valid_ranges=pipeline.valid_ranges,
        known_aliases={a: c for c, a in pipeline.field_renames.items()},
        quarantine_fields=pipeline.quarantine_fields(),
    )
    assert success, verification.get("error_trace", "")[:300]
    return pipeline


# -- policy declaration -----------------------------------------------------

def test_every_pipeline_protects_its_identity_fields():
    """Identity fields must never be silently invented."""
    expected = {
        "weather": {"latitude", "longitude", "time"},
        "finance": {"transaction_id", "user_id", "merchant_id", "amount"},
        "ecommerce": {"item_id", "sku", "unit_price", "stock_quantity"},
        "iot": {"device_id", "timestamp"},
    }
    for pipeline_id, fields in expected.items():
        pipeline = registry.require_pipeline(pipeline_id)
        assert set(pipeline.quarantine_fields()) == fields


def test_unlisted_fields_default_to_backfill():
    pipeline = registry.require_pipeline("finance")
    assert pipeline.policy_for("currency") == POLICY_BACKFILL
    assert pipeline.policy_for("risk_score") == POLICY_BACKFILL
    assert pipeline.policy_for("amount") == POLICY_QUARANTINE


def test_policy_naming_an_unknown_field_is_rejected():
    """A typo in the policy table must fail loudly, not silently do nothing."""
    with pytest.raises(ValueError, match="unknown fields"):
        BasePipeline(
            pipeline_id="probe", name="p", icon="x", description="d",
            expected_schema={"a": float},
            valid_ranges={}, field_renames={},
            missing_value_policy={"does_not_exist": POLICY_QUARANTINE},
        )


def test_invalid_policy_value_is_rejected():
    with pytest.raises(ValueError, match="invalid policies"):
        BasePipeline(
            pipeline_id="probe", name="p", icon="x", description="d",
            expected_schema={"a": float},
            valid_ranges={}, field_renames={},
            missing_value_policy={"a": "do_whatever"},
        )


def test_policy_is_published_in_the_pipeline_description():
    """The API must expose which fields are protected."""
    described = {item["id"]: item for item in registry.list_pipelines()}
    assert "amount" in described["finance"]["quarantine_fields"]
    assert "currency" not in described["finance"]["quarantine_fields"]


# -- invented values are flagged -------------------------------------------

@pytest.mark.parametrize("pipeline_id,field", [
    ("finance", "transaction_id"),
    ("finance", "user_id"),
    ("finance", "merchant_id"),
    ("finance", "amount"),
    ("weather", "latitude"),
    ("weather", "longitude"),
    ("weather", "time"),
    ("ecommerce", "item_id"),
    ("ecommerce", "sku"),
    ("ecommerce", "unit_price"),
    ("ecommerce", "stock_quantity"),
    ("iot", "device_id"),
    ("iot", "timestamp"),
])
def test_missing_protected_field_quarantines_the_record(
    orchestrator, pipeline_id, field
):
    pipeline = _activate(orchestrator, pipeline_id)
    record = dict(pipeline.fetch_clean_sample())
    record.pop(field)

    healed = orchestrator.parse(record, pipeline_id=pipeline_id)

    assert orchestrator.is_quarantined(healed)
    assert field in orchestrator.quarantined_fields(healed)


@pytest.mark.parametrize("pipeline_id,field", [
    ("finance", "currency"),
    ("finance", "risk_score"),
    ("finance", "timestamp"),
    ("weather", "temperature_2m"),
    ("weather", "precipitation"),
    ("ecommerce", "category"),
    ("iot", "battery_pct"),
    ("iot", "motor_rpm"),
])
def test_missing_backfill_field_does_not_quarantine(
    orchestrator, pipeline_id, field
):
    """A dropped sensor reading is an ordinary event, not a tainted record."""
    pipeline = _activate(orchestrator, pipeline_id)
    record = dict(pipeline.fetch_clean_sample())
    record.pop(field)

    healed = orchestrator.parse(record, pipeline_id=pipeline_id)

    assert not orchestrator.is_quarantined(healed)
    # It is still disclosed as healed, just not quarantined.
    assert field in healed["_healing"]


def test_clamping_a_protected_field_quarantines_it():
    """An out-of-range amount clamped to the ceiling would skew SUM()."""
    orchestrator_local = SelfHealingOrchestrator()
    pipeline = _activate(orchestrator_local, "finance")
    record = dict(pipeline.fetch_clean_sample())
    record["amount"] = 999999.0  # range is (0.01, 50000.0)

    healed = orchestrator_local.parse(record, pipeline_id="finance")

    assert "amount" in orchestrator_local.quarantined_fields(healed)
    assert healed["amount"] == pytest.approx(50000.0)


def test_coercion_does_not_quarantine(orchestrator):
    """'42.5' -> 42.5 preserves the value; nothing was invented."""
    pipeline = _activate(orchestrator, "finance")
    record = dict(pipeline.fetch_clean_sample())
    record["amount"] = "42.5"

    healed = orchestrator.parse(record, pipeline_id="finance")

    assert not orchestrator.is_quarantined(healed)
    assert healed["amount"] == pytest.approx(42.5)
    assert "coerced_to_number" in healed["_healing"]["amount"]["actions"]


def test_clean_record_is_not_quarantined(orchestrator):
    pipeline = _activate(orchestrator, "finance")
    healed = orchestrator.parse(pipeline.fetch_clean_sample(), pipeline_id="finance")
    assert QUARANTINE_KEY not in healed
    assert not orchestrator.is_quarantined(healed)


# -- the shape contract -----------------------------------------------------

def test_quarantined_record_keeps_a_valid_shape(orchestrator):
    """It must still flow through typed systems; it just must not be trusted."""
    pipeline = _activate(orchestrator, "finance")
    record = dict(pipeline.fetch_clean_sample())
    record.pop("amount")
    record.pop("transaction_id")

    healed = orchestrator.parse(record, pipeline_id="finance")

    assert set(orchestrator.quarantined_fields(healed)) == {"amount", "transaction_id"}
    for field, declared in pipeline.get_schema_spec().items():
        assert field in healed
        if declared == "float":
            assert isinstance(healed[field], (int, float))
            assert not isinstance(healed[field], bool)
        else:
            assert isinstance(healed[field], str)


def test_quarantine_flag_does_not_change_record_identity():
    """Annotating a record must not make it look like different data."""
    from src.ingestion.dedup import fingerprint

    record = {"amount": 1.0, "currency": "USD"}
    annotated = dict(record, _quarantine=["amount"], _healing={"amount": {}})
    assert fingerprint(annotated) == fingerprint(record)


def test_healing_still_succeeds_for_a_quarantined_record(orchestrator):
    """Quarantine is an outcome of healing, not a failure of it."""
    pipeline = registry.require_pipeline("finance")
    record = dict(pipeline.fetch_clean_sample())
    record.pop("amount")

    report = DiagnosticReport(
        cause="missing amount", affected_field="amount",
        anomaly_category="null_field", expected_type="float",
        actual_type="absent", suggested_fix="backfill",
    )
    success, _, verification = orchestrator.heal_pipeline(
        diagnostic=report, failing_payload=record, pipeline_id="finance",
        expected_schema=pipeline.expected_schema,
        valid_ranges=pipeline.valid_ranges,
        known_aliases={a: c for c, a in pipeline.field_renames.items()},
        quarantine_fields=pipeline.quarantine_fields(),
    )
    assert success is True
    assert verification["passed"] is True
