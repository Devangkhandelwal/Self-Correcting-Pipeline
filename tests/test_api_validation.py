"""Request-validation tests for the REST layer.

Before the fix, /api/run-pipeline forwarded unvalidated input: an unknown
anomaly_type silently no-opped yet was echoed back as `injected_anomaly`,
and an unknown pipeline_id silently resolved to weather -- so a typo
returned confident weather results for a finance request.
"""
import pytest

from api.server import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def test_unknown_anomaly_type_is_rejected(client):
    res = client.post(
        "/api/run-pipeline",
        json={"pipeline_id": "weather", "anomaly_type": "not_a_real_anomaly"},
    )
    assert res.status_code == 400
    assert "anomaly_type" in res.get_json().get("error", "")


def test_unknown_pipeline_id_is_rejected(client):
    res = client.post(
        "/api/run-pipeline",
        json={"pipeline_id": "nonexistent_pipeline", "anomaly_type": "schema_drift"},
    )
    assert res.status_code == 400
    assert "pipeline_id" in res.get_json().get("error", "")


def test_unknown_pipeline_id_on_schema_endpoint_is_rejected(client):
    assert client.get("/api/schema?pipeline_id=nope").status_code == 400


def test_valid_request_succeeds_and_reports_the_right_pipeline(client):
    res = client.post(
        "/api/run-pipeline",
        json={"pipeline_id": "finance", "anomaly_type": "schema_drift"},
    )
    assert res.status_code == 200
    body = res.get_json()
    assert body["pipeline_info"]["id"] == "finance"
    assert body["injected_anomaly"] == "schema_drift"


def test_non_json_body_is_rejected_not_crashed(client):
    res = client.post(
        "/api/run-pipeline", data="not json", content_type="text/plain"
    )
    assert res.status_code == 400


def test_audit_log_limit_is_bounded(client):
    """A caller must not be able to request an unbounded log dump."""
    res = client.get("/api/audit-logs?limit=999999")
    assert res.status_code == 200
    assert len(res.get_json()["logs"]) <= 500


def test_malformed_limit_does_not_crash(client):
    assert client.get("/api/audit-logs?limit=abc").status_code in (200, 400)


def test_health_endpoint_reports_component_state(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.get_json()
    assert "status" in body
    assert "model_loaded" in body


def test_repeated_normal_requests_are_not_reported_as_duplicates(client):
    """Each request simulates an isolated stream event.

    The weather upstream returns the same observation for an hour, so sharing
    a long-lived duplicate tracker across requests made a second `normal` run
    report `duplicate_record`.
    """
    for _ in range(3):
        body = client.post(
            "/api/run-pipeline",
            json={"pipeline_id": "weather", "anomaly_type": "normal"},
        ).get_json()
        assert body["layer1_detection"]["anomaly_type"] == "normal"
        assert body["is_duplicate"] is False


def test_duplicate_request_is_detected_on_the_very_first_call(client):
    """Must not depend on a previous request having populated history."""
    body = client.post(
        "/api/run-pipeline",
        json={"pipeline_id": "iot", "anomaly_type": "duplicate_record"},
    ).get_json()
    assert body["layer1_detection"]["anomaly_type"] == "duplicate_record"
    assert body["is_duplicate"] is True
    assert body["healed_payload"]["_is_duplicate"] is True
