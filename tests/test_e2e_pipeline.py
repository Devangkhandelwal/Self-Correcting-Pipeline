import pytest
from src.ingestion.api_client import WeatherAPIClient
from src.ingestion.injector import SyntheticAnomalyInjector
from src.layer1_detector.model import MLAnomalyDetector
from src.layer2_diagnostician.agent import LLMDiagnosticAgent
from src.layer3_patcher.auto_healer import SelfHealingOrchestrator

def test_end_to_end_self_healing_pipeline():
    client = WeatherAPIClient()
    injector = SyntheticAnomalyInjector()
    detector = MLAnomalyDetector()
    diagnostician = LLMDiagnosticAgent()
    orchestrator = SelfHealingOrchestrator()

    # 1. Fetch clean payload & inject schema_drift anomaly
    clean = client.fetch_live_sample()
    corrupted, label, meta = injector.inject_anomaly(clean, target_anomaly="schema_drift")

    # 2. Layer 1 Anomaly Detection
    det_res = detector.detect(corrupted)
    assert det_res["is_anomaly"] == True

    # 3. Layer 2 LLM Diagnostic Agent
    diag_report = diagnostician.diagnose(
        failing_payload=corrupted,
        expected_schema=client.get_expected_schema_spec(),
        layer1_result=det_res
    )
    if "renamed_from" in meta:
        assert diag_report.affected_field == meta["renamed_from"]
    else:
        assert diag_report.affected_field in client.get_expected_schema_spec()

    # 4. Layer 3 Patch Generation & Sandbox Verification
    success, patch_code, verif_res = orchestrator.heal_pipeline(
        diagnostic=diag_report,
        failing_payload=corrupted
    )
    assert success == True
    assert verif_res["passed"] == True

    # 5. Parse corrupted record using healed parser
    healed_output = orchestrator.parse(corrupted)
    affected = diag_report.affected_field
    assert affected in healed_output
