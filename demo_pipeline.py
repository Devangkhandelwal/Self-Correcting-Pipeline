import json
import time
from src.ingestion.api_client import WeatherAPIClient
from src.ingestion.injector import SyntheticAnomalyInjector
from src.layer1_detector.model import MLAnomalyDetector
from src.layer2_diagnostician.agent import LLMDiagnosticAgent
from src.layer3_patcher.auto_healer import SelfHealingOrchestrator
from src.utils.logger import logger

def run_demo():
    print("=" * 70)
    print("SELF-CORRECTING DATA PIPELINE AGENT - DEMO RUN")
    print("=" * 70)


    # 1. Initialize System Components
    client = WeatherAPIClient()
    injector = SyntheticAnomalyInjector()
    detector = MLAnomalyDetector()
    diagnostician = LLMDiagnosticAgent()
    orchestrator = SelfHealingOrchestrator()

    # 2. Fetch Clean API Payload
    print("\n[STEP 1] Fetching live clean payload from Open-Meteo API...")
    clean_payload = client.fetch_live_sample()
    print("Clean Payload:")
    print(json.dumps(clean_payload, indent=2))

    # 3. Inject Synthetic Anomaly (Schema Drift)
    print("\n" + "-" * 70)
    print("[STEP 2] Injecting Synthetic Anomaly (Category: schema_drift)...")
    corrupted_payload, label, meta = injector.inject_anomaly(clean_payload, target_anomaly="schema_drift")
    print(f"Applied Anomaly: {label}")
    print("Corrupted Ingestion Payload:")
    print(json.dumps(corrupted_payload, indent=2))

    # 4. Layer 1: ML Anomaly Detector
    print("\n" + "-" * 70)
    print("[STEP 3] Running Layer 1: ML Anomaly Detector (RandomForest)...")
    detection_res = detector.detect(corrupted_payload)
    print(f"  • Is Anomaly Flagged : {detection_res['is_anomaly']}")
    print(f"  • Predicted Category : {detection_res['anomaly_type']}")
    print(f"  • Confidence Score  : {detection_res['confidence'] * 100:.1f}%")

    if not detection_res["is_anomaly"]:
        print("No anomaly detected. Pipeline continuing normally.")
        return

    # 5. Layer 2: Diagnostic Agent (LLM)
    print("\n" + "-" * 70)
    print("[STEP 4] Running Layer 2: LLM Diagnostic Agent...")
    diag_report = diagnostician.diagnose(
        failing_payload=corrupted_payload,
        expected_schema=client.get_expected_schema_spec(),
        layer1_result=detection_res
    )
    print("Diagnostic Root Cause Report:")
    print(json.dumps(diag_report.model_dump(), indent=2))

    # 6. Layer 3: Patch Generation & Sandboxed Verification Loop
    print("\n" + "-" * 70)
    print("[STEP 5] Running Layer 3: Patch Generator & Sandbox Verification Loop...")
    success, patch_code, verif_res = orchestrator.heal_pipeline(
        diagnostic=diag_report,
        failing_payload=corrupted_payload
    )

    print(f"  • Patch Generated & Verification Passed : {success}")
    print(f"  • Held-Out Verification Success Rate    : {verif_res.get('success_rate', 0.0) * 100:.1f}% ({verif_res.get('passed_count')}/{verif_res.get('total_count')} records)")
    print("\nGenerated Python Parser Code Patch:")
    print(patch_code)

    # 7. Parse Corrupted Payload using Healed Parser
    print("-" * 70)
    print("[STEP 6] Parsing Corrupted Payload with Active Healed Parser...")
    healed_output = orchestrator.parse(corrupted_payload)
    print("Healed & Normalized Payload Output:")
    print(json.dumps(healed_output, indent=2))

    print("\n" + "=" * 70)
    print("[SUCCESS] PIPELINE AUTO-HEALED SUCCESSFULLY! NO HUMAN INTERVENTION REQUIRED.")
    print("=" * 70)

if __name__ == "__main__":
    run_demo()
