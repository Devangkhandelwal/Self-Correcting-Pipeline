import os
import sys
import json
from flask import Flask, jsonify, request
from flask_cors import CORS

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ingestion.pipeline_registry import registry
from src.ingestion.injector import SyntheticAnomalyInjector, ANOMALY_CLASSES
from src.layer1_detector.model import MLAnomalyDetector
from src.layer1_detector.evaluator import evaluate_layer1
from src.layer2_diagnostician.agent import LLMDiagnosticAgent
from src.layer3_patcher.auto_healer import SelfHealingOrchestrator
from src.utils.logger import get_recent_audit_logs

app = Flask(__name__)
CORS(app)

# Initialize pipeline components once at startup
injector = SyntheticAnomalyInjector()
detector = MLAnomalyDetector()
diagnostician = LLMDiagnosticAgent()
orchestrator = SelfHealingOrchestrator()


@app.route("/api/pipelines", methods=["GET"])
def get_pipelines():
    return jsonify({"pipelines": registry.list_pipelines()})


@app.route("/api/anomaly-classes", methods=["GET"])
def get_anomaly_classes():
    return jsonify({"classes": ANOMALY_CLASSES})


@app.route("/api/schema", methods=["GET"])
def get_schema():
    pipeline_id = request.args.get("pipeline_id", "weather")
    pipeline = registry.get_pipeline(pipeline_id)
    return jsonify({"schema": pipeline.get_schema_spec()})


@app.route("/api/run-pipeline", methods=["POST"])
def run_pipeline():
    """Full pipeline: fetch -> inject -> detect -> diagnose -> heal."""
    data = request.get_json() or {}
    anomaly_type = data.get("anomaly_type", "schema_drift")
    pipeline_id = data.get("pipeline_id", "weather")

    pipeline = registry.get_pipeline(pipeline_id)

    # Step 1: Fetch clean payload for selected pipeline
    clean_payload = pipeline.fetch_clean_sample()

    # Step 2: Inject anomaly using pipeline's rename candidates & valid ranges
    corrupted, label, meta = injector.inject_anomaly(
        clean_payload,
        target_anomaly=anomaly_type,
        field_renames=pipeline.field_renames,
        valid_ranges=pipeline.valid_ranges
    )

    # Step 3: Layer 1 - ML Detection
    detection = detector.detect(
        corrupted,
        expected_fields=pipeline.expected_schema,
        valid_ranges=pipeline.valid_ranges
    )

    result = {
        "pipeline_info": {
            "id": pipeline.pipeline_id,
            "name": pipeline.name,
            "icon": pipeline.icon,
        },
        "clean_payload": clean_payload,
        "corrupted_payload": corrupted,
        "injected_anomaly": label,
        "injection_meta": meta,
        "layer1_detection": detection,
        "layer2_diagnosis": None,
        "layer3_patch": None,
        "layer3_verification": None,
        "healed_payload": None,
        "heal_success": None,
    }

    # Step 4 & 5: If anomaly detected, run Layer 2 & 3
    if detection["is_anomaly"]:
        diag = diagnostician.diagnose(
            failing_payload=corrupted,
            expected_schema=pipeline.get_schema_spec(),
            layer1_result=detection,
        )
        result["layer2_diagnosis"] = diag.model_dump()

        success, patch_code, verif = orchestrator.heal_pipeline(
            diagnostic=diag,
            failing_payload=corrupted,
            pipeline_id=pipeline.pipeline_id
        )
        result["layer3_patch"] = patch_code
        result["layer3_verification"] = verif
        result["heal_success"] = success

        if success:
            result["healed_payload"] = orchestrator.parse(dict(corrupted), pipeline_id=pipeline.pipeline_id)

    return jsonify(result)


@app.route("/api/evaluate", methods=["GET"])
def run_evaluation():
    """Evaluate Layer 1 ML model metrics."""
    metrics = evaluate_layer1()
    return jsonify(metrics)


@app.route("/api/audit-logs", methods=["GET"])
def get_audit_logs():
    limit = request.args.get("limit", 30, type=int)
    logs = get_recent_audit_logs(limit=limit)
    return jsonify({"logs": logs})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
