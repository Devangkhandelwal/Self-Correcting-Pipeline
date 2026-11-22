"""REST API for the self-healing pipeline agent.

Security and correctness fixes over the previous version:
  * no more `debug=True` on `0.0.0.0`, which exposed the Werkzeug console
    (remote code execution) to anything that could reach the port; host,
    port and debug now come from configuration and default to loopback with
    the debugger off;
  * CORS is restricted to configured origins instead of every origin;
  * `pipeline_id` and `anomaly_type` are validated and rejected with 400
    rather than silently no-opping or falling back to the weather pipeline;
  * `/api/evaluate` is cached, so a public GET no longer re-runs a full
    1200-record evaluation on every request;
  * the app is built by a factory so tests can drive it without binding a port.
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Any, Dict, Optional, Tuple

from flask import Flask, jsonify, request
from flask_cors import CORS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ingestion.dedup import DuplicateTracker
from src.ingestion.injector import ANOMALY_CLASSES, SyntheticAnomalyInjector
from src.ingestion.pipeline_registry import registry
from src.layer1_detector.evaluator import evaluate_layer1
from src.layer2_diagnostician.agent import LLMDiagnosticAgent
from src.layer3_patcher.auto_healer import SelfHealingOrchestrator
from src.layer1_detector.model import MLAnomalyDetector
from src.utils import config
from src.utils.logger import get_recent_audit_logs, logger


def create_app() -> Flask:
    """Build and configure the Flask application."""
    app = Flask(__name__)
    CORS(app, origins=config.CORS_ORIGINS)

    # Components are built once; each is cheap to share and holds no
    # per-request state. The orchestrator's parser table is process-wide by
    # design -- that is what "hot-swapped parser" means here.
    injector = SyntheticAnomalyInjector()
    detector = MLAnomalyDetector()
    diagnostician = LLMDiagnosticAgent()
    orchestrator = SelfHealingOrchestrator()

    # Memoised evaluation: the underlying dataset only changes when the
    # training pipeline is re-run, so there is no reason to recompute it per
    # request. Guarded by a lock so concurrent callers do not stampede.
    evaluation_cache: Dict[str, Any] = {}
    evaluation_lock = threading.Lock()

    def _json_body() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Parse a JSON request body without raising on malformed input."""
        body = request.get_json(silent=True)
        if body is None:
            return None, "Request body must be valid JSON."
        if not isinstance(body, dict):
            return None, "Request body must be a JSON object."
        return body, None

    def _validate_pipeline_id(pipeline_id: Any) -> Tuple[Optional[Any], Optional[str]]:
        if not isinstance(pipeline_id, str) or not pipeline_id:
            return None, "pipeline_id must be a non-empty string."
        pipeline = registry.get_pipeline(pipeline_id)
        if pipeline is None:
            return None, (
                f"Unknown pipeline_id '{pipeline_id}'. "
                f"Valid ids: {', '.join(registry.pipeline_ids())}."
            )
        return pipeline, None

    def _validate_anomaly_type(anomaly_type: Any) -> Tuple[Optional[str], Optional[str]]:
        if not isinstance(anomaly_type, str) or not anomaly_type:
            return None, "anomaly_type must be a non-empty string."
        if anomaly_type not in ANOMALY_CLASSES:
            return None, (
                f"Unknown anomaly_type '{anomaly_type}'. "
                f"Valid types: {', '.join(ANOMALY_CLASSES)}."
            )
        return anomaly_type, None

    # -- routes --------------------------------------------------------------

    @app.get("/api/health")
    def health():
        """Readiness probe reporting which optional components are available."""
        return jsonify({
            "status": "ok",
            "model_loaded": detector.is_trained,
            "llm_enabled": diagnostician.llm_enabled,
            "pipelines": registry.pipeline_ids(),
            "active_parsers": sorted(orchestrator.active_parsers),
        })

    @app.get("/api/pipelines")
    def get_pipelines():
        return jsonify({"pipelines": registry.list_pipelines()})

    @app.get("/api/anomaly-classes")
    def get_anomaly_classes():
        return jsonify({"classes": list(ANOMALY_CLASSES)})

    @app.get("/api/schema")
    def get_schema():
        pipeline, error = _validate_pipeline_id(
            request.args.get("pipeline_id", registry.DEFAULT_PIPELINE_ID)
        )
        if error:
            return jsonify({"error": error}), 400
        return jsonify({
            "pipeline_id": pipeline.pipeline_id,
            "schema": pipeline.get_schema_spec(),
            "valid_ranges": {k: list(v) for k, v in pipeline.valid_ranges.items()},
        })

    @app.post("/api/run-pipeline")
    def run_pipeline():
        """Full loop: fetch -> inject -> detect -> diagnose -> heal."""
        body, error = _json_body()
        if error:
            return jsonify({"error": error}), 400

        pipeline, error = _validate_pipeline_id(
            body.get("pipeline_id", registry.DEFAULT_PIPELINE_ID)
        )
        if error:
            return jsonify({"error": error}), 400

        anomaly_type, error = _validate_anomaly_type(
            body.get("anomaly_type", "schema_drift")
        )
        if error:
            return jsonify({"error": error}), 400

        try:
            clean_payload = pipeline.fetch_clean_sample()
            corrupted, label, meta = injector.inject_anomaly(
                clean_payload,
                target_anomaly=anomaly_type,
                field_renames=pipeline.field_renames,
                valid_ranges=pipeline.valid_ranges,
            )

            # Each request simulates one isolated stream event, so it gets
            # its own duplicate tracker. Sharing the detector's long-lived
            # tracker across requests meant a second `normal` weather run
            # was reported as a duplicate -- correct for a real stream,
            # since the upstream returns the same observation for an hour,
            # but wrong for an endpoint whose requests are independent.
            request_tracker = DuplicateTracker()
            if label == "duplicate_record":
                # By definition this content arrived earlier in the stream.
                request_tracker.observe(corrupted)

            detection = detector.detect(
                corrupted,
                expected_fields=pipeline.expected_schema,
                valid_ranges=pipeline.valid_ranges,
                duplicate_tracker=request_tracker,
            )
        except Exception as exc:
            logger.exception("Pipeline run failed during ingestion/detection")
            return jsonify({"error": f"Pipeline run failed: {exc}"}), 500

        result: Dict[str, Any] = {
            "pipeline_info": {
                "id": pipeline.pipeline_id,
                "name": pipeline.name,
                "icon": pipeline.icon,
                "data_source": pipeline.last_fetch_source,
            },
            "clean_payload": clean_payload,
            "corrupted_payload": corrupted,
            "injected_anomaly": label,
            "injection_meta": meta,
            "layer1_detection": detection,
            "layer2_diagnosis": None,
            "diagnosis_source": None,
            "layer3_patch": None,
            "layer3_verification": None,
            "healed_payload": None,
            "heal_success": None,
        }

        is_duplicate = bool(
            detection.get("features", {}).get("duplicate_flag", 0.0)
        )
        result["is_duplicate"] = is_duplicate

        if not detection["is_anomaly"]:
            return jsonify(result)

        known_aliases = {
            alias: canonical for canonical, alias in pipeline.field_renames.items()
        }

        try:
            diagnosis = diagnostician.diagnose(
                failing_payload=corrupted,
                expected_schema=pipeline.get_schema_spec(),
                layer1_result=detection,
                valid_ranges=pipeline.valid_ranges,
                known_aliases=known_aliases,
            )
            result["layer2_diagnosis"] = diagnosis.model_dump()
            result["diagnosis_source"] = "gemini" if diagnostician.llm_enabled else "deterministic"

            success, patch_code, verification = orchestrator.heal_pipeline(
                diagnostic=diagnosis,
                failing_payload=corrupted,
                pipeline_id=pipeline.pipeline_id,
                expected_schema=pipeline.expected_schema,
                valid_ranges=pipeline.valid_ranges,
                known_aliases=known_aliases,
            )
            result["layer3_patch"] = patch_code
            result["layer3_verification"] = verification
            result["heal_success"] = success

            if success:
                result["healed_payload"] = orchestrator.parse(
                    corrupted,
                    pipeline_id=pipeline.pipeline_id,
                    is_duplicate=is_duplicate,
                )
        except Exception as exc:
            logger.exception("Pipeline run failed during diagnosis/healing")
            return jsonify({"error": f"Diagnosis or healing failed: {exc}"}), 500

        return jsonify(result)

    @app.get("/api/evaluate")
    def run_evaluation():
        """Layer 1 benchmark metrics, computed once and cached."""
        refresh = request.args.get("refresh", "").lower() in {"1", "true", "yes"}
        with evaluation_lock:
            if refresh or "metrics" not in evaluation_cache:
                try:
                    evaluation_cache["metrics"] = evaluate_layer1()
                except Exception as exc:
                    logger.exception("Layer 1 evaluation failed")
                    return jsonify({"error": f"Evaluation failed: {exc}"}), 500
            metrics = evaluation_cache["metrics"]

        if not metrics:
            return jsonify({
                "error": "No evaluation data available. Run run_data_and_train.py "
                         "to generate the dataset and train the model."
            }), 503
        return jsonify(metrics)

    @app.get("/api/audit-logs")
    def get_audit_logs():
        raw_limit = request.args.get("limit", "30")
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            limit = 30
        logs = get_recent_audit_logs(limit=limit)
        return jsonify({"logs": logs, "limit": min(max(limit, 1), config.AUDIT_LOG_MAX_LIMIT)})

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({"error": "Endpoint not found."}), 404

    @app.errorhandler(500)
    def server_error(_error):  # pragma: no cover - defensive
        return jsonify({"error": "Internal server error."}), 500

    return app


app = create_app()


if __name__ == "__main__":
    if config.API_DEBUG:
        logger.warning(
            "API_DEBUG is enabled: the Werkzeug debugger allows arbitrary code "
            "execution. Never enable this on a reachable interface."
        )
    logger.info("Starting API on http://%s:%d", config.API_HOST, config.API_PORT)
    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=config.API_DEBUG,
        use_reloader=False,
    )
