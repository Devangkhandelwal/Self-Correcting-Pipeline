"""CLI walkthrough of the full detect -> diagnose -> heal loop.

Accepts --pipeline and --anomaly so the demo covers all four domains and the
whole anomaly taxonomy, rather than only weather schema drift. Exits non-zero
when healing fails, so it is usable as a smoke check in CI.
"""
from __future__ import annotations

import argparse
import json
import sys

from src.ingestion.injector import ANOMALY_CLASSES, SyntheticAnomalyInjector
from src.ingestion.pipeline_registry import registry
from src.layer1_detector.model import MLAnomalyDetector
from src.layer2_diagnostician.agent import LLMDiagnosticAgent
from src.layer3_patcher.auto_healer import SelfHealingOrchestrator

RULE = "=" * 72
THIN = "-" * 72


def run_demo(pipeline_id: str = "weather", anomaly: str = "schema_drift") -> bool:
    """Run one end-to-end healing cycle; returns True when the pipeline healed."""
    print(RULE)
    print("SELF-CORRECTING DATA PIPELINE AGENT - DEMO RUN")
    print(RULE)

    pipeline = registry.require_pipeline(pipeline_id)
    known_aliases = {
        alias: canonical for canonical, alias in pipeline.field_renames.items()
    }

    injector = SyntheticAnomalyInjector()
    detector = MLAnomalyDetector()
    diagnostician = LLMDiagnosticAgent()
    orchestrator = SelfHealingOrchestrator()

    print(f"\n[STEP 1] Fetching a clean payload from {pipeline.name}...")
    clean_payload = pipeline.fetch_clean_sample()
    print(f"Data source: {pipeline.last_fetch_source}")
    print(json.dumps(clean_payload, indent=2))

    print("\n" + THIN)
    print(f"[STEP 2] Injecting synthetic anomaly (category: {anomaly})...")
    corrupted, label, meta = injector.inject_anomaly(
        clean_payload,
        target_anomaly=anomaly,
        field_renames=pipeline.field_renames,
        valid_ranges=pipeline.valid_ranges,
    )
    print(f"Applied anomaly: {label}")
    print(json.dumps(corrupted, indent=2))

    print("\n" + THIN)
    print("[STEP 3] Layer 1: ML anomaly detector...")
    detection = detector.detect(
        corrupted,
        expected_fields=pipeline.expected_schema,
        valid_ranges=pipeline.valid_ranges,
    )
    print(f"  - Engine             : {detection.get('engine', 'unknown')}")
    print(f"  - Anomaly flagged    : {detection['is_anomaly']}")
    print(f"  - Predicted category : {detection['anomaly_type']}")
    print(f"  - Confidence         : {detection['confidence'] * 100:.1f}%")

    if not detection["is_anomaly"]:
        print("\nNo anomaly detected; pipeline continues normally.")
        return True

    print("\n" + THIN)
    print("[STEP 4] Layer 2: diagnostic agent...")
    print(f"  - Reasoning engine   : "
          f"{'Gemini LLM' if diagnostician.llm_enabled else 'deterministic fallback'}")
    diagnosis = diagnostician.diagnose(
        failing_payload=corrupted,
        expected_schema=pipeline.get_schema_spec(),
        layer1_result=detection,
        valid_ranges=pipeline.valid_ranges,
        known_aliases=known_aliases,
    )
    print(json.dumps(diagnosis.model_dump(), indent=2))

    print("\n" + THIN)
    print("[STEP 5] Layer 3: patch generation & sandboxed verification...")
    success, patch_code, verification = orchestrator.heal_pipeline(
        diagnostic=diagnosis,
        failing_payload=corrupted,
        pipeline_id=pipeline_id,
        expected_schema=pipeline.expected_schema,
        valid_ranges=pipeline.valid_ranges,
        known_aliases=known_aliases,
    )
    print(f"  - Verification passed : {success}")
    print(f"  - Held-out pass rate  : "
          f"{verification.get('success_rate', 0.0) * 100:.1f}% "
          f"({verification.get('passed_count')}/{verification.get('total_count')} records)")
    if not success and verification.get("error_trace"):
        print(f"  - Failure detail      : {verification['error_trace'][:500]}")

    print("\nGenerated parser patch:")
    print(patch_code)

    print(THIN)
    print("[STEP 6] Parsing the corrupted payload with the active parser...")
    healed = orchestrator.parse(corrupted, pipeline_id=pipeline_id)
    print(json.dumps(healed, indent=2))

    print("\n" + RULE)
    if success:
        print("[SUCCESS] PIPELINE AUTO-HEALED - NO HUMAN INTERVENTION REQUIRED.")
    else:
        print("[FAILED] PIPELINE COULD NOT BE HEALED AUTOMATICALLY.")
        print("         The previous parser remains in effect; escalate to an operator.")
    print(RULE)
    return success


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline", default="weather", choices=registry.pipeline_ids(),
        help="which ingestion pipeline to demonstrate",
    )
    parser.add_argument(
        "--anomaly", default="schema_drift", choices=list(ANOMALY_CLASSES),
        help="which anomaly class to inject",
    )
    args = parser.parse_args()
    return 0 if run_demo(args.pipeline, args.anomaly) else 1


if __name__ == "__main__":
    sys.exit(main())
