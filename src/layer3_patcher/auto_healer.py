import os
import json
import importlib.util
from typing import Dict, Any, Tuple, Optional
from src.layer2_diagnostician.agent import DiagnosticReport
from src.layer3_patcher.generator import PatchGenerator
from src.layer3_patcher.sandbox import SandboxedVerifier
from src.utils.logger import logger, log_audit_event

PATCH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "patches")
os.makedirs(PATCH_DIR, exist_ok=True)

class SelfHealingOrchestrator:
    """Layer 3: Self-Healing Pipeline Orchestrator with Patch Generation & Verification Retry Loop."""

    def __init__(self, max_retries: int = 3):
        self.generator = PatchGenerator()
        self.verifier = SandboxedVerifier()
        self.max_retries = max_retries
        self.active_parsers: Dict[str, Any] = {}
        self.load_all_active_parsers()

    def _get_patch_filepath(self, pipeline_id: str) -> str:
        return os.path.join(PATCH_DIR, f"active_parser_{pipeline_id}.py")

    def heal_pipeline(
        self,
        diagnostic: DiagnosticReport,
        failing_payload: Dict[str, Any],
        pipeline_id: str = "weather"
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Executes detect -> diagnose -> patch -> verify -> heal loop.
        Returns: (success: bool, active_patch_code: str, verification_report: dict)
        """
        feedback_context = ""
        verification_result = {}
        last_patch_code = ""

        # Test batch including current failing payload + held-out clean records
        test_batch = [failing_payload] + self.verifier._load_heldout_records()[:30]

        for attempt in range(1, self.max_retries + 1):
            logger.info(f"Self-Healing Loop ({pipeline_id}): Patch Generation Attempt {attempt}/{self.max_retries}")

            # Generate patch code
            patch_code = self.generator.generate_patch(diagnostic, feedback_context=feedback_context)
            last_patch_code = patch_code

            # Verify patch in sandbox
            verification_result = self.verifier.verify_patch(patch_code, test_payloads=test_batch)

            if verification_result["passed"]:
                logger.info(f"Patch VERIFIED SUCCESSFULLY for '{pipeline_id}' on attempt {attempt}! Hot-swapping parser.")
                self._save_and_activate_patch(pipeline_id, patch_code)
                log_audit_event("PIPELINE_HEALED", {
                    "pipeline_id": pipeline_id,
                    "attempt": attempt,
                    "affected_field": diagnostic.affected_field,
                    "anomaly_category": diagnostic.anomaly_category,
                    "verification": verification_result
                })
                return True, patch_code, verification_result
            else:
                err_trace = verification_result.get("error_trace", "Unknown sandbox error")
                logger.warning(f"Patch verification failed for '{pipeline_id}' on attempt {attempt}: {err_trace}")
                feedback_context = f"Attempt {attempt} failed verification with error: {err_trace}"

        logger.error(f"Self-Healing loop for '{pipeline_id}' exhausted all retry attempts without producing a passing patch.")
        log_audit_event("HEAL_FAILED", {
            "pipeline_id": pipeline_id,
            "diagnostic": diagnostic.model_dump(),
            "verification": verification_result
        })
        return False, last_patch_code, verification_result

    def _save_and_activate_patch(self, pipeline_id: str, patch_code: str):
        """Saves verified patch code to disk and updates live function reference per pipeline."""
        patch_file = self._get_patch_filepath(pipeline_id)
        with open(patch_file, "w") as f:
            f.write(patch_code)

        spec = importlib.util.spec_from_file_location(f"active_parser_{pipeline_id}", patch_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.active_parsers[pipeline_id] = mod.parse_payload

    def load_all_active_parsers(self):
        """Loads previously active patch parsers for all pipelines if exist."""
        for p_id in ["weather", "finance", "ecommerce", "iot"]:
            patch_file = self._get_patch_filepath(p_id)
            # Fallback legacy active_parser.py check for weather
            if not os.path.exists(patch_file) and p_id == "weather":
                legacy_file = os.path.join(PATCH_DIR, "active_parser.py")
                if os.path.exists(legacy_file):
                    patch_file = legacy_file

            if os.path.exists(patch_file):
                try:
                    spec = importlib.util.spec_from_file_location(f"active_parser_{p_id}", patch_file)
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    self.active_parsers[p_id] = mod.parse_payload
                    logger.info(f"Loaded active parser patch for pipeline '{p_id}' from {patch_file}")
                except Exception as e:
                    logger.warning(f"Could not load active parser patch for '{p_id}': {e}")

    def parse(self, raw_payload: dict, pipeline_id: str = "weather") -> dict:
        """Parses payload using active self-healed parser (or identity if none active)."""
        parser_func = self.active_parsers.get(pipeline_id)
        if parser_func:
            try:
                return parser_func(raw_payload)
            except Exception as e:
                logger.warning(f"Active parser failed for '{pipeline_id}': {e}")
        return raw_payload
