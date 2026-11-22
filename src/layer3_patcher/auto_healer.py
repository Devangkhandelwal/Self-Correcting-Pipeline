"""Layer 3 orchestrator: generate -> verify -> hot-swap, with real retries.

Changes from the previous version:
  * the pipeline's schema and ranges are resolved and threaded into both the
    generator and the verifier, so healing is no longer weather-only;
  * each retry escalates to a different generation strategy and the prior
    failure is fed back, instead of regenerating byte-identical code;
  * the verification batch includes corrupted records, not only clean ones,
    so a patch is actually tested against the fault it was written to fix;
  * patch files loaded from disk are AST-screened before execution;
  * the pipeline id list comes from the registry rather than a hardcoded
    literal, and the legacy `active_parser.py` fallback is gone;
  * `parse()` copies its input, so a generated parser cannot mutate the
    caller's dict (the Streamlit dashboard passed session state directly).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from src.layer2_diagnostician.agent import DiagnosticReport
from src.layer3_patcher.generator import PatchGenerator
from src.layer3_patcher.sandbox import (
    PatchValidationError,
    SandboxedVerifier,
    validate_patch_source,
)
from src.utils import config
from src.utils.logger import log_audit_event, logger

config.PATCH_DIR.mkdir(parents=True, exist_ok=True)


class SelfHealingOrchestrator:
    """Runs the patch/verify/activate loop for one or more pipelines."""

    def __init__(self, max_retries: Optional[int] = None):
        self.generator = PatchGenerator()
        self.verifier = SandboxedVerifier()
        self.max_retries = max_retries or config.MAX_HEAL_RETRIES
        self.active_parsers: Dict[str, Callable[[dict], dict]] = {}
        self.load_all_active_parsers()

    # -- paths ---------------------------------------------------------------

    @staticmethod
    def _patch_filepath(pipeline_id: str) -> Path:
        return config.PATCH_DIR / f"active_parser_{pipeline_id}.py"

    @staticmethod
    def _resolve_contract(
        pipeline_id: str,
        expected_schema: Optional[Dict[str, Any]],
        valid_ranges: Optional[Dict[str, tuple]],
        known_aliases: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[str, str], Dict[str, tuple], Dict[str, str]]:
        """Return (schema_spec, valid_ranges, known_aliases) for a pipeline.

        Anything the caller does not supply is looked up in the registry, so
        existing two-argument callers keep working.
        """
        from src.ingestion.pipeline_registry import registry

        pipeline = registry.get_pipeline(pipeline_id)
        if pipeline is None and expected_schema is None:
            raise ValueError(f"Unknown pipeline_id '{pipeline_id}'")

        if expected_schema is not None:
            schema_spec = {
                field: (declared if isinstance(declared, str)
                        else getattr(declared, "__name__", "str"))
                for field, declared in expected_schema.items()
            }
        else:
            schema_spec = pipeline.get_schema_spec()

        if valid_ranges is None:
            valid_ranges = pipeline.valid_ranges if pipeline else {}

        if known_aliases is None:
            known_aliases = (
                {alias: canonical for canonical, alias in pipeline.field_renames.items()}
                if pipeline else {}
            )

        return schema_spec, valid_ranges, known_aliases

    # -- healing loop --------------------------------------------------------

    def heal_pipeline(
        self,
        diagnostic: DiagnosticReport,
        failing_payload: Dict[str, Any],
        pipeline_id: str = "weather",
        expected_schema: Optional[Dict[str, Any]] = None,
        valid_ranges: Optional[Dict[str, tuple]] = None,
        known_aliases: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Generate, verify and activate a patch for a diagnosed failure.

        Returns (success, patch_code, verification_report).
        """
        schema_spec, ranges, aliases = self._resolve_contract(
            pipeline_id, expected_schema, valid_ranges, known_aliases
        )
        test_batch = self._build_test_batch(failing_payload, pipeline_id)

        feedback_context = ""
        verification_result: Dict[str, Any] = {}
        last_patch_code = ""

        for attempt in range(1, self.max_retries + 1):
            strategy = self.generator.describe_strategy(attempt)
            logger.info(
                "Self-healing loop (%s): attempt %d/%d using '%s' strategy",
                pipeline_id, attempt, self.max_retries, strategy,
            )

            patch_code = self.generator.generate_patch(
                diagnostic,
                schema_spec=schema_spec,
                valid_ranges=ranges,
                observed_payload=failing_payload,
                feedback_context=feedback_context,
                attempt=attempt,
                pipeline_id=pipeline_id,
                known_aliases=aliases,
            )
            last_patch_code = patch_code

            verification_result = self.verifier.verify_patch(
                patch_code,
                test_payloads=test_batch,
                schema_spec=schema_spec,
                valid_ranges=ranges,
                pipeline_id=pipeline_id,
            )

            if verification_result["passed"]:
                activated = self._save_and_activate_patch(pipeline_id, patch_code)
                log_audit_event("PIPELINE_HEALED", {
                    "pipeline_id": pipeline_id,
                    "attempt": attempt,
                    "strategy": strategy,
                    "affected_field": diagnostic.affected_field,
                    "anomaly_category": diagnostic.anomaly_category,
                    "activated": activated,
                    "verification": verification_result,
                })
                if activated:
                    logger.info(
                        "Patch verified for '%s' on attempt %d; parser hot-swapped.",
                        pipeline_id, attempt,
                    )
                    return True, patch_code, verification_result

                # Verified but could not be activated: report honestly
                # rather than claiming a heal that is not in effect.
                logger.error(
                    "Patch for '%s' verified but could not be activated.", pipeline_id
                )
                verification_result = dict(verification_result)
                verification_result["passed"] = False
                verification_result["error_trace"] = (
                    "Patch passed verification but could not be activated; "
                    "the previous parser remains in effect."
                )
                return False, patch_code, verification_result

            error_trace = verification_result.get("error_trace", "unknown sandbox error")
            logger.warning(
                "Patch verification failed for '%s' on attempt %d (%s): %s",
                pipeline_id, attempt, strategy, error_trace,
            )
            feedback_context = (
                f"Attempt {attempt} using the '{strategy}' strategy failed "
                f"verification at {verification_result.get('success_rate', 0.0) * 100:.0f}% "
                f"with: {error_trace}"
            )

        logger.error(
            "Self-healing loop for '%s' exhausted %d attempts without a passing patch.",
            pipeline_id, self.max_retries,
        )
        log_audit_event("HEAL_FAILED", {
            "pipeline_id": pipeline_id,
            "attempts": self.max_retries,
            "anomaly_category": diagnostic.anomaly_category,
            "affected_field": diagnostic.affected_field,
            "diagnostic": diagnostic.model_dump(),
            "verification": verification_result,
        })
        return False, last_patch_code, verification_result

    def _build_test_batch(self, failing_payload: Dict[str, Any], pipeline_id: str) -> list:
        """Clean held-out records plus the record that actually broke.

        The old batch was 30 near-identical clean records; since all of them
        were clean, a patch was never tested against the corruption it
        existed to repair.
        """
        heldout = self.verifier.load_heldout_records(pipeline_id=pipeline_id)
        return [dict(failing_payload)] + heldout

    # -- activation ----------------------------------------------------------

    def _save_and_activate_patch(self, pipeline_id: str, patch_code: str) -> bool:
        """Persist a verified patch and bind it as the live parser."""
        patch_file = self._patch_filepath(pipeline_id)
        try:
            patch_file.write_text(patch_code, encoding="utf-8")
        except OSError as exc:
            logger.error("Could not write patch file %s: %s", patch_file, exc)
            return False

        parser = self._load_parser(patch_file)
        if parser is None:
            return False
        self.active_parsers[pipeline_id] = parser
        return True

    @staticmethod
    def _load_parser(patch_file: Path) -> Optional[Callable[[dict], dict]]:
        """AST-screen a patch file, then import it and return its entrypoint.

        Activation necessarily executes the patch in this process -- that is
        what hot-swapping means -- so screening is what makes it defensible.
        Verification itself runs out-of-process (see sandbox.py).
        """
        try:
            source = patch_file.read_text(encoding="utf-8")
        except OSError as exc:
            logger.error("Could not read patch file %s: %s", patch_file, exc)
            return None

        try:
            validate_patch_source(source)
        except PatchValidationError as exc:
            logger.error(
                "Refusing to activate %s: failed patch validation (%s)",
                patch_file.name, exc,
            )
            return None

        try:
            spec = importlib.util.spec_from_file_location(patch_file.stem, patch_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"no import spec for {patch_file}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            parser = getattr(module, "parse_payload", None)
            if not callable(parser):
                raise AttributeError("patch defines no callable 'parse_payload'")
            return parser
        except Exception as exc:
            logger.error("Could not activate parser from %s: %s", patch_file.name, exc)
            return None

    def load_all_active_parsers(self) -> None:
        """Re-bind previously verified parsers for every known pipeline."""
        from src.ingestion.pipeline_registry import registry

        for pipeline_id in registry.pipeline_ids():
            patch_file = self._patch_filepath(pipeline_id)
            if not patch_file.exists():
                continue
            parser = self._load_parser(patch_file)
            if parser is not None:
                self.active_parsers[pipeline_id] = parser
                logger.info(
                    "Loaded active parser patch for pipeline '%s'", pipeline_id
                )

    # -- runtime -------------------------------------------------------------

    def parse(
        self,
        raw_payload: dict,
        pipeline_id: str = "weather",
        is_duplicate: bool = False,
    ) -> dict:
        """Parse a record with the active healed parser, if one is bound.

        `is_duplicate` carries the Layer 1 verdict into the output. The
        parser cannot determine this itself: deduplication needs state
        across records and a generated `parse_payload` sees exactly one.
        Detection owns that state; this records its finding.
        """
        parser = self.active_parsers.get(pipeline_id)
        if parser is None:
            parsed = dict(raw_payload)
        else:
            try:
                parsed = parser(dict(raw_payload))
            except Exception as exc:
                logger.warning(
                    "Active parser for '%s' raised %s: %s; returning record unchanged.",
                    pipeline_id, type(exc).__name__, exc,
                )
                parsed = dict(raw_payload)

        if is_duplicate:
            parsed["_is_duplicate"] = True
        return parsed

    def has_active_parser(self, pipeline_id: str) -> bool:
        return pipeline_id in self.active_parsers
