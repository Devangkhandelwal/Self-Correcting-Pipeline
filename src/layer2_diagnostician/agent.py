"""Layer 2: root-cause diagnosis via Gemini, with a deterministic fallback.

The fallback engine previously recognised outliers only in the hardcoded
field `temperature_2m`. Every other field -- 4 of 5 weather metrics and every
finance, e-commerce and IoT range -- fell through to a generic report with
affected_field="general", which Layer 3 turned into a no-op patch. The
anomaly then flowed downstream untouched while the UI showed a green
"healed" badge. Outlier detection is now driven by the pipeline's declared
`valid_ranges`.

The cascade is also dispatched on the category Layer 1 predicted, rather than
always running schema-drift checks first, so a confident Layer 1 prediction is
no longer discarded by an earlier branch.
"""
from __future__ import annotations

import difflib
import json
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field

from src.utils import config
from src.utils.logger import log_audit_event, logger

NUMERIC_TYPES = {"float", "int"}


class DiagnosticReport(BaseModel):
    cause: str = Field(description="Summary explanation of the root cause")
    affected_field: str = Field(description="Name of the affected or missing schema field")
    anomaly_category: str = Field(description="Category of anomaly flagged")
    expected_type: str = Field(description="Expected data type or status")
    actual_type: str = Field(description="Observed data type or status")
    found_replacement_key: Optional[str] = Field(
        default=None, description="Suggested replacement key if field was renamed"
    )
    suggested_fix: str = Field(description="Recommended fix strategy for Layer 3 patch generation")


class LLMDiagnosticAgent:
    """Diagnoses a flagged ingestion failure and reports it as structured data."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key if api_key is not None else config.GEMINI_API_KEY
        self.model = model or config.GEMINI_MODEL
        self.similarity_threshold = config.KEY_SIMILARITY_THRESHOLD

    @property
    def llm_enabled(self) -> bool:
        """Whether the Gemini path is configured; drives UI provenance labels."""
        return bool(self.api_key)

    def diagnose(
        self,
        failing_payload: Dict[str, Any],
        expected_schema: Dict[str, Any],
        layer1_result: Dict[str, Any],
        recent_logs: str = "",
        valid_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        known_aliases: Optional[Dict[str, str]] = None,
    ) -> DiagnosticReport:
        """Produce a root-cause report for a flagged payload.

        `known_aliases` maps an observed key onto the canonical schema field
        it stands in for -- the rename table the pipeline already declares.
        Consulting it first matters because semantic renames such as
        `reorder_level` -> `min_stock` score only 0.09 on string similarity
        and are invisible to lexical matching.
        """
        category = layer1_result.get("anomaly_type", "unknown")
        confidence = layer1_result.get("confidence", 0.0)
        ranges = valid_ranges or {}
        aliases = known_aliases or {}

        if self.api_key:
            try:
                report = self._call_llm_diagnose(
                    failing_payload, expected_schema, category, confidence, recent_logs
                )
                log_audit_event("LAYER2_DIAGNOSIS", {**report.model_dump(), "source": "gemini"})
                return report
            except Exception as exc:
                logger.warning(
                    "Gemini diagnosis failed (%s: %s); using deterministic engine.",
                    type(exc).__name__, exc,
                )

        # Duplicate detection is content-based and stateful, so the verdict
        # comes from Layer 1 rather than from marker keys in the payload.
        is_duplicate = bool(
            layer1_result.get("features", {}).get("duplicate_flag", 0.0)
        )
        report = self._fallback_reasoning_engine(
            failing_payload, expected_schema, category, ranges, aliases, is_duplicate
        )
        log_audit_event("LAYER2_DIAGNOSIS", {**report.model_dump(), "source": "deterministic"})
        return report

    # -- LLM path ------------------------------------------------------------

    def _call_llm_diagnose(
        self,
        payload: dict,
        schema: dict,
        category: str,
        confidence: float,
        logs: str,
    ) -> DiagnosticReport:
        from google import genai

        client = genai.Client(api_key=self.api_key)
        prompt = f"""You are an expert MLOps and data engineering diagnostic agent.
A data ingestion pipeline experienced a failure flagged by a Layer 1 ML anomaly detector.

Layer 1 detection result:
- Anomaly category: {category} (confidence: {confidence})

Expected schema specification:
{json.dumps(schema, indent=2)}

Failing JSON payload sample:
{json.dumps(payload, indent=2, default=str)}

Recent pipeline error logs:
{logs or "(none)"}

Perform a step-by-step root-cause diagnosis.
Respond STRICTLY with valid JSON matching this schema:
{{
    "cause": "...",
    "affected_field": "one of the expected schema field names",
    "anomaly_category": "{category}",
    "expected_type": "...",
    "actual_type": "...",
    "found_replacement_key": "the observed key that replaced the affected field, or null",
    "suggested_fix": "..."
}}
"""
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": DiagnosticReport,
            },
        )
        if not getattr(response, "text", None):
            raise ValueError("Gemini returned an empty response")

        report = DiagnosticReport(**json.loads(response.text))
        return self._sanitise_llm_report(report, payload, schema)

    @staticmethod
    def _sanitise_llm_report(
        report: DiagnosticReport, payload: dict, schema: dict
    ) -> DiagnosticReport:
        """Reject model-invented field names before they reach codegen.

        Layer 3 generates code from these values, so an hallucinated field
        name would produce a patch that silently targets nothing.
        """
        if report.affected_field not in schema:
            logger.warning(
                "Gemini named unknown field '%s'; discarding LLM report.",
                report.affected_field,
            )
            raise ValueError(f"LLM returned unknown affected_field '{report.affected_field}'")

        if report.found_replacement_key and report.found_replacement_key not in payload:
            logger.warning(
                "Gemini named replacement key '%s' absent from the payload; clearing it.",
                report.found_replacement_key,
            )
            report = report.model_copy(update={"found_replacement_key": None})

        return report

    # -- deterministic fallback ---------------------------------------------

    def _fallback_reasoning_engine(
        self,
        payload: dict,
        schema: dict,
        category: str,
        valid_ranges: Dict[str, Tuple[float, float]],
        known_aliases: Dict[str, str],
        is_duplicate: bool = False,
    ) -> DiagnosticReport:
        """Rule-based diagnosis, dispatched on the predicted category.

        The predicted category is tried first, then the remaining detectors in
        order of specificity, so a confident Layer 1 prediction is honoured
        without losing the ability to re-classify a mislabelled payload.
        """
        detectors = {
            "schema_drift": lambda: self._detect_drift(payload, schema, known_aliases),
            "type_mismatch": lambda: self._detect_type_mismatch(payload, schema),
            "outlier": lambda: self._detect_outlier(payload, schema, valid_ranges),
            "null_field": lambda: self._detect_missing_or_null(payload, schema),
            "duplicate_record": lambda: self._detect_duplicate(payload, is_duplicate),
        }

        order = ([category] if category in detectors else []) + [
            name for name in ("schema_drift", "type_mismatch", "outlier",
                              "null_field", "duplicate_record")
            if name != category
        ]

        for name in order:
            report = detectors[name]()
            if report is not None:
                return report

        return DiagnosticReport(
            cause="Payload deviates from the expected schema, but no specific "
                  "rule matched. Applying general normalisation.",
            affected_field=next(iter(schema), "general"),
            anomaly_category=category,
            expected_type="valid_payload",
            actual_type="corrupted",
            found_replacement_key=None,
            suggested_fix="Apply full schema normalisation: coerce declared types, "
                          "backfill missing fields and clamp values to valid ranges.",
        )

    def _detect_drift(
        self,
        payload: dict,
        schema: dict,
        known_aliases: Optional[Dict[str, str]] = None,
    ) -> Optional[DiagnosticReport]:
        missing = [field for field in schema if field not in payload]
        extra = [key for key in payload if key not in schema]
        if not (missing and extra):
            return None

        # A declared alias is authoritative configuration; prefer it over a
        # lexical guess, which cannot see semantic renames.
        for observed_key in extra:
            canonical = (known_aliases or {}).get(observed_key)
            if canonical and canonical in missing:
                return DiagnosticReport(
                    cause=f"Field rename detected: observed key '{observed_key}' is a "
                          f"declared alias for '{canonical}' in this pipeline's rename table.",
                    affected_field=canonical,
                    anomaly_category="schema_drift",
                    expected_type=str(schema.get(canonical, "float")),
                    actual_type="missing_renamed",
                    found_replacement_key=observed_key,
                    suggested_fix=f"Update the parser to map alias key '{observed_key}' onto "
                                  f"expected key '{canonical}'.",
                )

        best: Optional[Tuple[str, str]] = None
        best_score = 0.0
        for expected_key in missing:
            for observed_key in extra:
                score = difflib.SequenceMatcher(None, expected_key, observed_key).ratio()
                if score > best_score:
                    best_score, best = score, (expected_key, observed_key)

        if not best or best_score < self.similarity_threshold:
            return None

        missing_field, found_key = best
        return DiagnosticReport(
            cause=f"Probable field rename inferred from key similarity: "
                  f"'{missing_field}' appears to have been renamed to '{found_key}' "
                  f"(similarity {best_score:.2f}; no declared alias matched).",
            affected_field=missing_field,
            anomaly_category="schema_drift",
            expected_type=str(schema.get(missing_field, "float")),
            actual_type="missing_renamed",
            found_replacement_key=found_key,
            suggested_fix=f"Update the parser to map alias key '{found_key}' onto "
                          f"expected key '{missing_field}'.",
        )

    @staticmethod
    def _detect_type_mismatch(payload: dict, schema: dict) -> Optional[DiagnosticReport]:
        for field, declared in schema.items():
            value = payload.get(field)
            if value is None:
                continue

            if isinstance(value, bool) and declared != "bool":
                return DiagnosticReport(
                    cause=f"Type mismatch on '{field}': expected {declared}, "
                          f"received boolean {value}.",
                    affected_field=field,
                    anomaly_category="type_mismatch",
                    expected_type=str(declared),
                    actual_type="bool",
                    found_replacement_key=None,
                    suggested_fix=f"Reject boolean values for '{field}' and substitute "
                                  f"a schema-safe default.",
                )

            if declared in NUMERIC_TYPES and isinstance(value, str):
                return DiagnosticReport(
                    cause=f"Type mismatch on '{field}': expected numeric {declared}, "
                          f"received the string {value!r}.",
                    affected_field=field,
                    anomaly_category="type_mismatch",
                    expected_type=str(declared),
                    actual_type="str",
                    found_replacement_key=None,
                    suggested_fix=f"Attempt numeric coercion of '{field}', falling back "
                                  f"to a schema-safe default when the value is not numeric.",
                )

            if declared == "str" and isinstance(value, (int, float)):
                return DiagnosticReport(
                    cause=f"Type mismatch on '{field}': expected str, "
                          f"received {type(value).__name__} {value}.",
                    affected_field=field,
                    anomaly_category="type_mismatch",
                    expected_type="str",
                    actual_type=type(value).__name__,
                    found_replacement_key=None,
                    suggested_fix=f"Cast '{field}' to str during parsing.",
                )
        return None

    @staticmethod
    def _detect_outlier(
        payload: dict,
        schema: dict,
        valid_ranges: Dict[str, Tuple[float, float]],
    ) -> Optional[DiagnosticReport]:
        """Flag any numeric field outside its declared range.

        Range-driven, so it covers every field of every pipeline instead of
        the single hardcoded `temperature_2m` check it replaces.
        """
        for field, bounds in valid_ranges.items():
            value = payload.get(field)
            if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            low, high = bounds
            if low <= value <= high:
                continue
            return DiagnosticReport(
                cause=f"Out-of-range value detected in '{field}': {value} falls outside "
                      f"the valid range ({low}, {high}).",
                affected_field=field,
                anomaly_category="outlier",
                expected_type=f"{schema.get(field, 'float')} [{low}, {high}]",
                actual_type=f"{type(value).__name__} ({value})",
                found_replacement_key=None,
                suggested_fix=f"Clamp '{field}' to the valid range ({low}, {high}) "
                              f"so extreme sensor spikes cannot propagate downstream.",
            )
        return None

    @staticmethod
    def _detect_missing_or_null(payload: dict, schema: dict) -> Optional[DiagnosticReport]:
        missing = [field for field in schema if field not in payload]
        if missing:
            field = missing[0]
            return DiagnosticReport(
                cause=f"Required schema field '{field}' is absent from the payload.",
                affected_field=field,
                anomaly_category="null_field",
                expected_type=str(schema.get(field, "any")),
                actual_type="absent",
                found_replacement_key=None,
                suggested_fix=f"Backfill '{field}' with a schema-safe default when absent.",
            )

        for field in schema:
            if field in payload and payload[field] is None:
                return DiagnosticReport(
                    cause=f"Null value encountered in required field '{field}'.",
                    affected_field=field,
                    anomaly_category="null_field",
                    expected_type=str(schema.get(field, "any")),
                    actual_type="null",
                    found_replacement_key=None,
                    suggested_fix=f"Apply a null-coalescing default for '{field}'.",
                )
        return None

    @staticmethod
    def _detect_duplicate(
        payload: dict, is_duplicate: bool = False
    ) -> Optional[DiagnosticReport]:
        """Report a replayed record.

        Driven by the content-fingerprint verdict from Layer 1. Legacy
        marker keys are still honoured so a dataset generated before
        content-based deduplication still diagnoses correctly.
        """
        legacy = [k for k in ("_is_duplicate_flag", "duplicate_uuid") if k in payload]
        if not is_duplicate and not legacy:
            return None

        evidence = (
            "its content fingerprint matches a record already seen in this stream"
            if is_duplicate
            else f"it carries legacy duplicate markers ({', '.join(legacy)})"
        )
        return DiagnosticReport(
            cause=f"Duplicate record detected in the ingestion stream: {evidence}.",
            affected_field=legacy[0] if legacy else "_is_duplicate",
            anomaly_category="duplicate_record",
            expected_type="unique_record",
            actual_type="duplicate",
            found_replacement_key=None,
            suggested_fix="Drop the record, or flag it so downstream consumers can "
                          "deduplicate. A stateless parser cannot do this: "
                          "deduplication needs stream state and belongs in the "
                          "ingestion filter.",
        )
