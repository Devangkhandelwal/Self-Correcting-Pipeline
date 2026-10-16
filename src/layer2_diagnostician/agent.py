import os
import json
import difflib
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from src.utils.logger import logger, log_audit_event

class DiagnosticReport(BaseModel):
    cause: str = Field(description="Summary explanation of the root cause")
    affected_field: str = Field(description="Name of the affected or missing schema field")
    anomaly_category: str = Field(description="Category of anomaly flagged")
    expected_type: str = Field(description="Expected data type or status")
    actual_type: str = Field(description="Observed data type or status")
    found_replacement_key: Optional[str] = Field(default=None, description="Suggested replacement key if field was renamed")
    suggested_fix: str = Field(description="Recommended fix strategy for Layer 3 patch generation")

class LLMDiagnosticAgent:
    """Layer 2: LLM Reasoning Agent for Data Pipeline Failure Root Cause Diagnosis."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def diagnose(
        self,
        failing_payload: Dict[str, Any],
        expected_schema: Dict[str, Any],
        layer1_result: Dict[str, Any],
        recent_logs: str = ""
    ) -> DiagnosticReport:
        """Runs diagnostic reasoning on pipeline failure."""

        anomaly_category = layer1_result.get("anomaly_type", "unknown")
        confidence = layer1_result.get("confidence", 0.0)

        # Attempt LLM call if API key present
        if self.api_key:
            try:
                report = self._call_llm_diagnose(failing_payload, expected_schema, anomaly_category, confidence, recent_logs)
                log_audit_event("LAYER2_DIAGNOSIS", report.model_dump())
                return report
            except Exception as e:
                logger.warning(f"LLM API Call failed ({e}). Falling back to internal reasoning engine.")

        # Fallback structured reasoning engine
        report = self._fallback_reasoning_engine(failing_payload, expected_schema, anomaly_category, confidence)
        log_audit_event("LAYER2_DIAGNOSIS", report.model_dump())
        return report

    def _call_llm_diagnose(self, payload: dict, schema: dict, category: str, confidence: float, logs: str) -> DiagnosticReport:
        from google import genai
        client = genai.Client(api_key=self.api_key)

        prompt = f"""
You are an expert MLOps & Data Engineering Diagnostic Agent.
A data ingestion pipeline experienced a failure flagged by Layer 1 ML Anomaly Detector.

Layer 1 Detection Result:
- Anomaly Category: {category} (Confidence: {confidence})

Expected Schema Specification:
{json.dumps(schema, indent=2)}

Failing JSON Payload Sample:
{json.dumps(payload, indent=2)}

Recent Pipeline Error Logs:
{logs}

Perform step-by-step root-cause diagnosis.
Respond STRICTLY with valid JSON matching this schema:
{{
    "cause": "...",
    "affected_field": "...",
    "anomaly_category": "{category}",
    "expected_type": "...",
    "actual_type": "...",
    "found_replacement_key": "... or null",
    "suggested_fix": "..."
}}
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )
        data = json.loads(response.text)
        return DiagnosticReport(**data)

    def _fallback_reasoning_engine(self, payload: dict, schema: dict, category: str, confidence: float) -> DiagnosticReport:
        keys = set(payload.keys())
        expected_keys = set(schema.keys())

        missing = list(expected_keys - keys)
        extra = list(keys - expected_keys)

        # Detect schema drift / renamed field
        if missing and extra:
            best_match = None
            best_sim = 0.0
            for exp_k in missing:
                for ext_k in extra:
                    sim = difflib.SequenceMatcher(None, exp_k, ext_k).ratio()
                    if sim > best_sim:
                        best_sim = sim
                        best_match = (exp_k, ext_k)
            
            if best_match and best_sim > 0.4:
                missing_field, found_key = best_match
                return DiagnosticReport(
                    cause=f"Field rename detected: API provider renamed '{missing_field}' to '{found_key}'.",
                    affected_field=missing_field,
                    anomaly_category="schema_drift",
                    expected_type=str(schema.get(missing_field, "float")),
                    actual_type="missing_renamed",
                    found_replacement_key=found_key,
                    suggested_fix=f"Update parser to map alias key '{found_key}' to expected key '{missing_field}'."
                )

        # Detect type mismatches
        for k, exp_type in schema.items():
            if k in payload:
                val = payload[k]
                if val is not None:
                    if exp_type in ("float", "int") and isinstance(val, str):
                        return DiagnosticReport(
                            cause=f"Type mismatch on field '{k}': expected numeric {exp_type}, but received string '{val}'.",
                            affected_field=k,
                            anomaly_category="type_mismatch",
                            expected_type=exp_type,
                            actual_type="str",
                            found_replacement_key=None,
                            suggested_fix=f"Add type-casting logic to attempt float/int conversion for field '{k}' with fallback to float conversion."
                        )
                    elif isinstance(val, bool) and exp_type != "bool":
                        return DiagnosticReport(
                            cause=f"Type mismatch on field '{k}': expected {exp_type}, but received boolean {val}.",
                            affected_field=k,
                            anomaly_category="type_mismatch",
                            expected_type=exp_type,
                            actual_type="bool",
                            found_replacement_key=None,
                            suggested_fix=f"Add validation to check boolean types before parsing numerical field '{k}'."
                        )

        # Detect missing / null fields
        if missing:
            field = missing[0]
            return DiagnosticReport(
                cause=f"Missing required schema field '{field}' in incoming payload.",
                affected_field=field,
                anomaly_category="null_field",
                expected_type=str(schema.get(field, "any")),
                actual_type="absent",
                found_replacement_key=None,
                suggested_fix=f"Add default fallback handling for missing field '{field}'."
            )

        for k, v in payload.items():
            if v is None:
                return DiagnosticReport(
                    cause=f"Null value encountered in field '{k}'.",
                    affected_field=k,
                    anomaly_category="null_field",
                    expected_type=str(schema.get(k, "any")),
                    actual_type="null",
                    found_replacement_key=None,
                    suggested_fix=f"Add null-coalescing fallback value (e.g. 0.0 or default median) for field '{k}'."
                )

        # Detect outliers
        for k, v in payload.items():
            if k == "temperature_2m" and isinstance(v, (int, float)) and (v > 100 or v < -100):
                return DiagnosticReport(
                    cause=f"Out-of-range sensor spike detected in field '{k}' with value {v}.",
                    affected_field=k,
                    anomaly_category="outlier",
                    expected_type="float [-60, 70]",
                    actual_type=f"float ({v})",
                    found_replacement_key=None,
                    suggested_fix=f"Add boundary clipping / outlier filter for field '{k}' to reject or smooth extreme values."
                )

        # Duplicate records
        if "_is_duplicate_flag" in payload or "duplicate_uuid" in payload:
            return DiagnosticReport(
                cause="Duplicate payload record detected in ingestion stream.",
                affected_field="duplicate_uuid",
                anomaly_category="duplicate_record",
                expected_type="unique_record",
                actual_type="duplicate",
                found_replacement_key=None,
                suggested_fix="Enable deduplication filter based on payload hash or timestamp."
            )

        return DiagnosticReport(
            cause="Unknown pipeline anomaly detected.",
            affected_field="general",
            anomaly_category=category,
            expected_type="valid_payload",
            actual_type="corrupted",
            found_replacement_key=None,
            suggested_fix="Inspect schema specification and apply robust parser validation."
        )
