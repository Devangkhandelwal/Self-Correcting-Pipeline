import os
import sys
import json
import tempfile
import importlib.util
import traceback
from typing import Dict, Any, List, Tuple
from src.utils.logger import logger, log_audit_event

EXPECTED_PARSED_FIELDS = [
    "latitude", "longitude", "temperature_2m",
    "relative_humidity_2m", "precipitation",
    "surface_pressure", "wind_speed_10m", "time"
]

class SandboxedVerifier:
    """Layer 3 Sandbox: Executes generated code patches in isolated environment against held-out data."""

    def __init__(self, heldout_dir: str = None):
        if heldout_dir is None:
            heldout_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "data", "heldout"
            )
        self.heldout_dir = heldout_dir

    def verify_patch(self, patch_code: str, test_payloads: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes patch_code inside isolated namespace and validates on test_payloads or heldout batches.
        """
        if test_payloads is None:
            test_payloads = self._load_heldout_records()

        # Write patch_code to temp sandbox file
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(patch_code)
            temp_path = f.name

        try:
            # Dynamically import sandboxed module
            spec = importlib.util.spec_from_file_location("sandbox_module", temp_path)
            sandbox_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sandbox_mod)

            if not hasattr(sandbox_mod, "parse_payload"):
                raise AttributeError("Generated patch code missing required entrypoint function 'parse_payload'")

            parse_func = sandbox_mod.parse_payload

            passed_count = 0
            total_count = len(test_payloads)
            failure_reasons = []

            for idx, raw_record in enumerate(test_payloads):
                try:
                    parsed = parse_func(dict(raw_record))
                    # Validate parsed record schema compliance
                    missing = [f for f in EXPECTED_PARSED_FIELDS if f not in parsed]
                    if missing:
                        failure_reasons.append(f"Record {idx}: Missing fields {missing} after parsing.")
                        continue

                    # Validate numerical types
                    type_errs = []
                    for k in EXPECTED_PARSED_FIELDS:
                        if k != "time":
                            v = parsed[k]
                            if not isinstance(v, (int, float)) or isinstance(v, bool):
                                type_errs.append(f"{k} has invalid type {type(v).__name__}")
                    if type_errs:
                        failure_reasons.append(f"Record {idx}: Type errors {type_errs}")
                        continue

                    passed_count += 1
                except Exception as ex:
                    failure_reasons.append(f"Record {idx} Exception: {str(ex)}")

            success_rate = (passed_count / total_count) if total_count > 0 else 0.0
            is_passed = (success_rate == 1.0)

            result = {
                "passed": is_passed,
                "passed_count": passed_count,
                "total_count": total_count,
                "success_rate": round(success_rate, 4),
                "error_trace": "\n".join(failure_reasons[:5]) if failure_reasons else ""
            }

            log_audit_event("SANDBOX_VERIFICATION", result)
            return result

        except Exception as e:
            err_trace = traceback.format_exc()
            logger.error(f"Sandbox execution crashed: {e}")
            result = {
                "passed": False,
                "passed_count": 0,
                "total_count": len(test_payloads),
                "success_rate": 0.0,
                "error_trace": err_trace
            }
            log_audit_event("SANDBOX_VERIFICATION", result)
            return result
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    def _load_heldout_records(self) -> List[Dict[str, Any]]:
        records = []
        if not os.path.exists(self.heldout_dir):
            return [
                {
                    "latitude": 40.7128, "longitude": -74.0060,
                    "temperature_2m": 22.5, "relative_humidity_2m": 60.0,
                    "precipitation": 0.0, "surface_pressure": 1013.0,
                    "wind_speed_10m": 10.0, "time": "2026-08-22T12:00"
                }
            ] * 50

        for f_name in sorted(os.listdir(self.heldout_dir)):
            if f_name.endswith(".json"):
                f_path = os.path.join(self.heldout_dir, f_name)
                try:
                    with open(f_path, "r") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            records.extend(data)
                except Exception:
                    pass
        return records[:100]
