import random
import copy
from typing import Dict, Any, Tuple, Optional

ANOMALY_CLASSES = [
    "normal",
    "schema_drift",
    "type_mismatch",
    "null_field",
    "outlier",
    "duplicate_record"
]

DEFAULT_FIELD_RENAMES = {
    "temperature_2m": "temp_2m",
    "relative_humidity_2m": "humidity",
    "precipitation": "precip",
    "surface_pressure": "pressure",
    "wind_speed_10m": "wind_speed"
}

class SyntheticAnomalyInjector:
    """Deliberately corrupts clean API payloads to produce labeled datasets and simulate live failure scenarios."""

    def inject_anomaly(
        self,
        payload: Dict[str, Any],
        target_anomaly: str = None,
        field_renames: Optional[Dict[str, str]] = None,
        valid_ranges: Optional[Dict[str, tuple]] = None
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
        """
        Injects an anomaly into payload.
        Returns: (corrupted_payload, anomaly_label, metadata_info)
        """
        if target_anomaly is None:
            target_anomaly = random.choice(ANOMALY_CLASSES)

        if field_renames is None:
            field_renames = DEFAULT_FIELD_RENAMES

        corrupted = copy.deepcopy(payload)
        metadata = {"original_payload": copy.deepcopy(payload), "applied_anomaly": target_anomaly}

        if target_anomaly == "normal":
            pass

        elif target_anomaly == "schema_drift":
            # Rename a field based on active pipeline's rename candidates or matching keys
            rename_candidates = [k for k in field_renames.keys() if k in corrupted]
            if not rename_candidates:
                # Fallback to any string key if no exact map match
                rename_candidates = [k for k, v in corrupted.items() if not k.startswith("_")]

            if rename_candidates:
                field = random.choice(rename_candidates)
                new_field = field_renames.get(field, f"{field}_renamed")
                corrupted[new_field] = corrupted.pop(field)
                metadata["renamed_from"] = field
                metadata["renamed_to"] = new_field

        elif target_anomaly == "type_mismatch":
            numeric_fields = [k for k, v in corrupted.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if numeric_fields:
                field = random.choice(numeric_fields)
                val = corrupted[field]
                corrupt_type = random.choice(["string_number", "string_text", "bool"])
                if corrupt_type == "string_number":
                    corrupted[field] = str(val)
                elif corrupt_type == "string_text":
                    corrupted[field] = "INVALID_NUMERIC"
                else:
                    corrupted[field] = True
                metadata["field"] = field
                metadata["corrupt_type"] = corrupt_type

        elif target_anomaly == "null_field":
            target_fields = [k for k in corrupted.keys() if not k.startswith("_")]
            if target_fields:
                field = random.choice(target_fields)
                action = random.choice(["set_null", "delete_key"])
                if action == "set_null":
                    corrupted[field] = None
                else:
                    corrupted.pop(field, None)
                metadata["field"] = field
                metadata["action"] = action

        elif target_anomaly == "outlier":
            numeric_fields = [k for k, v in corrupted.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if numeric_fields:
                field = random.choice(numeric_fields)
                val = float(corrupted[field])
                # Generate extreme outlier relative to magnitude or valid range
                outlier_val = (val * 100.0) + 9999.0 if val >= 0 else (val * 100.0) - 9999.0
                if valid_ranges and field in valid_ranges:
                    max_range = valid_ranges[field][1]
                    outlier_val = max_range * 10.0 + 999.0

                corrupted[field] = round(outlier_val, 2)
                metadata["field"] = field
                metadata["outlier_val"] = corrupted[field]

        elif target_anomaly == "duplicate_record":
            corrupted["_is_duplicate_flag"] = True
            corrupted["duplicate_uuid"] = "DUP_12345_ID"
            metadata["action"] = "duplicate_inserted"

        return corrupted, target_anomaly, metadata
