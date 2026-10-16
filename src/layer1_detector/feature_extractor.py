import json
import difflib
from typing import Dict, Any, List, Optional

DEFAULT_EXPECTED_FIELDS = {
    "latitude": float,
    "longitude": float,
    "temperature_2m": float,
    "relative_humidity_2m": float,
    "precipitation": float,
    "surface_pressure": float,
    "wind_speed_10m": float,
    "time": str
}

DEFAULT_VALID_RANGES = {
    "temperature_2m": (-60.0, 70.0),
    "relative_humidity_2m": (0.0, 100.0),
    "precipitation": (0.0, 500.0),
    "surface_pressure": (800.0, 1200.0),
    "wind_speed_10m": (0.0, 300.0)
}

class FeatureExtractor:
    """Extracts numeric feature vectors from JSON payloads for ML Anomaly Classification."""

    FEATURE_NAMES = [
        "total_keys",
        "missing_expected_count",
        "unexpected_extra_count",
        "type_mismatch_count",
        "null_val_count",
        "max_key_similarity",
        "outlier_count",
        "duplicate_flag",
        "string_type_count",
        "bool_type_count"
    ]

    def extract_features(
        self,
        payload: Dict[str, Any],
        expected_fields: Optional[Dict[str, type]] = None,
        valid_ranges: Optional[Dict[str, tuple]] = None
    ) -> List[float]:
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = {}

        if expected_fields is None:
            expected_fields = DEFAULT_EXPECTED_FIELDS

        if valid_ranges is None:
            valid_ranges = DEFAULT_VALID_RANGES

        keys = set(payload.keys())
        expected_keys = set(expected_fields.keys())

        missing_keys = expected_keys - keys
        extra_keys = keys - expected_keys

        # Type mismatches & nulls
        type_mismatches = 0
        null_count = 0
        outliers = 0
        string_count = 0
        bool_count = 0

        for k, v in payload.items():
            if v is None:
                null_count += 1
            elif isinstance(v, bool):
                bool_count += 1
                if k in expected_fields and expected_fields[k] != bool:
                    type_mismatches += 1
            elif isinstance(v, str):
                string_count += 1
                if k in expected_fields and expected_fields[k] in (float, int):
                    type_mismatches += 1
            elif isinstance(v, (int, float)):
                if k in expected_fields and expected_fields[k] == str:
                    type_mismatches += 1
                if k in valid_ranges:
                    min_v, max_v = valid_ranges[k]
                    if v < min_v or v > max_v:
                        outliers += 1

        # Key similarity score (for schema drift detection)
        max_sim = 0.0
        for extra in extra_keys:
            for missing in missing_keys:
                sim = difflib.SequenceMatcher(None, extra, missing).ratio()
                if sim > max_sim:
                    max_sim = sim

        duplicate_flag = 1.0 if "_is_duplicate_flag" in payload or "duplicate_uuid" in payload else 0.0

        return [
            float(len(keys)),
            float(len(missing_keys)),
            float(len(extra_keys)),
            float(type_mismatches),
            float(null_count),
            float(max_sim),
            float(outliers),
            float(duplicate_flag),
            float(string_count),
            float(bool_count)
        ]
