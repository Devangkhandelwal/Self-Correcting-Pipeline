"""Layer 1 feature extraction: JSON payload -> numeric feature vector.

The expected-field and valid-range defaults used to be a second, hand-copied
definition of the weather schema. It silently diverged from the registry's --
once latitude/longitude gained declared ranges there, outliers injected into
those fields were invisible to the extractor, so 10 of 240 test rows were
labelled `outlier` while their features looked perfectly normal. The defaults
are now resolved from the registry, so there is one definition.
"""
from __future__ import annotations

import difflib
import json
from typing import Any, Dict, List, Optional, Tuple

#: Keys the injector adds to mark a replayed record.
DUPLICATE_MARKER_KEYS = ("_is_duplicate_flag", "duplicate_uuid")

_DEFAULTS_CACHE: Optional[Tuple[Dict[str, type], Dict[str, tuple]]] = None


def _default_contract() -> Tuple[Dict[str, type], Dict[str, tuple]]:
    """Weather schema and ranges, resolved from the registry and cached.

    Imported lazily so this module stays free of import-time dependencies.
    """
    global _DEFAULTS_CACHE
    if _DEFAULTS_CACHE is None:
        from src.ingestion.pipeline_registry import registry

        pipeline = registry.require_pipeline(registry.DEFAULT_PIPELINE_ID)
        _DEFAULTS_CACHE = (dict(pipeline.expected_schema), dict(pipeline.valid_ranges))
    return _DEFAULTS_CACHE


def __getattr__(name: str):
    """Expose DEFAULT_EXPECTED_FIELDS / DEFAULT_VALID_RANGES for compatibility."""
    if name == "DEFAULT_EXPECTED_FIELDS":
        return _default_contract()[0]
    if name == "DEFAULT_VALID_RANGES":
        return _default_contract()[1]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class FeatureExtractor:
    """Extracts a numeric feature vector from a JSON payload."""

    FEATURE_NAMES = (
        "total_keys",
        "missing_expected_count",
        "unexpected_extra_count",
        "type_mismatch_count",
        "null_val_count",
        "max_key_similarity",
        "outlier_count",
        "duplicate_flag",
        "string_type_count",
        "bool_type_count",
    )

    def extract_features(
        self,
        payload: Dict[str, Any],
        expected_fields: Optional[Dict[str, type]] = None,
        valid_ranges: Optional[Dict[str, tuple]] = None,
    ) -> List[float]:
        """Vectorise one payload into FEATURE_NAMES order."""
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (TypeError, ValueError):
                payload = {}
        if not isinstance(payload, dict):
            payload = {}

        if expected_fields is None or valid_ranges is None:
            default_fields, default_ranges = _default_contract()
            expected_fields = default_fields if expected_fields is None else expected_fields
            valid_ranges = default_ranges if valid_ranges is None else valid_ranges

        keys = set(payload)
        expected_keys = set(expected_fields)
        missing_keys = expected_keys - keys
        extra_keys = keys - expected_keys

        type_mismatches = 0
        null_count = 0
        outliers = 0
        string_count = 0
        bool_count = 0

        for key, value in payload.items():
            declared = expected_fields.get(key)

            if value is None:
                null_count += 1
            elif isinstance(value, bool):
                # Checked before int/float: bool is a subclass of int.
                bool_count += 1
                if declared is not None and declared is not bool:
                    type_mismatches += 1
            elif isinstance(value, str):
                string_count += 1
                if declared in (float, int):
                    type_mismatches += 1
            elif isinstance(value, (int, float)):
                if declared is str:
                    type_mismatches += 1
                bounds = valid_ranges.get(key)
                if bounds is not None and not (bounds[0] <= value <= bounds[1]):
                    outliers += 1

        # Best string similarity between a missing field and an unexpected
        # key; a high score suggests a rename rather than a true omission.
        max_similarity = 0.0
        for extra in extra_keys:
            for missing in missing_keys:
                ratio = difflib.SequenceMatcher(None, extra, missing).ratio()
                if ratio > max_similarity:
                    max_similarity = ratio

        duplicate_flag = 1.0 if any(marker in payload for marker in DUPLICATE_MARKER_KEYS) else 0.0

        return [
            float(len(keys)),
            float(len(missing_keys)),
            float(len(extra_keys)),
            float(type_mismatches),
            float(null_count),
            float(max_similarity),
            float(outliers),
            float(duplicate_flag),
            float(string_count),
            float(bool_count),
        ]

    def extract_feature_dict(
        self,
        payload: Dict[str, Any],
        expected_fields: Optional[Dict[str, type]] = None,
        valid_ranges: Optional[Dict[str, tuple]] = None,
    ) -> Dict[str, float]:
        """Same as extract_features, keyed by feature name."""
        values = self.extract_features(payload, expected_fields, valid_ranges)
        return dict(zip(self.FEATURE_NAMES, values))
