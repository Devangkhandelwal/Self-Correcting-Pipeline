"""Synthetic anomaly injection.

Deliberately corrupts clean payloads to build labelled training data and to
simulate live upstream failures.

Changes from the previous version: the injector accepts an injectable RNG so
dataset generation is reproducible, no longer mutates shared default state,
and skips its own bookkeeping keys when choosing a field to corrupt.
"""
from __future__ import annotations

import copy
import random
from typing import Any, Dict, Optional, Tuple

ANOMALY_CLASSES = (
    "normal",
    "schema_drift",
    "type_mismatch",
    "null_field",
    "outlier",
    "duplicate_record",
)

DEFAULT_FIELD_RENAMES = {
    "temperature_2m": "temp_2m",
    "relative_humidity_2m": "humidity",
    "precipitation": "precip",
    "surface_pressure": "pressure",
    "wind_speed_10m": "wind_speed",
}

#: Keys the injector itself adds; never chosen as a corruption target.
RESERVED_KEYS = ("_is_duplicate_flag", "duplicate_uuid")


class SyntheticAnomalyInjector:
    """Corrupts clean API payloads into labelled anomalous records."""

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()

    def _candidate_fields(self, payload: Dict[str, Any]) -> list:
        return [
            key for key in payload
            if not key.startswith("_") and key not in RESERVED_KEYS
        ]

    def _numeric_fields(self, payload: Dict[str, Any]) -> list:
        return [
            key for key, value in payload.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and key not in RESERVED_KEYS
        ]

    def inject_anomaly(
        self,
        payload: Dict[str, Any],
        target_anomaly: Optional[str] = None,
        field_renames: Optional[Dict[str, str]] = None,
        valid_ranges: Optional[Dict[str, tuple]] = None,
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
        """Inject one anomaly.

        Returns (corrupted_payload, anomaly_label, metadata).
        """
        if target_anomaly is None:
            target_anomaly = self.rng.choice(ANOMALY_CLASSES)

        renames = DEFAULT_FIELD_RENAMES if field_renames is None else field_renames
        corrupted = copy.deepcopy(payload)
        metadata: Dict[str, Any] = {
            "original_payload": copy.deepcopy(payload),
            "applied_anomaly": target_anomaly,
        }

        handler = {
            "normal": self._inject_normal,
            "schema_drift": self._inject_schema_drift,
            "type_mismatch": self._inject_type_mismatch,
            "null_field": self._inject_null_field,
            "outlier": self._inject_outlier,
            "duplicate_record": self._inject_duplicate,
        }.get(target_anomaly)

        if handler is None:
            raise ValueError(
                f"Unknown anomaly type '{target_anomaly}'. "
                f"Valid types: {', '.join(ANOMALY_CLASSES)}"
            )

        handler(corrupted, metadata, renames, valid_ranges or {})
        return corrupted, target_anomaly, metadata

    # -- individual injectors ------------------------------------------------

    def _inject_normal(self, corrupted, metadata, renames, valid_ranges) -> None:
        """A clean control record; intentionally unchanged."""

    def _inject_schema_drift(self, corrupted, metadata, renames, valid_ranges) -> None:
        candidates = [field for field in renames if field in corrupted]
        if not candidates:
            candidates = self._candidate_fields(corrupted)
        if not candidates:
            return

        field = self.rng.choice(candidates)
        new_field = renames.get(field, f"{field}_renamed")
        corrupted[new_field] = corrupted.pop(field)
        metadata["renamed_from"] = field
        metadata["renamed_to"] = new_field

    def _inject_type_mismatch(self, corrupted, metadata, renames, valid_ranges) -> None:
        numeric = self._numeric_fields(corrupted)
        if not numeric:
            return

        field = self.rng.choice(numeric)
        value = corrupted[field]
        corrupt_type = self.rng.choice(["string_number", "string_text", "bool"])

        if corrupt_type == "string_number":
            corrupted[field] = str(value)
        elif corrupt_type == "string_text":
            corrupted[field] = "INVALID_NUMERIC"
        else:
            corrupted[field] = True

        metadata["field"] = field
        metadata["corrupt_type"] = corrupt_type

    def _inject_null_field(self, corrupted, metadata, renames, valid_ranges) -> None:
        candidates = self._candidate_fields(corrupted)
        if not candidates:
            return

        field = self.rng.choice(candidates)
        action = self.rng.choice(["set_null", "delete_key"])
        if action == "set_null":
            corrupted[field] = None
        else:
            corrupted.pop(field, None)

        metadata["field"] = field
        metadata["action"] = action

    def _inject_outlier(self, corrupted, metadata, renames, valid_ranges) -> None:
        numeric = self._numeric_fields(corrupted)
        if not numeric:
            return

        # Prefer a field with a declared range so the outlier is unambiguous.
        ranged = [field for field in numeric if field in valid_ranges]
        field = self.rng.choice(ranged or numeric)
        value = float(corrupted[field])

        if field in valid_ranges:
            high = float(valid_ranges[field][1])
            outlier = high * 10.0 + 999.0
        else:
            outlier = (value * 100.0) + 9999.0 if value >= 0 else (value * 100.0) - 9999.0

        corrupted[field] = round(outlier, 2)
        metadata["field"] = field
        metadata["outlier_val"] = corrupted[field]

    def _inject_duplicate(self, corrupted, metadata, renames, valid_ranges) -> None:
        corrupted["_is_duplicate_flag"] = True
        corrupted["duplicate_uuid"] = f"DUP-{self.rng.randint(10000, 99999)}"
        metadata["action"] = "duplicate_inserted"
