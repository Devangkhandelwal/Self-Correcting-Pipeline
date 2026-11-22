"""Content-based duplicate detection.

Replaces a circular arrangement. The injector used to mark a replayed record
by adding `_is_duplicate_flag` / `duplicate_uuid`, and the feature extractor
set its `duplicate_flag` feature from the presence of those exact keys. For
the duplicate class the feature therefore *was* the label: the classifier was
reading a marker the test harness had planted, not detecting anything.

Here a duplicate is what a duplicate actually is -- a record whose content has
been seen before. That requires state across records, which is why this lives
in the ingestion stage rather than in a stateless parser: `parse_payload`
sees one record at a time and cannot, even in principle, know it is a replay.

Fingerprints cover only schema-relevant data. Bookkeeping keys are excluded so
that a record does not change identity merely by being annotated.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any, Dict, Iterable, Optional

#: Keys excluded from a fingerprint: annotations, not data.
EXCLUDED_KEYS = frozenset({
    "_healing", "_is_duplicate", "_is_duplicate_flag", "duplicate_uuid", "_source",
})

DEFAULT_CAPACITY = 2048


def fingerprint(payload: Dict[str, Any], exclude: Optional[Iterable[str]] = None) -> str:
    """Stable content hash of a payload.

    Key order does not affect the result, so two records carrying the same
    data fingerprint identically regardless of serialisation order.
    """
    excluded = EXCLUDED_KEYS if exclude is None else frozenset(exclude) | EXCLUDED_KEYS
    material = {key: value for key, value in payload.items() if key not in excluded}
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class DuplicateTracker:
    """Bounded memory of recently seen record fingerprints.

    Capacity is bounded so a long-running pipeline cannot grow this without
    limit; the oldest fingerprint is evicted first. A duplicate separated from
    its original by more than `capacity` records is therefore not detected,
    which is the normal trade for streaming deduplication.
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self._seen: "OrderedDict[str, int]" = OrderedDict()
        self.observed_count = 0
        self.duplicate_count = 0

    def __len__(self) -> int:
        return len(self._seen)

    def seen(self, payload: Dict[str, Any]) -> bool:
        """Whether this content has been observed, without recording it."""
        return fingerprint(payload) in self._seen

    def observe(self, payload: Dict[str, Any]) -> bool:
        """Record this payload and report whether it is a repeat.

        Returns True when the content was already known.
        """
        digest = fingerprint(payload)
        self.observed_count += 1

        if digest in self._seen:
            self._seen[digest] += 1
            self._seen.move_to_end(digest)
            self.duplicate_count += 1
            return True

        self._seen[digest] = 1
        if len(self._seen) > self.capacity:
            self._seen.popitem(last=False)
        return False

    def reset(self) -> None:
        """Forget everything; used between evaluation runs."""
        self._seen.clear()
        self.observed_count = 0
        self.duplicate_count = 0

    def stats(self) -> Dict[str, int]:
        return {
            "tracked_fingerprints": len(self._seen),
            "observed_count": self.observed_count,
            "duplicate_count": self.duplicate_count,
            "capacity": self.capacity,
        }
