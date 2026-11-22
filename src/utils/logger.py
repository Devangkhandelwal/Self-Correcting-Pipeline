"""Console + rotating-file logging and the structured audit trail.

Fixes over the previous version:
  * a file handler actually exists (the old comment claimed one but only a
    StreamHandler was installed);
  * LOG_LEVEL from the environment is honoured instead of hardcoded INFO;
  * the audit log rotates instead of growing without bound;
  * audit tailing uses a bounded deque instead of readlines(), which loaded
    the entire file into memory to return the last 30 entries;
  * timestamps are timezone-aware instead of naive datetime.now();
  * audit events log a compact summary rather than dumping full payloads at
    INFO on every single event.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List

from src.utils import config

config.LOG_DIR.mkdir(parents=True, exist_ok=True)

_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"

logger = logging.getLogger("SelfHealingPipeline")
logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

if not logger.handlers:
    _console = logging.StreamHandler()
    _console.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(_console)

    try:
        _file = RotatingFileHandler(
            config.LOG_DIR / "pipeline.log",
            maxBytes=config.AUDIT_LOG_MAX_BYTES,
            backupCount=config.AUDIT_LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        _file.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(_file)
    except OSError as exc:  # pragma: no cover - e.g. read-only filesystem
        logger.warning("File logging unavailable (%s); continuing with console only.", exc)

# A dedicated logger owns the JSONL audit trail so rotation is handled by the
# logging machinery rather than by appending to a file by hand.
_audit_logger = logging.getLogger("SelfHealingPipeline.audit")
_audit_logger.setLevel(logging.INFO)
_audit_logger.propagate = False

if not _audit_logger.handlers:
    try:
        _audit_handler = RotatingFileHandler(
            config.AUDIT_LOG_PATH,
            maxBytes=config.AUDIT_LOG_MAX_BYTES,
            backupCount=config.AUDIT_LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        _audit_handler.setFormatter(logging.Formatter("%(message)s"))
        _audit_logger.addHandler(_audit_handler)
    except OSError as exc:  # pragma: no cover
        logger.error("Audit log unavailable (%s); audit events will not persist.", exc)


def _summarise(details: Dict[str, Any]) -> str:
    """Build a short human-readable line instead of dumping the full payload."""
    interesting = ("pipeline_id", "anomaly_category", "affected_field",
                   "attempt", "passed", "success_rate", "strategy")
    parts = [f"{key}={details[key]}" for key in interesting if key in details]
    return " ".join(parts) if parts else f"{len(details)} field(s)"


def log_audit_event(event_type: str, details: Dict[str, Any]) -> None:
    """Append a structured JSON audit event to the rotating audit trail."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "details": details,
    }
    logger.info("AUDIT [%s]: %s", event_type, _summarise(details))
    logger.debug("AUDIT [%s] full details: %s", event_type, details)
    try:
        _audit_logger.info(json.dumps(entry, default=str))
    except (TypeError, ValueError) as exc:
        logger.error("Failed to serialise audit event %s: %s", event_type, exc)


def get_recent_audit_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """Return up to `limit` most recent audit events, newest last.

    Reads with a bounded deque so a large audit file is streamed rather than
    loaded into memory in full.
    """
    limit = max(1, min(int(limit), config.AUDIT_LOG_MAX_LIMIT))

    if not config.AUDIT_LOG_PATH.exists():
        return []

    tail: deque[str] = deque(maxlen=limit)
    try:
        with open(config.AUDIT_LOG_PATH, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    tail.append(line)
    except OSError as exc:
        logger.error("Error reading audit log: %s", exc)
        return []

    events: List[Dict[str, Any]] = []
    for line in tail:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # A torn final line during rotation is expected; skip it.
            continue
    return events
