import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, List

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
AUDIT_LOG_FILE = os.path.join(LOG_DIR, "pipeline_audit.jsonl")

# Setup console & file logging
logger = logging.getLogger("SelfHealingPipeline")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def log_audit_event(event_type: str, details: Dict[str, Any]):
    """Logs structured JSON audit events to file for dashboard and audit review."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "details": details
    }
    logger.info(f"AUDIT [{event_type}]: {json.dumps(details)}")
    try:
        with open(AUDIT_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to write audit log: {e}")

def get_recent_audit_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves recent audit log events."""
    if not os.path.exists(AUDIT_LOG_FILE):
        return []
    logs = []
    try:
        with open(AUDIT_LOG_FILE, "r") as f:
            lines = f.readlines()
            for line in lines[-limit:]:
                if line.strip():
                    logs.append(json.loads(line))
    except Exception as e:
        logger.error(f"Error reading audit logs: {e}")
    return logs
