"""Centralised configuration, read once from the environment.

Everything that used to be a magic number scattered through the codebase
lives here. Values come from the environment, optionally seeded from a
`.env` file at the project root (see `.env.example`).

Previously `.env.example` documented PORT and LOG_LEVEL but nothing read a
`.env` file at all, so the documented setup path silently did nothing --
including the GEMINI_API_KEY that Layer 2's LLM path depends on.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
HELDOUT_DIR = DATA_DIR / "heldout"
PATCH_DIR = DATA_DIR / "patches"
LOG_DIR = PROJECT_ROOT / "logs"

DATASET_PATH = SYNTHETIC_DIR / "dataset.csv"
MODEL_PATH = DATA_DIR / "model.joblib"
SPLIT_PATH = DATA_DIR / "train_test_split.json"
AUDIT_LOG_PATH = LOG_DIR / "pipeline_audit.jsonl"


def _load_dotenv() -> None:
    """Seed os.environ from .env if python-dotenv is available.

    Kept optional so the project still runs if the dependency is absent;
    real environment variables always win over .env values.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - optional dependency
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


_load_dotenv()


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# -- Secrets -----------------------------------------------------------------
# Never hardcoded; absent key simply disables the LLM path.
GEMINI_API_KEY = _env_str("GEMINI_API_KEY", "") or _env_str("GOOGLE_API_KEY", "")
GEMINI_MODEL = _env_str("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TIMEOUT_SECONDS = _env_float("GEMINI_TIMEOUT_SECONDS", 20.0)

# -- HTTP server -------------------------------------------------------------
# Defaults are deliberately safe: loopback only, debugger off. The previous
# default was host 0.0.0.0 with debug=True, which exposes the Werkzeug
# console (remote code execution) to anything that can reach the port.
API_HOST = _env_str("API_HOST", "127.0.0.1")
API_PORT = _env_int("PORT", 5000)
API_DEBUG = _env_bool("API_DEBUG", False)
CORS_ORIGINS = [
    origin.strip()
    for origin in _env_str("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if origin.strip()
]

# -- Logging -----------------------------------------------------------------
LOG_LEVEL = _env_str("LOG_LEVEL", "INFO").upper()
AUDIT_LOG_MAX_BYTES = _env_int("AUDIT_LOG_MAX_BYTES", 5 * 1024 * 1024)
AUDIT_LOG_BACKUP_COUNT = _env_int("AUDIT_LOG_BACKUP_COUNT", 3)
AUDIT_LOG_MAX_LIMIT = _env_int("AUDIT_LOG_MAX_LIMIT", 500)

# -- Upstream ingestion ------------------------------------------------------
WEATHER_API_URL = _env_str("WEATHER_API_URL", "https://api.open-meteo.com/v1/forecast")
WEATHER_LATITUDE = _env_float("WEATHER_LATITUDE", 40.7128)
WEATHER_LONGITUDE = _env_float("WEATHER_LONGITUDE", -74.0060)
HTTP_TIMEOUT_SECONDS = _env_float("HTTP_TIMEOUT_SECONDS", 10.0)

# -- Layer 1 -----------------------------------------------------------------
RANDOM_SEED = _env_int("RANDOM_SEED", 42)
TEST_SPLIT_SIZE = _env_float("TEST_SPLIT_SIZE", 0.2)
# Minimum string similarity before two keys are treated as a rename.
KEY_SIMILARITY_THRESHOLD = _env_float("KEY_SIMILARITY_THRESHOLD", 0.4)

# -- Layer 3 -----------------------------------------------------------------
MAX_HEAL_RETRIES = _env_int("MAX_HEAL_RETRIES", 3)
SANDBOX_TIMEOUT_SECONDS = _env_float("SANDBOX_TIMEOUT_SECONDS", 15.0)
# How many held-out records each verification run is scored against.
VERIFICATION_SAMPLE_SIZE = _env_int("VERIFICATION_SAMPLE_SIZE", 120)
