"""Weather ingestion client.

This used to carry its own copy of the Open-Meteo fetch, URL, schema and
offline fallback -- duplicating WeatherPipeline in pipeline_registry, with the
two drifting apart. It is now a thin facade over that single definition, kept
because the CLI demo and tests use this entrypoint.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.ingestion.pipeline_registry import WeatherPipeline

#: Canonical weather schema. Re-exported for backward compatibility; the
#: definition lives on WeatherPipeline.
EXPECTED_SCHEMA = WeatherPipeline().expected_schema


class WeatherAPIClient:
    """Fetches clean weather payloads from the Open-Meteo API."""

    def __init__(self, lat: Optional[float] = None, lon: Optional[float] = None):
        self._pipeline = WeatherPipeline(latitude=lat, longitude=lon)

    @property
    def last_fetch_source(self) -> str:
        """Provenance of the most recent fetch: 'live' or 'cached_baseline'."""
        return self._pipeline.last_fetch_source

    @property
    def valid_ranges(self) -> Dict[str, tuple]:
        return self._pipeline.valid_ranges

    def fetch_live_sample(self) -> Dict[str, Any]:
        """Fetch one weather observation, falling back to a cached baseline."""
        return self._pipeline.fetch_clean_sample()

    def get_expected_schema_spec(self) -> Dict[str, str]:
        return self._pipeline.get_schema_spec()
