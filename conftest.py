"""Pytest configuration.

Puts the project root on sys.path, and keeps the test suite off the network.
"""
import os
import sys

import pytest

root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


@pytest.fixture(autouse=True)
def offline_weather(monkeypatch):
    """Serve the weather pipeline from its cached baseline during tests.

    `WeatherPipeline.fetch_clean_sample()` otherwise makes a live HTTP request
    to Open-Meteo. That made the suite depend on a third-party service being
    reachable -- slow, flaky, and failing outright offline, with the weather
    cases dominating the runtime.

    The live path is exercised by `demo_pipeline.py` and the running API
    instead, where a real fetch is the point.
    """
    from src.ingestion.pipeline_registry import WeatherPipeline

    monkeypatch.setattr(
        WeatherPipeline,
        "fetch_clean_sample",
        WeatherPipeline.cached_baseline,
        raising=True,
    )
