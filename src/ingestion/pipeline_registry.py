"""Definitions of every ingestion pipeline domain, and the registry over them.

Two behavioural fixes here:
  * `get_pipeline` no longer silently falls back to the weather pipeline for
    an unknown id. A typo used to return confident weather results for, say,
    a finance request; callers now get None and can reject the input.
  * the live weather fetch no longer swallows every exception and returns a
    hardcoded payload as though it were live data. Network failures are
    narrowed, logged, and the resulting record is tagged with its provenance.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.utils import config
from src.utils.logger import logger

#: Provenance values reported via BasePipeline.last_fetch_source.
SOURCE_LIVE = "live"
SOURCE_CACHED = "cached_baseline"
SOURCE_SIMULATED = "simulated"


class BasePipeline:
    """Base definition for a data pipeline domain."""

    def __init__(
        self,
        pipeline_id: str,
        name: str,
        icon: str,
        description: str,
        expected_schema: Dict[str, type],
        valid_ranges: Dict[str, Tuple[float, float]],
        field_renames: Dict[str, str],
    ):
        self.pipeline_id = pipeline_id
        self.name = name
        self.icon = icon
        self.description = description
        self.expected_schema = expected_schema
        self.valid_ranges = valid_ranges
        self.field_renames = field_renames
        # Provenance of the most recent fetch_clean_sample() call. Held
        # out-of-band rather than injected into the payload, which would show
        # up as an unexpected extra key in the Layer 1 feature vector.
        self.last_fetch_source: str = SOURCE_SIMULATED

    def get_schema_spec(self) -> Dict[str, str]:
        """Schema as JSON-serialisable type names, e.g. {'amount': 'float'}."""
        return {field: declared.__name__ for field, declared in self.expected_schema.items()}

    def fetch_clean_sample(self) -> Dict[str, Any]:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {
            "id": self.pipeline_id,
            "name": self.name,
            "icon": self.icon,
            "description": self.description,
            "schema": self.get_schema_spec(),
            "valid_ranges": {k: list(v) for k, v in self.valid_ranges.items()},
        }


class WeatherPipeline(BasePipeline):
    """Live meteorological observations from the Open-Meteo API."""

    CURRENT_FIELDS = (
        "temperature_2m,relative_humidity_2m,precipitation,"
        "surface_pressure,wind_speed_10m"
    )

    def __init__(self, latitude: Optional[float] = None, longitude: Optional[float] = None):
        super().__init__(
            pipeline_id="weather",
            name="Weather Forecast API",
            icon="🌤️",
            description="Meteorological observation metrics from the Open-Meteo API.",
            expected_schema={
                "latitude": float,
                "longitude": float,
                "temperature_2m": float,
                "relative_humidity_2m": float,
                "precipitation": float,
                "surface_pressure": float,
                "wind_speed_10m": float,
                "time": str,
            },
            valid_ranges={
                "latitude": (-90.0, 90.0),
                "longitude": (-180.0, 180.0),
                "temperature_2m": (-60.0, 70.0),
                "relative_humidity_2m": (0.0, 100.0),
                "precipitation": (0.0, 500.0),
                "surface_pressure": (800.0, 1200.0),
                "wind_speed_10m": (0.0, 300.0),
            },
            field_renames={
                "temperature_2m": "temp_2m",
                "relative_humidity_2m": "humidity",
                "precipitation": "precip",
                "surface_pressure": "pressure",
                "wind_speed_10m": "wind_speed",
            },
        )
        self.latitude = latitude if latitude is not None else config.WEATHER_LATITUDE
        self.longitude = longitude if longitude is not None else config.WEATHER_LONGITUDE

    def fetch_clean_sample(self) -> Dict[str, Any]:
        """Fetch one live observation, falling back to a cached baseline.

        Provenance is recorded on `last_fetch_source` so callers can tell the
        two apart. Previously any exception -- including a genuine bug in the
        parsing below -- returned a hardcoded literal indistinguishable from
        real data, with nothing logged.
        """
        try:
            response = requests.get(
                config.WEATHER_API_URL,
                params={
                    "latitude": self.latitude,
                    "longitude": self.longitude,
                    "current": self.CURRENT_FIELDS,
                },
                timeout=config.HTTP_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            body = response.json()
            current = body.get("current") or {}

            sample = {
                "latitude": float(body.get("latitude", self.latitude)),
                "longitude": float(body.get("longitude", self.longitude)),
                "temperature_2m": float(current["temperature_2m"]),
                "relative_humidity_2m": float(current["relative_humidity_2m"]),
                "precipitation": float(current["precipitation"]),
                "surface_pressure": float(current["surface_pressure"]),
                "wind_speed_10m": float(current["wind_speed_10m"]),
                "time": str(current.get("time", "")),
            }
            self.last_fetch_source = SOURCE_LIVE
            return sample

        except requests.RequestException as exc:
            logger.warning(
                "Open-Meteo request failed (%s: %s); using cached baseline sample.",
                type(exc).__name__, exc,
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "Open-Meteo returned an unexpected shape (%s: %s); using cached baseline.",
                type(exc).__name__, exc,
            )

        return self.cached_baseline()

    def cached_baseline(self) -> Dict[str, Any]:
        """A known-good offline record; provenance recorded out-of-band."""
        self.last_fetch_source = SOURCE_CACHED
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "temperature_2m": 22.4,
            "relative_humidity_2m": 65.0,
            "precipitation": 0.0,
            "surface_pressure": 1013.2,
            "wind_speed_10m": 12.5,
            "time": "1970-01-01T00:00",
        }


class FinancePipeline(BasePipeline):
    """Simulated payment-processing and fraud-scoring feed."""

    def __init__(self):
        super().__init__(
            pipeline_id="finance",
            name="Financial Transactions API",
            icon="💳",
            description="Real-time payment processing & fraud scoring feed.",
            expected_schema={
                "transaction_id": str,
                "user_id": str,
                "amount": float,
                "currency": str,
                "merchant_id": str,
                "risk_score": float,
                "timestamp": str,
            },
            valid_ranges={
                "amount": (0.01, 50000.0),
                "risk_score": (0.0, 1.0),
            },
            field_renames={
                "transaction_id": "txn_id",
                "merchant_id": "merchant",
                "risk_score": "fraud_score",
                "amount": "tx_amount",
            },
        )

    def fetch_clean_sample(self) -> Dict[str, Any]:
        self.last_fetch_source = SOURCE_SIMULATED
        return {
            "transaction_id": f"TXN-{random.randint(10000, 99999)}",
            "user_id": f"USR-{random.randint(100, 999)}",
            "amount": round(random.uniform(15.50, 450.00), 2),
            "currency": "USD",
            "merchant_id": f"MERCH-{random.randint(10, 99)}",
            "risk_score": round(random.uniform(0.01, 0.15), 4),
            "timestamp": "1970-01-01T00:00:00Z",
        }


class ECommercePipeline(BasePipeline):
    """Simulated warehouse stock movement and reorder feed."""

    def __init__(self):
        super().__init__(
            pipeline_id="ecommerce",
            name="E-Commerce Inventory API",
            icon="📦",
            description="Warehouse stock movement and automated reorder payload.",
            expected_schema={
                "item_id": str,
                "sku": str,
                "stock_quantity": float,
                "unit_price": float,
                "category": str,
                "reorder_level": float,
                "timestamp": str,
            },
            valid_ranges={
                "stock_quantity": (0.0, 10000.0),
                "unit_price": (0.50, 5000.0),
                "reorder_level": (1.0, 500.0),
            },
            field_renames={
                "stock_quantity": "qty_on_hand",
                "unit_price": "price",
                "reorder_level": "min_stock",
                "category": "item_cat",
            },
        )

    def fetch_clean_sample(self) -> Dict[str, Any]:
        self.last_fetch_source = SOURCE_SIMULATED
        return {
            "item_id": f"ITEM-{random.randint(1000, 9999)}",
            "sku": f"SKU-ELECTRONICS-{random.randint(100, 999)}",
            "stock_quantity": float(random.randint(150, 800)),
            "unit_price": round(random.uniform(29.99, 199.99), 2),
            "category": "Electronics",
            "reorder_level": 50.0,
            "timestamp": "1970-01-01T00:00:00Z",
        }


class IoTPipeline(BasePipeline):
    """Simulated industrial turbine telemetry stream."""

    def __init__(self):
        super().__init__(
            pipeline_id="iot",
            name="Industrial IoT Telemetry API",
            icon="⚙️",
            description="Factory turbine sensor stream monitoring vibration & thermal stress.",
            expected_schema={
                "device_id": str,
                "vibration_rms": float,
                "temperature_c": float,
                "pressure_psi": float,
                "motor_rpm": float,
                "battery_pct": float,
                "timestamp": str,
            },
            valid_ranges={
                "vibration_rms": (0.0, 15.0),
                "temperature_c": (-20.0, 120.0),
                "pressure_psi": (0.0, 300.0),
                "motor_rpm": (0.0, 6000.0),
                "battery_pct": (0.0, 100.0),
            },
            field_renames={
                "vibration_rms": "vib_rms",
                "temperature_c": "temp_celsius",
                "pressure_psi": "press_psi",
                "motor_rpm": "rpm",
            },
        )

    def fetch_clean_sample(self) -> Dict[str, Any]:
        self.last_fetch_source = SOURCE_SIMULATED
        return {
            "device_id": f"TURBINE-{random.randint(1, 12):02d}",
            "vibration_rms": round(random.uniform(1.2, 3.8), 2),
            "temperature_c": round(random.uniform(45.0, 78.5), 1),
            "pressure_psi": round(random.uniform(115.0, 145.0), 1),
            "motor_rpm": float(random.randint(2800, 3400)),
            "battery_pct": round(random.uniform(85.0, 99.0), 1),
            "timestamp": "1970-01-01T00:00:00Z",
        }


class PipelineRegistry:
    """Registry managing all active data pipeline definitions."""

    DEFAULT_PIPELINE_ID = "weather"

    def __init__(self):
        self.pipelines: Dict[str, BasePipeline] = {
            pipeline.pipeline_id: pipeline
            for pipeline in (
                WeatherPipeline(),
                FinancePipeline(),
                ECommercePipeline(),
                IoTPipeline(),
            )
        }

    def get_pipeline(self, pipeline_id: str) -> Optional[BasePipeline]:
        """Return the named pipeline, or None if it is not registered."""
        return self.pipelines.get(pipeline_id)

    def require_pipeline(self, pipeline_id: str) -> BasePipeline:
        """Return the named pipeline, raising KeyError if it is unknown."""
        pipeline = self.pipelines.get(pipeline_id)
        if pipeline is None:
            raise KeyError(
                f"Unknown pipeline_id '{pipeline_id}'. "
                f"Known ids: {', '.join(self.pipeline_ids())}"
            )
        return pipeline

    def pipeline_ids(self) -> List[str]:
        return list(self.pipelines)

    def list_pipelines(self) -> List[Dict[str, Any]]:
        return [pipeline.describe() for pipeline in self.pipelines.values()]


registry = PipelineRegistry()
