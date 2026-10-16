import random
import requests
from typing import Dict, Any, List

class BasePipeline:
    """Base definition for a data pipeline domain."""
    def __init__(
        self,
        pipeline_id: str,
        name: str,
        icon: str,
        description: str,
        expected_schema: Dict[str, type],
        valid_ranges: Dict[str, tuple],
        field_renames: Dict[str, str]
    ):
        self.pipeline_id = pipeline_id
        self.name = name
        self.icon = icon
        self.description = description
        self.expected_schema = expected_schema
        self.valid_ranges = valid_ranges
        self.field_renames = field_renames

    def get_schema_spec(self) -> Dict[str, str]:
        return {k: v.__name__ for k, v in self.expected_schema.items()}

    def fetch_clean_sample(self) -> Dict[str, Any]:
        raise NotImplementedError


class WeatherPipeline(BasePipeline):
    def __init__(self):
        super().__init__(
            pipeline_id="weather",
            name="Weather Forecast API",
            icon="🌤️",
            description="Meteorological observation metrics from Open-Meteo API.",
            expected_schema={
                "latitude": float,
                "longitude": float,
                "temperature_2m": float,
                "relative_humidity_2m": float,
                "precipitation": float,
                "surface_pressure": float,
                "wind_speed_10m": float,
                "time": str
            },
            valid_ranges={
                "temperature_2m": (-60.0, 70.0),
                "relative_humidity_2m": (0.0, 100.0),
                "precipitation": (0.0, 500.0),
                "surface_pressure": (800.0, 1200.0),
                "wind_speed_10m": (0.0, 300.0)
            },
            field_renames={
                "temperature_2m": "temp_2m",
                "relative_humidity_2m": "humidity",
                "precipitation": "precip",
                "surface_pressure": "pressure",
                "wind_speed_10m": "wind_speed"
            }
        )

    def fetch_clean_sample(self) -> Dict[str, Any]:
        try:
            url = "https://api.open-meteo.com/v1/forecast?latitude=40.7128&longitude=-74.0060&current=temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m"
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            current = data.get("current", {})
            return {
                "latitude": data.get("latitude", 40.7128),
                "longitude": data.get("longitude", -74.0060),
                "temperature_2m": float(current.get("temperature_2m", 22.4)),
                "relative_humidity_2m": float(current.get("relative_humidity_2m", 65.0)),
                "precipitation": float(current.get("precipitation", 0.0)),
                "surface_pressure": float(current.get("surface_pressure", 1013.2)),
                "wind_speed_10m": float(current.get("wind_speed_10m", 12.5)),
                "time": str(current.get("time", "2026-08-22T12:00"))
            }
        except Exception:
            return {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "temperature_2m": 22.4,
                "relative_humidity_2m": 65.0,
                "precipitation": 0.0,
                "surface_pressure": 1013.2,
                "wind_speed_10m": 12.5,
                "time": "2026-08-22T12:00"
            }


class FinancePipeline(BasePipeline):
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
                "timestamp": str
            },
            valid_ranges={
                "amount": (0.01, 50000.0),
                "risk_score": (0.0, 1.0)
            },
            field_renames={
                "transaction_id": "txn_id",
                "merchant_id": "merchant",
                "risk_score": "fraud_score",
                "amount": "tx_amount"
            }
        )

    def fetch_clean_sample(self) -> Dict[str, Any]:
        txn_num = random.randint(10000, 99999)
        return {
            "transaction_id": f"TXN-{txn_num}",
            "user_id": f"USR-{random.randint(100, 999)}",
            "amount": round(random.uniform(15.50, 450.00), 2),
            "currency": "USD",
            "merchant_id": f"MERCH-{random.randint(10, 99)}",
            "risk_score": round(random.uniform(0.01, 0.15), 4),
            "timestamp": "2026-08-22T14:55:00Z"
        }


class ECommercePipeline(BasePipeline):
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
                "timestamp": str
            },
            valid_ranges={
                "stock_quantity": (0.0, 10000.0),
                "unit_price": (0.50, 5000.0),
                "reorder_level": (1.0, 500.0)
            },
            field_renames={
                "stock_quantity": "qty_on_hand",
                "unit_price": "price",
                "reorder_level": "min_stock",
                "category": "item_cat"
            }
        )

    def fetch_clean_sample(self) -> Dict[str, Any]:
        return {
            "item_id": f"ITEM-{random.randint(1000, 9999)}",
            "sku": "SKU-ELECTRONICS-882",
            "stock_quantity": float(random.randint(150, 800)),
            "unit_price": round(random.uniform(29.99, 199.99), 2),
            "category": "Electronics",
            "reorder_level": 50.0,
            "timestamp": "2026-08-22T14:55:00Z"
        }


class IoTPipeline(BasePipeline):
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
                "timestamp": str
            },
            valid_ranges={
                "vibration_rms": (0.0, 15.0),
                "temperature_c": (-20.0, 120.0),
                "pressure_psi": (0.0, 300.0),
                "motor_rpm": (0.0, 6000.0),
                "battery_pct": (0.0, 100.0)
            },
            field_renames={
                "vibration_rms": "vib_rms",
                "temperature_c": "temp_celsius",
                "pressure_psi": "press_psi",
                "motor_rpm": "rpm"
            }
        )

    def fetch_clean_sample(self) -> Dict[str, Any]:
        return {
            "device_id": f"TURBINE-{random.randint(1, 12):02d}",
            "vibration_rms": round(random.uniform(1.2, 3.8), 2),
            "temperature_c": round(random.uniform(45.0, 78.5), 1),
            "pressure_psi": round(random.uniform(115.0, 145.0), 1),
            "motor_rpm": float(random.randint(2800, 3400)),
            "battery_pct": round(random.uniform(85.0, 99.0), 1),
            "timestamp": "2026-08-22T14:55:00Z"
        }


class PipelineRegistry:
    """Registry managing all active data pipeline definitions."""
    def __init__(self):
        self.pipelines: Dict[str, BasePipeline] = {
            "weather": WeatherPipeline(),
            "finance": FinancePipeline(),
            "ecommerce": ECommercePipeline(),
            "iot": IoTPipeline()
        }

    def get_pipeline(self, pipeline_id: str) -> BasePipeline:
        return self.pipelines.get(pipeline_id, self.pipelines["weather"])

    def list_pipelines(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": p.pipeline_id,
                "name": p.name,
                "icon": p.icon,
                "description": p.description,
                "schema": p.get_schema_spec()
            }
            for p in self.pipelines.values()
        ]

registry = PipelineRegistry()
