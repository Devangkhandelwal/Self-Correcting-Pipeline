import os
import json
import random
import pandas as pd
from src.ingestion.api_client import WeatherAPIClient, EXPECTED_SCHEMA
from src.ingestion.injector import SyntheticAnomalyInjector, ANOMALY_CLASSES
from src.utils.logger import logger

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
SYNTHETIC_DIR = os.path.join(DATA_DIR, "synthetic")
HELDOUT_DIR = os.path.join(DATA_DIR, "heldout")

os.makedirs(SYNTHETIC_DIR, exist_ok=True)
os.makedirs(HELDOUT_DIR, exist_ok=True)

def generate_datasets(num_samples: int = 1200):
    """Generates synthetic dataset for training Layer 1 and held-out evaluation batches."""
    client = WeatherAPIClient()
    base_sample = client.fetch_live_sample()
    injector = SyntheticAnomalyInjector()

    records = []
    heldout_batches = []

    for i in range(num_samples):
        # Slightly fluctuate clean numbers
        sample = copy_and_fluctuate(base_sample)
        # Pick anomaly class evenly
        anomaly_type = ANOMALY_CLASSES[i % len(ANOMALY_CLASSES)]
        corrupted, label, meta = injector.inject_anomaly(sample, target_anomaly=anomaly_type)
        
        record = {
            "id": i,
            "payload_json": json.dumps(corrupted),
            "label": label,
            "metadata_json": json.dumps(meta)
        }
        records.append(record)

    # Save to synthetic CSV
    df = pd.DataFrame(records)
    train_path = os.path.join(SYNTHETIC_DIR, "dataset.csv")
    df.to_csv(train_path, index=False)
    logger.info(f"Generated {len(df)} synthetic anomaly samples saved to {train_path}")

    # Generate held-out test batches for Layer 3 verification loop
    for batch_id in range(10):
        batch = []
        for _ in range(50):
            sample = copy_and_fluctuate(base_sample)
            batch.append(sample)
        batch_path = os.path.join(HELDOUT_DIR, f"heldout_batch_{batch_id}.json")
        with open(batch_path, "w") as f:
            json.dump(batch, f, indent=2)
    logger.info(f"Generated 10 held-out verification batches in {HELDOUT_DIR}")

def copy_and_fluctuate(base: dict) -> dict:
    d = dict(base)
    d["temperature_2m"] = round(float(d.get("temperature_2m", 20.0)) + random.uniform(-3.0, 3.0), 2)
    d["relative_humidity_2m"] = max(0.0, min(100.0, round(float(d.get("relative_humidity_2m", 60.0)) + random.uniform(-5.0, 5.0), 2)))
    d["precipitation"] = max(0.0, round(random.uniform(0.0, 2.0), 2))
    d["surface_pressure"] = round(1013.2 + random.uniform(-10.0, 10.0), 2)
    d["wind_speed_10m"] = round(max(0.0, 10.0 + random.uniform(-4.0, 4.0)), 2)
    return d

if __name__ == "__main__":
    generate_datasets()
