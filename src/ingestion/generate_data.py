"""Synthetic dataset and held-out verification batch generation.

Fixes over the previous version:
  * generation is seeded, so a dataset is reproducible;
  * every record is no longer derived from a single API sample -- latitude,
    longitude and timestamps now vary, where before all 500 held-out records
    shared identical values and verification was near-meaningless;
  * held-out batches are generated per pipeline, not weather-only, so the
    other three domains have real verification data;
  * batches include corrupted records alongside clean ones, so a patch is
    tested against the fault it exists to fix.
"""
from __future__ import annotations

import json
import random
from typing import Any, Dict, List

import pandas as pd

from src.ingestion.injector import ANOMALY_CLASSES, SyntheticAnomalyInjector
from src.ingestion.pipeline_registry import registry
from src.utils import config
from src.utils.logger import logger

config.SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
config.HELDOUT_DIR.mkdir(parents=True, exist_ok=True)

HELDOUT_BATCHES = 10
HELDOUT_BATCH_SIZE = 50
#: Share of each held-out batch that carries an injected anomaly.
HELDOUT_CORRUPT_FRACTION = 0.3


def _jitter_weather(base: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Produce a varied-but-valid weather record from a baseline sample."""
    hour = rng.randint(0, 23)
    return {
        "latitude": round(base.get("latitude", 40.7128) + rng.uniform(-0.25, 0.25), 6),
        "longitude": round(base.get("longitude", -74.0060) + rng.uniform(-0.25, 0.25), 6),
        "temperature_2m": round(rng.uniform(-8.0, 38.0), 2),
        "relative_humidity_2m": round(rng.uniform(15.0, 100.0), 2),
        "precipitation": round(max(0.0, rng.gauss(0.6, 1.2)), 2),
        "surface_pressure": round(rng.uniform(985.0, 1038.0), 2),
        "wind_speed_10m": round(max(0.0, rng.gauss(11.0, 5.0)), 2),
        "time": f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T{hour:02d}:00",
    }


def _sample_for(pipeline_id: str, base: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Draw one clean record for a pipeline, with variation."""
    if pipeline_id == "weather":
        return _jitter_weather(base, rng)
    # The simulated pipelines already randomise each call.
    return registry.require_pipeline(pipeline_id).fetch_clean_sample()


def generate_datasets(
    num_samples: int = 1200,
    seed: int = None,
    pipeline_id: str = "weather",
) -> Dict[str, Any]:
    """Generate the Layer 1 training dataset and held-out batches.

    Returns a summary dict describing what was written.
    """
    seed = config.RANDOM_SEED if seed is None else seed
    rng = random.Random(seed)
    injector = SyntheticAnomalyInjector(rng=rng)

    pipeline = registry.require_pipeline(pipeline_id)
    base_sample = pipeline.fetch_clean_sample()
    logger.info(
        "Generating %d samples for '%s' (seed=%d, source=%s)",
        num_samples, pipeline_id, seed, pipeline.last_fetch_source,
    )

    records: List[Dict[str, Any]] = []
    for index in range(num_samples):
        sample = _sample_for(pipeline_id, base_sample, rng)
        anomaly_type = ANOMALY_CLASSES[index % len(ANOMALY_CLASSES)]
        corrupted, label, meta = injector.inject_anomaly(
            sample,
            target_anomaly=anomaly_type,
            field_renames=pipeline.field_renames,
            valid_ranges=pipeline.valid_ranges,
        )
        records.append({
            "id": index,
            "payload_json": json.dumps(corrupted),
            "label": label,
            "metadata_json": json.dumps(meta),
        })

    frame = pd.DataFrame(records)
    frame.to_csv(config.DATASET_PATH, index=False)
    logger.info("Wrote %d training samples to %s", len(frame), config.DATASET_PATH)

    batches_written = _write_heldout_batches(rng, injector)

    return {
        "dataset_path": str(config.DATASET_PATH),
        "samples": len(frame),
        "seed": seed,
        "heldout_batches": batches_written,
    }


def _write_heldout_batches(rng: random.Random, injector: SyntheticAnomalyInjector) -> int:
    """Write held-out verification batches for every registered pipeline."""
    written = 0
    pipeline_ids = registry.pipeline_ids()

    for batch_id in range(HELDOUT_BATCHES):
        # Rotate pipelines across batches so every domain gets coverage.
        target_id = pipeline_ids[batch_id % len(pipeline_ids)]
        pipeline = registry.require_pipeline(target_id)
        base = pipeline.fetch_clean_sample()

        batch: List[Dict[str, Any]] = []
        corrupt_quota = int(HELDOUT_BATCH_SIZE * HELDOUT_CORRUPT_FRACTION)

        for position in range(HELDOUT_BATCH_SIZE):
            sample = _sample_for(target_id, base, rng)
            if position < corrupt_quota:
                anomaly = ANOMALY_CLASSES[position % len(ANOMALY_CLASSES)]
                sample, _, _ = injector.inject_anomaly(
                    sample,
                    target_anomaly=anomaly,
                    field_renames=pipeline.field_renames,
                    valid_ranges=pipeline.valid_ranges,
                )
            batch.append(sample)

        rng.shuffle(batch)
        path = config.HELDOUT_DIR / f"heldout_{target_id}_batch_{batch_id}.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(batch, handle, indent=2)
        written += 1

    logger.info("Wrote %d held-out verification batches to %s", written, config.HELDOUT_DIR)
    return written


if __name__ == "__main__":
    generate_datasets()
