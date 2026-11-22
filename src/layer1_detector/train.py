"""Layer 1 training: fit the anomaly classifier and persist its split.

The split indices are written to disk alongside the model so evaluation can
score on data the model never saw. Previously the evaluator re-read the whole
dataset and reported metrics computed over the training rows as well, which
is why the reported figures were not a measurement of generalisation.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.ingestion.injector import ANOMALY_CLASSES
from src.layer1_detector.feature_extractor import FeatureExtractor
from src.utils import config
from src.utils.logger import logger

LABEL_TO_INDEX = {name: index for index, name in enumerate(ANOMALY_CLASSES)}


def build_feature_matrix(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorise a dataset frame into (X, y, row_ids)."""
    extractor = FeatureExtractor()
    features, labels, row_ids = [], [], []

    for _, row in frame.iterrows():
        label = row["label"]
        if label not in LABEL_TO_INDEX:
            continue
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, ValueError) as exc:
            logger.warning("Skipping row %s with unparseable payload: %s", row.get("id"), exc)
            continue

        features.append(extractor.extract_features(payload))
        labels.append(LABEL_TO_INDEX[label])
        row_ids.append(int(row["id"]))

    return np.array(features), np.array(labels), np.array(row_ids)


def train_model(dataset_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Train the Layer 1 classifier; returns a summary or None on failure."""
    path = dataset_path or config.DATASET_PATH
    if not str(path) or not config.DATASET_PATH.exists() and dataset_path is None:
        logger.error(
            "Dataset missing at %s. Run run_data_and_train.py (or "
            "src/ingestion/generate_data.py) first.",
            config.DATASET_PATH,
        )
        return None

    frame = pd.read_csv(path)
    X, y, row_ids = build_feature_matrix(frame)

    if len(X) == 0:
        logger.error("Dataset at %s produced no usable rows.", path)
        return None

    present = sorted(set(y.tolist()))
    missing_classes = [
        ANOMALY_CLASSES[index] for index in range(len(ANOMALY_CLASSES))
        if index not in present
    ]
    if missing_classes:
        logger.warning(
            "Dataset contains no examples of: %s. The model cannot predict them.",
            ", ".join(missing_classes),
        )

    # Stratify only when every class has enough members for the split.
    _, counts = np.unique(y, return_counts=True)
    stratify = y if counts.min() >= 2 else None

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=config.TEST_SPLIT_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=stratify,
    )

    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    classifier.fit(X[train_idx], y[train_idx])

    train_accuracy = float(classifier.score(X[train_idx], y[train_idx]))
    test_accuracy = float(classifier.score(X[test_idx], y[test_idx]))
    logger.info(
        "Training complete. Train accuracy: %.4f, held-out test accuracy: %.4f",
        train_accuracy, test_accuracy,
    )

    joblib.dump(classifier, config.MODEL_PATH)
    logger.info("Saved Layer 1 model to %s", config.MODEL_PATH)

    split = {
        "seed": config.RANDOM_SEED,
        "test_size": config.TEST_SPLIT_SIZE,
        "dataset_path": str(path),
        "dataset_rows": int(len(frame)),
        "feature_names": list(FeatureExtractor.FEATURE_NAMES),
        "train_row_ids": sorted(int(i) for i in row_ids[train_idx]),
        "test_row_ids": sorted(int(i) for i in row_ids[test_idx]),
    }
    with open(config.SPLIT_PATH, "w", encoding="utf-8") as handle:
        json.dump(split, handle, indent=2)
    logger.info("Saved train/test split to %s", config.SPLIT_PATH)

    return {
        "train_accuracy": round(train_accuracy, 4),
        "test_accuracy": round(test_accuracy, 4),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "missing_classes": missing_classes,
    }


if __name__ == "__main__":
    train_model()
