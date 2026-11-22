"""Layer 1 evaluation on held-out data only.

The previous evaluator scored the model over the entire dataset -- including
the 80% it was fitted on -- so its "benchmark" was not a measure of
generalisation. It also extracted features twice per row and reached into the
detector's private `_rule_based_fallback`.

This version scores the persisted test split, falls back to the full dataset
only when no split file exists (and says so in the result), and reports the
feature-leakage caveat alongside the numbers so the metric cannot be read as
stronger than it is.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.ingestion.dedup import DuplicateTracker
from src.ingestion.injector import ANOMALY_CLASSES
from src.layer1_detector.feature_extractor import FeatureExtractor
from src.layer1_detector.model import MLAnomalyDetector
from src.utils import config
from src.utils.logger import logger

#: Features previously determined by injector marker keys rather than by
#: record content. Now empty: `duplicate_flag` is computed from a content
#: fingerprint (see src/ingestion/dedup.py), so no feature encodes a label.
LEAKY_FEATURES = ()


#: Below this F1 difference the trained model is treated as adding nothing.
LIFT_SIGNIFICANCE = 0.01


def _lift_verdict(f1_lift: float) -> str:
    """State plainly whether the trained model is earning its place."""
    if f1_lift > LIFT_SIGNIFICANCE:
        return (
            "The model is outperforming the rules, so it is carrying its weight "
            "on this dataset: the engineered features leave genuine ambiguity "
            "that a learned decision boundary resolves better than fixed "
            "thresholds do."
        )
    if f1_lift < -LIFT_SIGNIFICANCE:
        return (
            "The model is performing WORSE than the handwritten rules. Prefer the "
            "rule baseline until this is understood."
        )
    return (
        "A lift at or near zero means the model is not earning its place on this "
        "dataset: the engineered features already encode the anomaly definitions "
        "the rules test directly, so there is nothing left to learn. Try "
        "--difficulty hard for a benchmark with genuine ambiguity."
    )


def _load_split() -> Optional[Dict[str, Any]]:
    if not config.SPLIT_PATH.exists():
        return None
    try:
        with open(config.SPLIT_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read split file %s: %s", config.SPLIT_PATH, exc)
        return None


def evaluate_layer1(detector: Optional[MLAnomalyDetector] = None) -> Dict[str, Any]:
    """Score the Layer 1 model and its rule baseline on held-out rows."""
    if not config.DATASET_PATH.exists():
        logger.error("Dataset missing at %s", config.DATASET_PATH)
        return {}

    frame = pd.read_csv(config.DATASET_PATH)
    detector = detector or MLAnomalyDetector()

    split = _load_split()
    if split and split.get("test_row_ids"):
        test_ids = set(split["test_row_ids"])
        evaluation_frame = frame[frame["id"].isin(test_ids)]
        split_note = (
            f"Scored on the {len(evaluation_frame)}-row held-out test split "
            f"(seed {split.get('seed')}); the model never saw these rows."
        )
        held_out = True
    else:
        evaluation_frame = frame
        split_note = (
            "No persisted train/test split found, so these figures include rows "
            "the model was trained on and OVERSTATE generalisation. "
            "Re-run run_data_and_train.py to produce an honest split."
        )
        held_out = False
        logger.warning(split_note)

    labels: List[str] = []
    model_predictions: List[str] = []
    baseline_predictions: List[str] = []

    # The whole dataset is replayed in file order through one tracker so
    # the content-based duplicate_flag matches what a live stream would
    # see, but only the evaluation rows are scored. Scoring the test rows
    # in isolation would miss every duplicate whose original sits in the
    # training split.
    tracker = DuplicateTracker()
    scored_ids = set(evaluation_frame["id"].tolist())

    for _, row in frame.iterrows():
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, ValueError):
            continue

        # Extract once, with the tracker, and score both engines on it.
        feature_dict = detector.extractor.extract_feature_dict(
            payload, duplicate_tracker=tracker
        )

        true_label = row["label"]
        if row["id"] not in scored_ids or true_label not in ANOMALY_CLASSES:
            continue

        labels.append(true_label)
        model_predictions.append(detector.classify(feature_dict)["anomaly_type"])
        baseline_predictions.append(
            detector.rule_based_predict(feature_dict)["anomaly_type"]
        )

    if not labels:
        logger.error("No evaluable rows found in %s", config.DATASET_PATH)
        return {}

    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(
        labels, model_predictions, average="macro", zero_division=0
    )
    base_precision, base_recall, base_f1, _ = precision_recall_fscore_support(
        labels, baseline_predictions, average="macro", zero_division=0
    )
    matrix = confusion_matrix(labels, model_predictions, labels=list(ANOMALY_CLASSES))

    # How much the trained model actually buys over the handwritten rules.
    # On the current synthetic benchmark this is zero: the two agree on every
    # row, because the features were engineered to encode the same anomaly
    # definitions the rules test directly.
    agreements = sum(
        1 for model_pred, base_pred in zip(model_predictions, baseline_predictions)
        if model_pred == base_pred
    )
    agreement_rate = agreements / len(labels)
    f1_lift = float(model_f1) - float(base_f1)

    metrics = {
        "ml_model": {
            "precision": round(float(model_precision), 4),
            "recall": round(float(model_recall), 4),
            "f1_score": round(float(model_f1), 4),
        },
        "rule_baseline": {
            "precision": round(float(base_precision), 4),
            "recall": round(float(base_recall), 4),
            "f1_score": round(float(base_f1), 4),
        },
        "classes": list(ANOMALY_CLASSES),
        "confusion_matrix": matrix.tolist(),
        "evaluated_rows": len(labels),
        "held_out": held_out,
        "split_note": split_note,
        "model_vs_baseline": {
            "f1_lift": round(f1_lift, 4),
            "agreement_rate": round(agreement_rate, 4),
            "verdict": _lift_verdict(f1_lift),
            "note": (
                "The trained model and the rule baseline agree on "
                f"{agreement_rate * 100:.1f}% of held-out rows for an F1 lift of "
                f"{f1_lift:+.4f}. {_lift_verdict(f1_lift)}"
            ),
        },
        "engine": "ml_model" if detector.is_trained else "rule_based",
        "caveat": (
            "Scores come from synthetic data whose anomalies are generated by "
            "the same schema definitions the features are computed against, so "
            "the classes are far more separable here than in real traffic. No "
            "feature encodes a label any more -- duplicate_flag is derived from "
            "a content fingerprint rather than injector marker keys -- but a "
            "high score remains a property of this benchmark, not evidence of "
            "real-world accuracy."
        ),
    }

    logger.info(
        "Layer 1 model -> precision %.4f, recall %.4f, F1 %.4f (%d rows, held_out=%s)",
        model_precision, model_recall, model_f1, len(labels), held_out,
    )
    logger.info(
        "Rule baseline -> precision %.4f, recall %.4f, F1 %.4f",
        base_precision, base_recall, base_f1,
    )
    return metrics


if __name__ == "__main__":
    evaluate_layer1()
