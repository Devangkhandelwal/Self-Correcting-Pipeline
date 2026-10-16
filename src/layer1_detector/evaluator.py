import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from src.layer1_detector.model import MLAnomalyDetector
from src.layer1_detector.feature_extractor import FeatureExtractor
from src.ingestion.injector import ANOMALY_CLASSES
from src.utils.logger import logger

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "synthetic", "dataset.csv")

def evaluate_layer1() -> dict:
    """Evaluates Layer 1 ML model precision, recall, F1, and confusion matrix vs baseline."""
    if not os.path.exists(DATA_PATH):
        logger.error(f"Dataset missing at {DATA_PATH}")
        return {}

    df = pd.read_csv(DATA_PATH)
    detector = MLAnomalyDetector()

    y_true = []
    y_pred_ml = []
    y_pred_baseline = []

    label_to_idx = {name: idx for idx, name in enumerate(ANOMALY_CLASSES)}

    for _, row in df.iterrows():
        payload = json.loads(row["payload_json"])
        true_label = row["label"]
        if true_label not in label_to_idx:
            continue

        y_true.append(true_label)

        # ML model prediction
        res_ml = detector.detect(payload)
        y_pred_ml.append(res_ml["anomaly_type"])

        # Simple rule baseline prediction
        res_base = detector._rule_based_fallback(
            detector.extractor.extract_features(payload),
            dict(zip(FeatureExtractor.FEATURE_NAMES, detector.extractor.extract_features(payload)))
        )
        y_pred_baseline.append(res_base["anomaly_type"])

    p_ml, r_ml, f1_ml, _ = precision_recall_fscore_support(y_true, y_pred_ml, average="macro", zero_division=0)
    p_base, r_base, f1_base, _ = precision_recall_fscore_support(y_true, y_pred_baseline, average="macro", zero_division=0)

    cm_ml = confusion_matrix(y_true, y_pred_ml, labels=ANOMALY_CLASSES)

    metrics = {
        "ml_model": {
            "precision": round(float(p_ml), 4),
            "recall": round(float(r_ml), 4),
            "f1_score": round(float(f1_ml), 4)
        },
        "rule_baseline": {
            "precision": round(float(p_base), 4),
            "recall": round(float(r_base), 4),
            "f1_score": round(float(f1_base), 4)
        },
        "classes": ANOMALY_CLASSES,
        "confusion_matrix": cm_ml.tolist()
    }

    logger.info(f"Layer 1 ML Model -> Precision: {p_ml:.4f}, Recall: {r_ml:.4f}, F1: {f1_ml:.4f}")
    logger.info(f"Rule Baseline  -> Precision: {p_base:.4f}, Recall: {r_base:.4f}, F1: {f1_base:.4f}")

    return metrics

if __name__ == "__main__":
    evaluate_layer1()
