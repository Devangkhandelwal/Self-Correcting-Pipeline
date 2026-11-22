"""Layer 1: supervised anomaly detection and anomaly-type classification.

Fixes over the previous version:
  * the predicted class is read from `model.classes_` and bounds-checked
    instead of indexing ANOMALY_CLASSES with a raw prediction, which silently
    mis-labels if a class is absent from the training data;
  * confidence is the probability of the predicted class specifically, rather
    than whichever probability happened to be the maximum;
  * the rule-based baseline is public API (the evaluator used to reach into
    `_rule_based_fallback`) and orders its checks by signal specificity;
  * `is_trained` lets callers and the health endpoint tell a loaded model
    from the heuristic fallback, which was previously indistinguishable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from src.ingestion.dedup import DuplicateTracker
from src.ingestion.injector import ANOMALY_CLASSES
from src.layer1_detector.feature_extractor import FeatureExtractor
from src.utils import config
from src.utils.logger import logger

MODEL_PATH = config.MODEL_PATH


class MLAnomalyDetector:
    """Classifies an ingestion payload as normal or as a specific anomaly."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        duplicate_tracker: Optional[DuplicateTracker] = None,
    ):
        self.extractor = FeatureExtractor()
        self.model = None
        self.model_path = Path(model_path) if model_path else config.MODEL_PATH
        # Duplicate detection is inherently stateful: it needs to remember
        # what has already flowed through the stream.
        self.duplicate_tracker = (
            duplicate_tracker if duplicate_tracker is not None else DuplicateTracker()
        )
        self.load_model()

    @property
    def is_trained(self) -> bool:
        """True when a trained model is driving predictions."""
        return self.model is not None

    def load_model(self) -> None:
        if not self.model_path.exists():
            logger.info(
                "No trained model at %s; using rule-based heuristics. "
                "Run run_data_and_train.py to train Layer 1.",
                self.model_path,
            )
            return
        try:
            self.model = joblib.load(self.model_path)
            logger.info("Loaded Layer 1 anomaly model from %s", self.model_path)
        except Exception as exc:
            logger.warning(
                "Could not load model %s (%s: %s); falling back to rule-based heuristics.",
                self.model_path, type(exc).__name__, exc,
            )
            self.model = None

    def detect(
        self,
        payload: Dict[str, Any],
        expected_fields: Optional[Dict[str, type]] = None,
        valid_ranges: Optional[Dict[str, tuple]] = None,
        track_duplicates: bool = True,
        duplicate_tracker: Optional[DuplicateTracker] = None,
    ) -> Dict[str, Any]:
        """Classify one payload.

        Returns is_anomaly, anomaly_type, confidence, the feature vector, the
        per-class probabilities when a model is loaded, and which engine
        produced the verdict.

        Observing the payload updates the duplicate tracker, so calling
        this twice on the same record reports the second as a duplicate --
        correct streaming behaviour. Pass track_duplicates=False to
        classify without recording, or `duplicate_tracker` to score
        against a specific stream's state instead of this detector's.
        """
        # Identity check, not truthiness: an empty tracker is falsy because
        # DuplicateTracker defines __len__, so `or` would discard it.
        tracker = (
            duplicate_tracker if duplicate_tracker is not None else self.duplicate_tracker
        )
        features = self.extractor.extract_features(
            payload,
            expected_fields=expected_fields,
            valid_ranges=valid_ranges,
            duplicate_tracker=tracker if track_duplicates else None,
        )
        feature_dict = dict(zip(FeatureExtractor.FEATURE_NAMES, features))
        return self.classify(feature_dict)

    def observe(self, payload: Dict[str, Any]) -> bool:
        """Register a record as having flowed through the stream.

        Used to seed the tracker when a record is known to have been seen
        earlier -- for instance when simulating a replayed payload.
        """
        return self.duplicate_tracker.observe(payload)

    def classify(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        """Classify a precomputed feature vector.

        Separated from detect() so a caller can extract features once and
        score them with both the model and the rule baseline; the evaluator
        previously extracted twice per row.
        """
        features = [feature_dict[name] for name in FeatureExtractor.FEATURE_NAMES]

        if self.model is None:
            result = self.rule_based_predict(feature_dict)
            result["engine"] = "rule_based"
            return result

        try:
            probabilities = self.model.predict_proba(np.array([features]))[0]
            classes = list(self.model.classes_)
            best = int(np.argmax(probabilities))
            class_index = int(classes[best])
            confidence = float(probabilities[best])
        except Exception as exc:
            logger.warning(
                "Model inference failed (%s: %s); falling back to rule-based heuristics.",
                type(exc).__name__, exc,
            )
            result = self.rule_based_predict(feature_dict)
            result["engine"] = "rule_based"
            return result

        if not 0 <= class_index < len(ANOMALY_CLASSES):
            logger.warning(
                "Model predicted out-of-range class index %d; treating as unknown.",
                class_index,
            )
            predicted = "unknown"
        else:
            predicted = ANOMALY_CLASSES[class_index]

        return {
            "is_anomaly": predicted != "normal",
            "anomaly_type": predicted,
            "confidence": round(confidence, 4),
            "features": feature_dict,
            "class_probabilities": {
                ANOMALY_CLASSES[int(cls)]: round(float(prob), 4)
                for cls, prob in zip(classes, probabilities)
                if 0 <= int(cls) < len(ANOMALY_CLASSES)
            },
            "engine": "ml_model",
        }

    def rule_based_predict(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        """Heuristic baseline, used when untrained and as an evaluation control.

        Checks run from most to least specific signal: an exact duplicate
        marker beats a rename inference, which beats a type error, and so on.
        """
        missing = feature_dict["missing_expected_count"]
        extra = feature_dict["unexpected_extra_count"]
        similarity = feature_dict["max_key_similarity"]
        type_mismatch = feature_dict["type_mismatch_count"]
        nulls = feature_dict["null_val_count"]
        outliers = feature_dict["outlier_count"]
        duplicate = feature_dict["duplicate_flag"]

        def verdict(anomaly_type: str, confidence: float) -> Dict[str, Any]:
            return {
                "is_anomaly": anomaly_type != "normal",
                "anomaly_type": anomaly_type,
                "confidence": confidence,
                "features": feature_dict,
            }

        if duplicate > 0:
            return verdict("duplicate_record", 0.99)
        if missing > 0 and extra > 0 and similarity > config.KEY_SIMILARITY_THRESHOLD:
            return verdict("schema_drift", 0.85)
        if type_mismatch > 0:
            return verdict("type_mismatch", 0.90)
        if outliers > 0:
            return verdict("outlier", 0.95)
        if nulls > 0 or missing > 0:
            return verdict("null_field", 0.88)
        return verdict("normal", 0.95)

    def _rule_based_fallback(self, features: List[float], feature_dict: Dict[str, float]) -> Dict[str, Any]:
        """Deprecated alias retained for backward compatibility."""
        return self.rule_based_predict(feature_dict)
