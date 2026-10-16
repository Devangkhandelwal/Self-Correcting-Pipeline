import os
import joblib
import numpy as np
from typing import Dict, Any, Tuple, Optional
from src.layer1_detector.feature_extractor import FeatureExtractor
from src.ingestion.injector import ANOMALY_CLASSES
from src.utils.logger import logger

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "model.joblib")

class MLAnomalyDetector:
    """Layer 1: Supervised ML Model for Anomaly Detection and Anomaly Type Classification."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.extractor = FeatureExtractor()
        self.model = None
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                logger.info(f"Loaded Layer 1 ML Anomaly Model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Could not load model file: {e}. Will rely on rule-based fallback until trained.")
        else:
            logger.info("No pre-trained model found. Initializing fallback heuristics.")

    def detect(
        self,
        payload: Dict[str, Any],
        expected_fields: Optional[Dict[str, type]] = None,
        valid_ranges: Optional[Dict[str, tuple]] = None
    ) -> Dict[str, Any]:
        """
        Runs Layer 1 anomaly detection on incoming payload.
        Returns:
        {
            "is_anomaly": bool,
            "anomaly_type": str,
            "confidence": float,
            "features": Dict[str, float]
        }
        """
        features = self.extractor.extract_features(
            payload,
            expected_fields=expected_fields,
            valid_ranges=valid_ranges
        )
        feature_dict = dict(zip(FeatureExtractor.FEATURE_NAMES, features))

        if self.model is not None:
            X = np.array([features])
            pred_idx = self.model.predict(X)[0]
            probs = self.model.predict_proba(X)[0]
            predicted_class = ANOMALY_CLASSES[pred_idx]
            confidence = float(np.max(probs))
            is_anomaly = (predicted_class != "normal")
            return {
                "is_anomaly": is_anomaly,
                "anomaly_type": predicted_class,
                "confidence": round(confidence, 4),
                "features": feature_dict
            }

        # Rule-based fallback if ML model not yet trained
        return self._rule_based_fallback(features, feature_dict)

    def _rule_based_fallback(self, features: list, feature_dict: dict) -> dict:
        missing = feature_dict["missing_expected_count"]
        extra = feature_dict["unexpected_extra_count"]
        sim = feature_dict["max_key_similarity"]
        type_mismatch = feature_dict["type_mismatch_count"]
        nulls = feature_dict["null_val_count"]
        outliers = feature_dict["outlier_count"]
        dup = feature_dict["duplicate_flag"]

        if missing > 0 and extra > 0 and sim > 0.4:
            return {"is_anomaly": True, "anomaly_type": "schema_drift", "confidence": 0.85, "features": feature_dict}
        elif type_mismatch > 0:
            return {"is_anomaly": True, "anomaly_type": "type_mismatch", "confidence": 0.90, "features": feature_dict}
        elif nulls > 0 or missing > 0:
            return {"is_anomaly": True, "anomaly_type": "null_field", "confidence": 0.88, "features": feature_dict}
        elif outliers > 0:
            return {"is_anomaly": True, "anomaly_type": "outlier", "confidence": 0.95, "features": feature_dict}
        elif dup > 0:
            return {"is_anomaly": True, "anomaly_type": "duplicate_record", "confidence": 0.99, "features": feature_dict}
        
        return {"is_anomaly": False, "anomaly_type": "normal", "confidence": 0.95, "features": feature_dict}
