import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.layer1_detector.feature_extractor import FeatureExtractor
from src.ingestion.injector import ANOMALY_CLASSES
from src.utils.logger import logger

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "synthetic", "dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "model.joblib")

def train_model():
    """Trains Random Forest classifier for Layer 1 Anomaly Detection."""
    if not os.path.exists(DATA_PATH):
        logger.error(f"Dataset file missing at {DATA_PATH}. Run generate_data.py first.")
        return None

    df = pd.read_csv(DATA_PATH)
    extractor = FeatureExtractor()

    X = []
    y = []

    label_to_idx = {name: idx for idx, name in enumerate(ANOMALY_CLASSES)}

    for _, row in df.iterrows():
        payload = json.loads(row["payload_json"])
        feats = extractor.extract_features(payload)
        label_str = row["label"]
        if label_str in label_to_idx:
            X.append(feats)
            y.append(label_to_idx[label_str])

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    logger.info(f"Model Training Complete! Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    joblib.dump(clf, MODEL_PATH)
    logger.info(f"Saved trained Layer 1 model to {MODEL_PATH}")
    return clf

if __name__ == "__main__":
    train_model()
