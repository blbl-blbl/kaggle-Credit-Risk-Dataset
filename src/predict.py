# predict.py

import joblib
import numpy as np
from src.config import PROJECT_ROOT


def load_model(path=PROJECT_ROOT / "models" / "credit_risk_model.joblib"):
    return joblib.load(path)

def predict(X, artifact):
    preprocessor = artifact['preprocessor']
    model = artifact['model']
    threshold = artifact['threshold']

    X_processed = preprocessor.transform(X)

    probabilities = model.predict_proba(X_processed)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return probabilities, predictions