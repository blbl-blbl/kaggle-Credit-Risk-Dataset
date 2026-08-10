# predict.py

import joblib
from src.config import PROJECT_ROOT
import pandas as pd


def load_artifact(path=PROJECT_ROOT / "models" / "credit_risk_model.joblib"):
    return joblib.load(path)

def predict(X, artifact):
    preprocessor = artifact['preprocessor']
    model = artifact['model']
    threshold = artifact['threshold']

    X_processed = preprocessor.transform(X)

    probabilities = model.predict_proba(X_processed)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    result = pd.DataFrame({
        "probability": probabilities,
        "prediction": predictions
    })

    return result