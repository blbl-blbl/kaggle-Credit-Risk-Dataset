# evaluate.py

from pathlib import Path
from src.predict import predict
from src.config import PROJECT_ROOT

import json
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    log_loss,
    brier_score_loss,
    recall_score,
    precision_score,
    confusion_matrix
)

def evaluate_model(X, y, artifact):
    result = predict(
        X,
        artifact
    )

    probabilities = result['probability']
    predictions = result['prediction']

    metrics = {
        'roc_auc': roc_auc_score(y, probabilities),
        'pr_auc': average_precision_score(y, probabilities),
        'log_loss': log_loss(y, probabilities),
        'brier_score': brier_score_loss(y, probabilities),

        'precision': precision_score(y, predictions,
                                     zero_division=0),
        'recall': recall_score(y, predictions,
                               zero_division=0),
        'f1': f1_score(y, predictions,
                       zero_division=0),
    }

    return metrics

def get_confusion_matrix(X, y, artifact):
    result = predict(
        X,
        artifact
    )

    predictions = result['prediction']

    return confusion_matrix(y, predictions).tolist()

def build_report(
        X, y,
        artifact,
        dataset_name
):
    return {
        "dataset": dataset_name,
        "metrics": evaluate_model(X, y, artifact),
        "confusion_matrix": get_confusion_matrix(X, y, artifact)
    }

def save_report_json(
        report,
        path=PROJECT_ROOT / "reports" / "metrics.json"
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
