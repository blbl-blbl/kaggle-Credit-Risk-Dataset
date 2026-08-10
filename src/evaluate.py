# evaluate.py

from src.predict import predict

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

        'precision': precision_score(y, predictions),
        'recall': recall_score(y, predictions),
        'f1': f1_score(y, predictions),
    }

    return metrics

def get_confusion_matrix(X, y, artifact):
    result = predict(
        X,
        artifact
    )

    predictions = result['prediction']

    return confusion_matrix(y, predictions)
