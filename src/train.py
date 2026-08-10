# train.py

import json
from pathlib import Path
import joblib

from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.preprocessing import build_preprocessor


def load_metadata(path="../models/metadata.json"):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_base_model(X, y, metadata):

    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    model_params = metadata['params'].copy()

    model = CatBoostClassifier(
        **model_params,
        cat_features=metadata["cat_features"],
        random_seed=metadata['random_seed'],
        verbose=False
    )

    model.fit(X_processed, y)

    return preprocessor, model


def calibrate_model(model, preprocessor,
                    X_cal, y_cal,
                    method='isotonic'):

    X_cal_processed = preprocessor.transform(X_cal)

    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(model),
        method=method
    )

    calibrated_model.fit(X_cal_processed, y_cal)

    return calibrated_model


def save_artifact(
        preprocessor,
        calibrated_model,
        threshold,
        path='models/credit_risk_model.joblib'
):

    artifact = {
        'preprocessor': preprocessor,
        'model': calibrated_model,
        'threshold': threshold
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact, path)


def train(
        X_train, y_train,
        X_cal, y_cal
):

    metadata = load_metadata()

    preprocessor, model = build_base_model(
        X_train,
        y_train,
        metadata
    )

    calibrated_model = calibrate_model(
        model,
        preprocessor,
        X_cal, y_cal,
        metadata['calibration']
    )

    threshold = metadata['threshold']

    save_artifact(
        preprocessor,
        calibrated_model,
        threshold
    )