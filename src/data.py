# data.py

import pandas as pd
from sklearn.model_selection import train_test_split
from config import (FEATURES, TARGET,
                    RANDOM_SEED)


def load_data(path):
    return pd.read_excel(path)

def validate_columns(df):
    required_columns = FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError (
            f"Missing columns are: {missing_columns}"
        )

def split_features(df):
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    return X, y

def train_val_cal_split(X, y):
    X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.1,
                                                        stratify=y, random_state=RANDOM_SEED)

    return X_train, y_train, X_cal, y_cal