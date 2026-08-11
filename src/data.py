# data.py

import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import (FEATURES, TARGET,
                    RANDOM_SEED, TARGET_MAPPING)


def load_data(path):
    return pd.read_excel(path)


def validate_columns(df):
    required_columns = FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError (
            f"Missing columns are: {missing_columns}"
        )

    return df[required_columns].copy()


def prepare_train_target(df):
    valid_values = set(TARGET_MAPPING)

    actual_values = set(df[TARGET].dropna().unique())
    invalid_values = actual_values - valid_values

    if invalid_values:
        raise ValueError (
            f"Unknown target values in train data {invalid_values}"
        )

    if df[TARGET].isna().any():
        raise ValueError (
            f"Target contains missing values in train data"
        )

    df = df.copy()
    df[TARGET] = (df[TARGET].map(TARGET_MAPPING).astype(int))

    return df


def prepare_test_target(df):
    valid_values = set(TARGET_MAPPING)

    invalid_mask = ~df[TARGET].isin(valid_values)
    invalid_count = invalid_mask.sum()

    if invalid_count > 0:
        print("Unknown target in the test")
        print(df.loc[invalid_mask, TARGET].value_counts(dropna=False))
        print(
            f"Dropping {invalid_count} rows "
            f"with unknown target values"
        )
    df = df.loc[~invalid_mask].copy()
    df[TARGET] = (df[TARGET].map(TARGET_MAPPING).astype(int))

    return df


def split_features(df):
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    return X, y


def train_cal_split(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=RANDOM_SEED
    )

    X_val, X_cal, y_val, y_cal = train_test_split(
        X_temp,
        y_temp,
        test_size=0.33,
        stratify=y_temp,
        random_state=RANDOM_SEED
    )

    X_train_final = pd.concat(
        [X_train, X_val]
    )

    y_train_final = pd.concat(
        [y_train, y_val]
    )

    return (
        X_train_final,
        y_train_final,
        X_cal,
        y_cal
    )


def prepare_train_data(path):
    data = load_data(path)
    data_validated = validate_columns(data)
    data_cleaned = prepare_train_target(data_validated)
    X, y = split_features(data_cleaned)
    X_train, y_train, X_cal, y_cal = train_cal_split(X, y)

    return X_train, y_train, X_cal, y_cal


def prepare_test_data(path):
    data = load_data(path)
    data_validated = validate_columns(data)
    data_cleaned = prepare_test_target(data_validated)
    X, y = split_features(data_cleaned)
    return X, y
