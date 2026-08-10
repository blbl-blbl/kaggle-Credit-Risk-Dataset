# preprocessing.py

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.config import (NUMERIC_FEATURES,
                    CATEGORICAL_FEATURES)


def build_preprocessor():
    # Numeric features preprocessing
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=-1))
    ])

    # Categorial features preprocessing
    categorial_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('zero_imputer', SimpleImputer(missing_values=0, strategy='constant', fill_value='missing'))
    ])

    transformer = ColumnTransformer(transformers=[
        ('numeric', numeric_pipeline, NUMERIC_FEATURES),
        ('categorical', categorial_pipeline, CATEGORICAL_FEATURES)
    ])

    return transformer