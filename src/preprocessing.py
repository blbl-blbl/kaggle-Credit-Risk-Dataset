# preprocessing.py

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.config import (NUMERIC_FEATURES,
                    CATEGORICAL_FEATURES)


def build_preprocessor():
    # Numeric features preprocessing
    numeric_pipeline_catboost = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=-1))
    ])

    # Categorial features preprocessing
    categorial_pipeline_catboost = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('zero_imputer', SimpleImputer(missing_values=0, strategy='constant', fill_value='missing'))
    ])

    catboost_transformer = ColumnTransformer(transformers=[
        ('numeric', numeric_pipeline_catboost, NUMERIC_FEATURES),
        ('categorical', categorial_pipeline_catboost, CATEGORICAL_FEATURES)
    ])

    cat_features_indices = list(
        range(
            len(NUMERIC_FEATURES),
            len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
        )
    )

    preprocessor_pipeline_catboost = Pipeline(steps=[
        ('preprocessor', catboost_transformer)
    ])

    return preprocessor_pipeline_catboost