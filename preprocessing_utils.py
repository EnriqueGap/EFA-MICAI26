from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


@dataclass(frozen=True)
class FeatureGroups:
    numeric: list[str]
    binary_categorical: list[str]
    multiclass_categorical: list[str]


@dataclass
class PreparedDataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series | np.ndarray
    y_test: pd.Series | np.ndarray
    feature_groups: FeatureGroups
    preprocessor: ColumnTransformer
    preprocessing_pipeline: Pipeline
    X_train_processed: Any
    X_test_processed: Any


def _ensure_dataframe(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()

    array = np.asarray(X)
    if array.ndim != 2:
        raise ValueError("X must be two-dimensional to build a preprocessing pipeline.")

    feature_names = [f"feature_{index}" for index in range(array.shape[1])]
    return pd.DataFrame(array, columns=feature_names)


def infer_feature_groups(X: pd.DataFrame | np.ndarray) -> FeatureGroups:
    X_df = _ensure_dataframe(X)

    numeric_features = X_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in X_df.columns if column not in numeric_features]

    binary_categorical = []
    multiclass_categorical = []

    for column in categorical_features:
        cardinality = X_df[column].nunique(dropna=True)
        if cardinality <= 2:
            binary_categorical.append(column)
        else:
            multiclass_categorical.append(column)

    return FeatureGroups(
        numeric=numeric_features,
        binary_categorical=binary_categorical,
        multiclass_categorical=multiclass_categorical,
    )


def build_preprocessor(
    X: pd.DataFrame | np.ndarray,
    *,
    feature_groups: FeatureGroups | None = None,
) -> ColumnTransformer:
    feature_groups = feature_groups or infer_feature_groups(X)

    transformers: list[tuple[str, Any, list[str]]] = []

    if feature_groups.numeric:
        transformers.append(("num", StandardScaler(), feature_groups.numeric))

    if feature_groups.binary_categorical:
        transformers.append(
            (
                "bin_cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                feature_groups.binary_categorical,
            )
        )

    if feature_groups.multiclass_categorical:
        transformers.append(
            (
                "multi_cat",
                OneHotEncoder(handle_unknown="ignore"),
                feature_groups.multiclass_categorical,
            )
        )

    if not transformers:
        raise ValueError("No features were found to build the preprocessing pipeline.")

    return ColumnTransformer(transformers=transformers)


def prepare_dataset(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> PreparedDataset:
    X_df = _ensure_dataframe(X)
    stratify_target = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X_df,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    feature_groups = infer_feature_groups(X_train)
    preprocessor = build_preprocessor(X_train, feature_groups=feature_groups)

    preprocessing_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
        ]
    )

    X_train_processed = preprocessing_pipeline.fit_transform(X_train)
    X_test_processed = preprocessing_pipeline.transform(X_test)

    return PreparedDataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_groups=feature_groups,
        preprocessor=preprocessor,
        preprocessing_pipeline=preprocessing_pipeline,
        X_train_processed=X_train_processed,
        X_test_processed=X_test_processed,
    )
