from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import optuna
import optuna.visualization as vis
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, recall_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


EstimatorFactory = Callable[[optuna.trial.Trial | None, dict[str, Any] | None, int], BaseEstimator]


@dataclass(frozen=True)
class ModelSpec:
    factory: EstimatorFactory
    contour_params: tuple[str, str] | None = None


@dataclass
class OptimizationArtifacts:
    model_name: str
    study: optuna.Study
    best_pipeline: Pipeline
    best_estimator: BaseEstimator
    contour_params: list[str]
    classes_: np.ndarray
    meta_reference_class: Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _slugify_label(value: Any) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", str(value)).strip("_").lower()
    return slug or "class_label"


def _positive_class_matrix(y_true: Any, classes: np.ndarray) -> np.ndarray:
    y_array = np.asarray(y_true)
    return np.column_stack([(y_array == class_label).astype(int) for class_label in classes])


def _resolve_estimator_classes(estimator: BaseEstimator) -> np.ndarray:
    if hasattr(estimator, "classes_"):
        return np.asarray(estimator.classes_)

    if hasattr(estimator, "named_steps") and "classifier" in estimator.named_steps:
        classifier = estimator.named_steps["classifier"]
        if hasattr(classifier, "classes_"):
            return np.asarray(classifier.classes_)

    raise AttributeError("The fitted estimator does not expose classes_.")


def _build_decision_tree(
    trial: optuna.trial.Trial | None = None,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> DecisionTreeClassifier:
    final_params = dict(params or {})

    if trial is not None:
        final_params.update(
            {
                "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
                "max_depth": trial.suggest_int("max_depth", 2, 32),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            }
        )

    final_params.setdefault("random_state", random_state)
    return DecisionTreeClassifier(**final_params)


def _build_random_forest(
    trial: optuna.trial.Trial | None = None,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> RandomForestClassifier:
    final_params = dict(params or {})

    if trial is not None:
        final_params.update(
            {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
                "max_depth": trial.suggest_int("max_depth", 2, 32),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical(
                    "max_features",
                    ["sqrt", "log2", None],
                ),
            }
        )

    final_params.setdefault("random_state", random_state)
    final_params.setdefault("n_jobs", -1)
    return RandomForestClassifier(**final_params)


def _build_xgboost(
    trial: optuna.trial.Trial | None = None,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> BaseEstimator:
    from xgboost import XGBClassifier

    final_params = dict(params or {})

    if trial is not None:
        final_params.update(
            {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
        )

    final_params.setdefault("random_state", random_state)
    final_params.setdefault("n_jobs", -1)
    final_params.setdefault("eval_metric", "mlogloss")
    return XGBClassifier(**final_params)


def _build_lightgbm(
    trial: optuna.trial.Trial | None = None,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> BaseEstimator:
    from lightgbm import LGBMClassifier

    final_params = dict(params or {})

    if trial is not None:
        final_params.update(
            {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "num_leaves": trial.suggest_int("num_leaves", 15, 255),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 16),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
        )

    final_params.setdefault("random_state", random_state)
    final_params.setdefault("n_jobs", -1)
    return LGBMClassifier(**final_params)


def _build_catboost(
    trial: optuna.trial.Trial | None = None,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> BaseEstimator:
    from catboost import CatBoostClassifier

    final_params = dict(params or {})

    if trial is not None:
        final_params.update(
            {
                "iterations": trial.suggest_int("iterations", 100, 600),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
                "border_count": trial.suggest_int("border_count", 32, 255),
            }
        )

    final_params.setdefault("random_state", random_state)
    final_params.setdefault("allow_writing_files", False)
    final_params.setdefault("verbose", False)
    return CatBoostClassifier(**final_params)


def _build_balanced_random_forest(
    trial: optuna.trial.Trial | None = None,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> BaseEstimator:
    from imblearn.ensemble import BalancedRandomForestClassifier

    final_params = dict(params or {})

    if trial is not None:
        final_params.update(
            {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "max_depth": trial.suggest_int("max_depth", 2, 32),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical(
                    "max_features",
                    ["sqrt", "log2", None],
                ),
                "replacement": trial.suggest_categorical("replacement", [True, False]),
            }
        )

    final_params.setdefault("random_state", random_state)
    final_params.setdefault("n_jobs", -1)
    return BalancedRandomForestClassifier(**final_params)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "decision_tree": ModelSpec(
        factory=_build_decision_tree,
        contour_params=("max_depth", "min_samples_split"),
    ),
    "random_forest": ModelSpec(
        factory=_build_random_forest,
        contour_params=("max_depth", "min_samples_split"),
    ),
    "xgboost": ModelSpec(
        factory=_build_xgboost,
        contour_params=("learning_rate", "max_depth"),
    ),
    "lightgbm": ModelSpec(
        factory=_build_lightgbm,
        contour_params=("learning_rate", "num_leaves"),
    ),
    "catboost": ModelSpec(
        factory=_build_catboost,
        contour_params=("learning_rate", "depth"),
    ),
    "balanced_random_forest": ModelSpec(
        factory=_build_balanced_random_forest,
        contour_params=("max_depth", "min_samples_split"),
    ),
}


def get_available_models() -> list[str]:
    available_models: list[str] = []

    for model_name in MODEL_REGISTRY:
        try:
            build_estimator(model_name=model_name, random_state=42)
        except ImportError:
            continue
        available_models.append(model_name)

    return available_models


def build_estimator(
    model_name: str,
    *,
    trial: optuna.trial.Trial | None = None,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> BaseEstimator:
    normalized_name = model_name.lower()
    if normalized_name not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}.")

    return MODEL_REGISTRY[normalized_name].factory(trial, params, random_state)


def fit_model_pipeline(
    preprocessor: Any,
    model_name: str,
    X_train: Any,
    y_train: Any,
    *,
    model_params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> tuple[Pipeline, BaseEstimator]:
    estimator = build_estimator(
        model_name=model_name,
        params=model_params,
        random_state=random_state,
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("classifier", estimator),
        ]
    )
    model_pipeline.fit(X_train, y_train)
    return model_pipeline, estimator


def compute_class_aucpr_scores(model_pipeline: Pipeline, X_test: Any, y_test: Any) -> OrderedDict[Any, float]:
    classes = _resolve_estimator_classes(model_pipeline)
    y_scores = model_pipeline.predict_proba(X_test)
    y_test_binary = _positive_class_matrix(y_test, classes)

    aucpr_scores: OrderedDict[Any, float] = OrderedDict()
    for index, class_label in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_test_binary[:, index], y_scores[:, index])
        aucpr_scores[_json_safe(class_label)] = auc(recall, precision)

    return aucpr_scores


def _make_macro_aucpr_scorer() -> Callable[[BaseEstimator, Any, Any], float]:
    def scorer(estimator: BaseEstimator, X: Any, y_true: Any) -> float:
        classes = _resolve_estimator_classes(estimator)
        y_scores = estimator.predict_proba(X)
        y_true_binary = _positive_class_matrix(y_true, classes)
        return average_precision_score(y_true_binary, y_scores, average="macro")

    return scorer


def _make_meta_recall_scorer(
    reference_class: Any | None = None,
) -> Callable[[BaseEstimator, Any, Any], float]:
    def scorer(estimator: BaseEstimator, X: Any, y_true: Any) -> float:
        classes = _resolve_estimator_classes(estimator)
        active_reference = classes[0] if reference_class is None else reference_class
        predictions = estimator.predict(X)

        y_true_binary = (np.asarray(y_true) != active_reference).astype(int)
        y_pred_binary = (np.asarray(predictions) != active_reference).astype(int)
        return recall_score(y_true_binary, y_pred_binary, zero_division=0)

    return scorer


def _make_class_recall_scorer(class_label: Any) -> Callable[[BaseEstimator, Any, Any], float]:
    def scorer(estimator: BaseEstimator, X: Any, y_true: Any) -> float:
        predictions = estimator.predict(X)
        y_true_binary = (np.asarray(y_true) == class_label).astype(int)
        y_pred_binary = (np.asarray(predictions) == class_label).astype(int)
        return recall_score(y_true_binary, y_pred_binary, zero_division=0)

    return scorer


def _make_class_aucpr_scorer(class_label: Any) -> Callable[[BaseEstimator, Any, Any], float]:
    def scorer(estimator: BaseEstimator, X: Any, y_true: Any) -> float:
        classes = _resolve_estimator_classes(estimator)
        y_scores = estimator.predict_proba(X)

        if class_label not in classes:
            raise ValueError(f"Class '{class_label}' is not present in the fitted estimator.")

        class_index = int(np.where(classes == class_label)[0][0])
        y_true_binary = (np.asarray(y_true) == class_label).astype(int)
        return average_precision_score(y_true_binary, y_scores[:, class_index])

    return scorer


def _build_scoring_bundle(
    classes: np.ndarray,
    *,
    meta_reference_class: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    active_meta_reference = classes[0] if meta_reference_class is None else meta_reference_class

    scoring: dict[str, Any] = {
        "bal_acc": "balanced_accuracy",
        "f1_macro": "f1_macro",
        "aucpr": _make_macro_aucpr_scorer(),
        "meta_rec": _make_meta_recall_scorer(active_meta_reference),
    }

    class_metrics: dict[str, Any] = {}
    for index, class_label in enumerate(classes):
        recall_key = f"rec_{index}"
        aucpr_key = f"aucpr_{index}"

        scoring[recall_key] = _make_class_recall_scorer(class_label)
        scoring[aucpr_key] = _make_class_aucpr_scorer(class_label)
        class_metrics[class_label] = {
            "recall_key": recall_key,
            "aucpr_key": aucpr_key,
            "slug": _slugify_label(class_label),
        }

    return scoring, class_metrics, active_meta_reference


def create_objective(
    *,
    model_name: str,
    preprocessor: Any,
    X_train: Any,
    y_train: Any,
    cv: int = 5,
    random_state: int = 42,
    meta_reference_class: Any | None = None,
) -> Callable[[optuna.trial.Trial], float]:
    classes = np.unique(np.asarray(y_train))
    scoring, class_metrics, active_meta_reference = _build_scoring_bundle(
        classes,
        meta_reference_class=meta_reference_class,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        estimator = build_estimator(
            model_name=model_name,
            trial=trial,
            random_state=random_state,
        )

        trial_pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("classifier", estimator),
            ]
        )

        cv_results = cross_validate(
            trial_pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
        )

        trial.set_user_attr("model_name", model_name)
        trial.set_user_attr("classes", _json_safe(classes))
        trial.set_user_attr("f1_macro", float(cv_results["test_f1_macro"].mean()))
        trial.set_user_attr("aucpr", float(cv_results["test_aucpr"].mean()))
        trial.set_user_attr("meta_reference_class", _json_safe(active_meta_reference))
        trial.set_user_attr("meta_non_reference_recall", float(cv_results["test_meta_rec"].mean()))

        for class_label, metric_info in class_metrics.items():
            recall_key = f"test_{metric_info['recall_key']}"
            aucpr_key = f"test_{metric_info['aucpr_key']}"
            slug = metric_info["slug"]

            trial.set_user_attr(f"class_label_{slug}", _json_safe(class_label))
            trial.set_user_attr(
                f"recall_class_{slug}",
                float(cv_results[recall_key].mean()),
            )
            trial.set_user_attr(
                f"aucpr_class_{slug}",
                float(cv_results[aucpr_key].mean()),
            )

        return float(cv_results["test_bal_acc"].mean())

    return objective


def optimize_model(
    *,
    model_name: str,
    preprocessor: Any,
    X_train: Any,
    y_train: Any,
    n_trials: int = 50,
    cv: int = 5,
    random_state: int = 42,
    direction: str = "maximize",
    meta_reference_class: Any | None = None,
) -> OptimizationArtifacts:
    objective = create_objective(
        model_name=model_name,
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train,
        cv=cv,
        random_state=random_state,
        meta_reference_class=meta_reference_class,
    )

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    best_pipeline, best_estimator = fit_model_pipeline(
        preprocessor=preprocessor,
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        model_params=study.best_params,
        random_state=random_state,
    )

    classes = np.unique(np.asarray(y_train))
    contour_params = resolve_contour_params(study, model_name=model_name)
    active_meta_reference = classes[0] if meta_reference_class is None else meta_reference_class

    return OptimizationArtifacts(
        model_name=model_name,
        study=study,
        best_pipeline=best_pipeline,
        best_estimator=best_estimator,
        contour_params=contour_params,
        classes_=classes,
        meta_reference_class=active_meta_reference,
    )


def resolve_contour_params(study: optuna.Study, *, model_name: str) -> list[str]:
    normalized_name = model_name.lower()
    preferred = MODEL_REGISTRY[normalized_name].contour_params
    best_param_names = list(study.best_params)

    if preferred and all(parameter in study.best_params for parameter in preferred):
        return list(preferred)

    return best_param_names[:2]


def build_optuna_figures(
    study: optuna.Study,
    *,
    contour_params: list[str] | tuple[str, ...] | None = None,
) -> OrderedDict[str, Any]:
    figures: OrderedDict[str, Any] = OrderedDict()
    figures["optimization_history"] = vis.plot_optimization_history(study)
    figures["param_importances"] = vis.plot_param_importances(study)
    figures["parallel_coordinate"] = vis.plot_parallel_coordinate(study)
    figures["slice"] = vis.plot_slice(study)
    figures["edf"] = vis.plot_edf(study)

    active_contour_params = list(contour_params or [])
    if len(active_contour_params) >= 2:
        figures["contour"] = vis.plot_contour(study, params=active_contour_params[:2])

    figures["rank"] = vis.plot_rank(study)
    return figures


def get_top_trials_dataframe(study: optuna.Study, *, top_n: int = 10) -> pd.DataFrame:
    results = study.trials_dataframe()
    return results.sort_values(by="value", ascending=False).head(top_n)
