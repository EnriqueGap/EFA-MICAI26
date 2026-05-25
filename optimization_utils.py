from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
import optuna.visualization as vis
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, recall_score
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


EstimatorFactory = Callable[[dict[str, Any]], BaseEstimator]
DEFAULT_MODEL_CONFIG_PATH = Path(__file__).with_name("model_optimization_config.json")


@dataclass(frozen=True)
class ModelSpec:
    factory: EstimatorFactory


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


def load_model_optimization_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path is not None else DEFAULT_MODEL_CONFIG_PATH

    with path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not isinstance(config.get("models"), dict):
        raise ValueError(f"Model optimization config '{path}' must contain a 'models' object.")

    return config


def _get_model_config(
    model_name: str,
    *,
    model_config_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_model_optimization_config(model_config_path)
    normalized_name = model_name.lower()
    model_config = config["models"].get(normalized_name)

    if not isinstance(model_config, dict):
        raise ValueError(f"No optimization config found for model '{model_name}'.")

    return model_config


def _get_model_default_params(model_config: dict[str, Any], *, model_name: str) -> dict[str, Any]:
    default_params = model_config.get("default_params", {})

    if not isinstance(default_params, dict):
        raise ValueError(f"Model '{model_name}' default_params must be an object.")

    return dict(default_params)


def _require_search_space_value(param_name: str, param_config: dict[str, Any], key: str) -> Any:
    if key not in param_config:
        raise ValueError(f"Search-space parameter '{param_name}' is missing '{key}'.")
    return param_config[key]


def _suggest_parameter(trial: optuna.trial.Trial, param_name: str, param_config: dict[str, Any]) -> Any:
    param_type = _require_search_space_value(param_name, param_config, "type")

    if param_type in {"categorical", "categorial"}:
        choices = _require_search_space_value(param_name, param_config, "choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"Search-space parameter '{param_name}' choices must be a non-empty list.")
        return trial.suggest_categorical(param_name, choices)

    if param_type == "int":
        suggest_kwargs = {
            "step": param_config["step"],
        } if "step" in param_config else {}

        if "log" in param_config:
            suggest_kwargs["log"] = bool(param_config["log"])

        return trial.suggest_int(
            param_name,
            _require_search_space_value(param_name, param_config, "low"),
            _require_search_space_value(param_name, param_config, "high"),
            **suggest_kwargs,
        )

    if param_type == "float":
        suggest_kwargs = {}
        if "step" in param_config:
            suggest_kwargs["step"] = param_config["step"]
        if "log" in param_config:
            suggest_kwargs["log"] = bool(param_config["log"])

        return trial.suggest_float(
            param_name,
            _require_search_space_value(param_name, param_config, "low"),
            _require_search_space_value(param_name, param_config, "high"),
            **suggest_kwargs,
        )

    raise ValueError(
        f"Unsupported search-space type '{param_type}' for parameter '{param_name}'. "
        "Supported types: categorical, int, float."
    )


def _suggest_model_params(
    trial: optuna.trial.Trial,
    model_config: dict[str, Any],
    *,
    model_name: str,
) -> dict[str, Any]:
    search_space = model_config.get("search_space", {})

    if not isinstance(search_space, dict):
        raise ValueError(f"Model '{model_name}' search_space must be an object.")

    suggested_params: dict[str, Any] = {}
    for param_name, param_config in search_space.items():
        if not isinstance(param_config, dict):
            raise ValueError(f"Search-space parameter '{param_name}' must be an object.")
        suggested_params[param_name] = _suggest_parameter(trial, param_name, param_config)

    return suggested_params


def _build_decision_tree(
    params: dict[str, Any],
) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(**params)


def _build_random_forest(
    params: dict[str, Any],
) -> RandomForestClassifier:
    return RandomForestClassifier(**params)


def _build_xgboost(
    params: dict[str, Any],
) -> BaseEstimator:
    from xgboost import XGBClassifier

    return XGBClassifier(**params)


def _build_lightgbm(
    params: dict[str, Any],
) -> BaseEstimator:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(**params)


def _build_catboost(
    params: dict[str, Any],
) -> BaseEstimator:
    from catboost import CatBoostClassifier

    return CatBoostClassifier(**params)


def _build_balanced_random_forest(
    params: dict[str, Any],
) -> BaseEstimator:
    from imblearn.ensemble import BalancedRandomForestClassifier

    return BalancedRandomForestClassifier(**params)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "decision_tree": ModelSpec(factory=_build_decision_tree),
    "random_forest": ModelSpec(factory=_build_random_forest),
    "xgboost": ModelSpec(factory=_build_xgboost),
    "lightgbm": ModelSpec(factory=_build_lightgbm),
    "catboost": ModelSpec(factory=_build_catboost),
    "balanced_random_forest": ModelSpec(factory=_build_balanced_random_forest),
}


def get_available_models(
    *,
    model_config_path: str | Path | None = None,
) -> list[str]:
    available_models: list[str] = []

    for model_name in MODEL_REGISTRY:
        try:
            build_estimator(
                model_name=model_name,
                random_state=42,
                model_config_path=model_config_path,
            )
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
    model_config_path: str | Path | None = None,
) -> BaseEstimator:
    normalized_name = model_name.lower()
    if normalized_name not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}.")

    model_config = _get_model_config(normalized_name, model_config_path=model_config_path)
    final_params = _get_model_default_params(model_config, model_name=normalized_name)
    final_params.update(params or {})

    if trial is not None:
        final_params.update(
            _suggest_model_params(
                trial,
                model_config,
                model_name=normalized_name,
            )
        )

    final_params.setdefault("random_state", random_state)
    return MODEL_REGISTRY[normalized_name].factory(final_params)


def fit_model_pipeline(
    preprocessor: Any,
    model_name: str,
    X_train: Any,
    y_train: Any,
    *,
    model_params: dict[str, Any] | None = None,
    random_state: int = 42,
    model_config_path: str | Path | None = None,
) -> tuple[Pipeline, BaseEstimator]:
    estimator = build_estimator(
        model_name=model_name,
        params=model_params,
        random_state=random_state,
        model_config_path=model_config_path,
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("classifier", estimator),
        ]
    )
    model_pipeline.fit(X_train, y_train)
    return model_pipeline, model_pipeline.named_steps["classifier"]


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
    model_config_path: str | Path | None = None,
) -> Callable[[optuna.trial.Trial], float]:
    classes = np.unique(np.asarray(y_train))

    cv_strategy = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    scoring, class_metrics, active_meta_reference = _build_scoring_bundle(
        classes,
        meta_reference_class=meta_reference_class,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        estimator = build_estimator(
            model_name=model_name,
            trial=trial,
            random_state=random_state,
            model_config_path=model_config_path,
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
            cv=cv_strategy,
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
    model_config_path: str | Path | None = None,
) -> OptimizationArtifacts:
    objective = create_objective(
        model_name=model_name,
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train,
        cv=cv,
        random_state=random_state,
        meta_reference_class=meta_reference_class,
        model_config_path=model_config_path,
    )

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
    )

    study.optimize(objective, n_trials=n_trials)

    best_pipeline, best_estimator = fit_model_pipeline(
        preprocessor=preprocessor,
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        model_params=study.best_params,
        random_state=random_state,
        model_config_path=model_config_path,
    )

    classes = np.unique(np.asarray(y_train))
    contour_params = resolve_contour_params(
        study,
        model_name=model_name,
        model_config_path=model_config_path,
    )
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


def resolve_contour_params(
    study: optuna.Study,
    *,
    model_name: str,
    model_config_path: str | Path | None = None,
) -> list[str]:
    normalized_name = model_name.lower()
    if normalized_name not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}.")

    model_config = _get_model_config(normalized_name, model_config_path=model_config_path)
    preferred = model_config.get("contour_params", [])
    if not isinstance(preferred, list):
        raise ValueError(f"Model '{model_name}' contour_params must be a list.")

    best_param_names = list(study.best_params)

    if preferred and all(parameter in study.best_params for parameter in preferred):
        return preferred

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
