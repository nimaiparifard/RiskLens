"""Optuna hyper-parameter search (TPE sampler + median pruner).

* each trial is a nested MLflow run
* the study can live in Postgres (``tuning.storage``) so it resumes after a crash
* LightGBM trials report validation AUC every 50 rounds -> unpromising trials pruned
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from risklens.config import TuningConfig
from risklens.registry import create_model
from risklens.tracking import Tracker

log = logging.getLogger(__name__)


def search_space(trial: Any, model: str) -> dict[str, Any]:
    if model == "lightgbm":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 300, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.8),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0, 0.1),
        }
    if model == "xgboost":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 100, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.8),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "gamma": trial.suggest_float("gamma", 0, 5),
        }
    if model == "catboost":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 9),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
            "random_strength": trial.suggest_float("random_strength", 0, 3),
        }
    raise ValueError(f"no search space for {model}")


def _lgbm_pruning_callback(trial: Any, every: int = 50) -> Callable[[Any], None]:
    import optuna

    def _callback(env: Any) -> None:
        if env.iteration % every:
            return
        for _data, metric, value, _higher in env.evaluation_result_list:
            if metric == "auc":
                trial.report(value, step=env.iteration)
                if trial.should_prune():
                    raise optuna.TrialPruned(f"pruned at iteration {env.iteration}")

    return _callback


def tune(
    cfg: TuningConfig,
    base_params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    tracker: Tracker,
    seed: int = 42,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    import optuna

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=200)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=cfg.storage,
        study_name=cfg.study_name or f"risklens-{cfg.model}",
        load_if_exists=True,
    )

    def objective(trial: Any) -> float:
        params = {**base_params, **search_space(trial, cfg.model)}
        with tracker.run(f"trial-{trial.number}", nested=True):
            if cfg.model == "lightgbm":
                import lightgbm as lgb

                # same as LGBMModel.fit but with a pruning callback
                p = dict(params)
                es = p.pop("early_stopping_rounds", 200)
                if pw := p.pop("pos_weight", None):
                    p["scale_pos_weight"] = pw
                p.setdefault("random_state", seed)
                p.setdefault("verbose", -1)
                p.setdefault("metric", "auc")
                clf = lgb.LGBMClassifier(**p)
                from risklens.models.lgbm import eval_kwargs

                clf.fit(
                    X,
                    y,
                    **eval_kwargs(X_valid, y_valid),
                    eval_metric="auc",
                    callbacks=[
                        lgb.early_stopping(es, verbose=False),
                        _lgbm_pruning_callback(trial),
                    ],
                )
                score = roc_auc_score(
                    y_valid,
                    np.asarray(clf.predict_proba(X_valid, num_iteration=clf.best_iteration_))[:, 1],
                )
            else:
                model = create_model(cfg.model, **params)
                model.fit(X, y, X_valid, y_valid)
                score = roc_auc_score(y_valid, model.predict_proba(X_valid))
            tracker.params({k: v for k, v in params.items() if k != "pos_weight"})
            tracker.metrics({"valid_roc_auc": score})
        return float(score)

    study.optimize(objective, n_trials=cfg.n_trials, timeout=cfg.timeout, gc_after_trial=True)
    log.info("optuna best AUC %.5f with %s", study.best_value, study.best_params)
    tracker.metrics({"tuning_best_valid_auc": study.best_value})
    tracker.log_dict(study.best_params, "tuning/best_params.json")

    if out_dir is not None and len(study.trials) >= 2:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from optuna.visualization.matplotlib import (
                plot_optimization_history,
                plot_param_importances,
            )

            out_dir.mkdir(parents=True, exist_ok=True)
            for name, fn in (
                ("param_importance", plot_param_importances),
                ("history", plot_optimization_history),
            ):
                fn(study)
                path = out_dir / f"optuna_{name}.png"
                plt.gcf().set_size_inches(8, 5)
                plt.tight_layout()
                plt.savefig(path, dpi=100)
                plt.close("all")
                tracker.artifact(path, "tuning")
        except Exception as exc:  # plots are nice-to-have
            log.warning("could not plot optuna study: %s", exc)
    return {**base_params, **study.best_params}


def trials_summary(study: Any) -> pd.DataFrame:
    return study.trials_dataframe()[["number", "value", "state"]].sort_values(
        "value", ascending=False
    )


__all__ = ["search_space", "trials_summary", "tune"]
