"""Task-agnostic training pipeline used by train_credit / train_fraud.

prepare (serving feature code) -> encode (OOF target enc.) -> select
-> imbalance -> [tune] -> fit every model (nested MLflow runs)
-> stack -> calibrate (valid) -> cost-optimal threshold (valid)
-> evaluate on FROZEN test -> SHAP -> fairness -> Scorer bundle -> MLflow
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from risklens.config import TrainConfig, config_to_dict
from risklens.data.split import TEST, TRAIN, VALID, ordered_split
from risklens.evaluation import plots
from risklens.evaluation.cost import amounts_for, optimal_threshold
from risklens.evaluation.metrics import threshold_at_fpr
from risklens.evaluation.report import df_to_markdown, evaluate_predictions, results_table
from risklens.explain.reasons import base_feature
from risklens.explain.shap_explainer import (
    dependence_plot,
    global_importance,
    make_explainer,
    shap_summary_plot,
)
from risklens.fairness.report import fairness_report
from risklens.features.encoding import TabularEncoder
from risklens.features.selection import FeatureSelector
from risklens.interfaces import Classifier
from risklens.models.base import apply_imbalance
from risklens.models.calibration import Calibrator
from risklens.models.stacking import StackingModel
from risklens.registry import create_model
from risklens.scorer import Scorer, prepare_input
from risklens.tracking import Tracker, dvc_data_hash, git_sha

log = logging.getLogger(__name__)


@dataclass
class TrainResult:
    scorer: Scorer
    scorer_path: Path
    test_metrics: dict[str, float]
    model_table: pd.DataFrame
    run_id: str | None = None
    model_version: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Trainer:
    def __init__(self, cfg: TrainConfig, tracker: Tracker | None = None) -> None:
        self.cfg = cfg
        self.tracker = tracker or Tracker(enabled=cfg.mlflow, experiment=cfg.experiment)
        self.reports = Path(cfg.reports_dir)
        self.reports.mkdir(parents=True, exist_ok=True)

    # ================================================================ public
    def run(
        self,
        frame: pd.DataFrame,
        split: np.ndarray,
        input_columns: list[str],
        history_columns: list[str],
        run_name: str | None = None,
    ) -> TrainResult:
        cfg = self.cfg
        t0 = time.time()
        tags = {
            "task": cfg.task,
            "git_sha": git_sha(),
            "dvc_data_hash": dvc_data_hash([cfg.data.processed_path]),
        }
        with self.tracker.run(run_name or f"{cfg.task}-{datetime.now(UTC):%Y%m%d-%H%M}", tags=tags):
            self.tracker.params(config_to_dict(cfg))
            self.tracker.log_dict(config_to_dict(cfg), "config.json")
            result = self._run(frame, split, input_columns, history_columns, tags)
            result.run_id = self.tracker.run_id
            self.tracker.metrics({"train_seconds": time.time() - t0})
        log.info("training finished in %.1f min", (time.time() - t0) / 60)
        return result

    # =============================================================== private
    def _run(
        self,
        frame: pd.DataFrame,
        split: np.ndarray,
        input_columns: list[str],
        history_columns: list[str],
        tags: dict[str, str],
    ) -> TrainResult:
        cfg = self.cfg
        prepared = prepare_input(cfg.task, frame)
        y = prepared[cfg.target].astype(int)
        tr, va, te = (split == TRAIN), (split == VALID), (split == TEST)
        log.info(
            "split sizes train=%d valid=%d test=%d (pos rate %.4f)",
            tr.sum(),
            va.sum(),
            te.sum(),
            y.mean(),
        )
        self.tracker.metrics(
            {"n_train": int(tr.sum()), "n_valid": int(va.sum()), "n_test": int(te.sum())}
        )

        # ---------------------------------------------------------- encode
        drop = [cfg.id_column, cfg.target, *cfg.exclude_features]
        encoder = TabularEncoder(
            target_encode=cfg.encoding.target_encode,
            one_hot_max_cardinality=cfg.encoding.one_hot_max_cardinality,
            n_folds=cfg.encoding.n_folds,
            smoothing=cfg.encoding.smoothing,
            seed=cfg.seed,
            drop=drop,
        )
        X_tr = encoder.fit_transform(prepared[tr], y[tr])
        X_va = encoder.transform(prepared[va])
        X_te = encoder.transform(prepared[te])
        y_tr, y_va, y_te = y[tr], y[va], y[te]
        log.info("encoded: %d features", X_tr.shape[1])

        # ---------------------------------------------------------- select
        selector = FeatureSelector(cfg.selection, seed=cfg.seed).fit(X_tr, y_tr, X_va, y_va)
        X_tr, X_va, X_te = (
            selector.transform(X_tr),
            selector.transform(X_va),
            selector.transform(X_te),
        )
        selector.report_.to_csv(self.reports / "feature_selection.csv", index=False)
        (self.reports / "features.txt").write_text("\n".join(selector.selected_), encoding="utf-8")
        self.tracker.artifact(self.reports / "feature_selection.csv", "features")
        self.tracker.artifact(self.reports / "features.txt", "features")
        self.tracker.metrics(
            {
                "n_features_encoded": encoder_size(encoder),
                "n_features_selected": len(selector.selected_),
            }
        )

        # ------------------------------------------------------- imbalance
        X_fit, y_fit, extra = apply_imbalance(cfg.imbalance, X_tr, y_tr, cfg.seed)
        self.tracker.params(
            {"imbalance_extra": json.dumps({k: round(v, 3) for k, v in extra.items()})}
        )

        params_by_model = {
            m.name: {**m.params, **extra} for m in cfg.enabled_models if m.name != "stacking"
        }

        # ---------------------------------------------------------- tuning
        if cfg.tuning.enabled and cfg.tuning.model in params_by_model:
            from risklens.tuning.optuna_search import tune

            with self.tracker.run(f"tuning-{cfg.tuning.model}", nested=True):
                params_by_model[cfg.tuning.model] = tune(
                    cfg.tuning,
                    params_by_model[cfg.tuning.model],
                    X_fit,
                    y_fit,
                    X_va,
                    y_va,
                    self.tracker,
                    seed=cfg.seed,
                    out_dir=self.reports,
                )

        # ---------------------------------------------------------- models
        frame_te = prepared[te]
        fitted: dict[str, Classifier] = {}
        rows: dict[str, dict[str, float]] = {}
        preds_te: dict[str, np.ndarray] = {}
        for spec in cfg.enabled_models:
            t_model = time.time()
            with self.tracker.run(spec.name, nested=True):
                if spec.name == "stacking":
                    params = {
                        **spec.params,
                        "base_params": {n: params_by_model[n] for n in spec.params["base_models"]},
                        "seed": cfg.seed,
                        "cv": "time" if cfg.split.strategy == "time" else "stratified",
                    }
                    model = create_model("stacking", **params)
                    assert isinstance(model, StackingModel)
                    model.set_prefit(
                        {n: fitted[n] for n in spec.params["base_models"] if n in fitted}
                    )
                else:
                    params = params_by_model[spec.name]
                    model = create_model(spec.name, **params)
                model.fit(X_fit, y_fit, X_va, y_va)
                fitted[spec.name] = model
                _cal, thr, p_te = self._calibrate_and_threshold(
                    model, X_va, y_va, prepared[va], X_te
                )
                rep = evaluate_predictions(y_te.to_numpy(), p_te, thr, cfg.costs, frame_te)
                rep.metrics["fit_seconds"] = time.time() - t_model
                self.tracker.params(
                    {"model": spec.name, **{k: v for k, v in params.items() if k != "base_params"}}
                )
                self.tracker.metrics(rep.metrics, prefix="test_")
                rows[spec.name] = rep.metrics
                preds_te[spec.name] = p_te
                if spec.name == "stacking":
                    self.tracker.log_dict(model.meta_weights(), "stacking_meta_weights.json")  # type: ignore[attr-defined]
                if spec.name == "logreg_woe":
                    sc = model.scorecard()  # type: ignore[attr-defined]
                    sc.to_csv(self.reports / "scorecard.csv", index=False)
                    self.tracker.artifact(self.reports / "scorecard.csv")
            log.info(
                "%-12s test AUC=%.4f  KS=%.4f  cost=%.0f  (%.0fs)",
                spec.name,
                rep.metrics["roc_auc"],
                rep.metrics["ks"],
                rep.metrics["expected_cost"],
                time.time() - t_model,
            )

        table = results_table(rows)
        table.to_csv(self.reports / "model_comparison.csv", index=False)
        (self.reports / "model_comparison.md").write_text(df_to_markdown(table), encoding="utf-8")
        self.tracker.artifact(self.reports / "model_comparison.md")
        plots.roc_plot(
            {n: (y_te.to_numpy(), p) for n, p in preds_te.items()},
            self.reports / "roc_all_models.png",
        )
        self.tracker.artifact(self.reports / "roc_all_models.png", "plots")

        # ------------------------------------------------------ final model
        final = fitted[cfg.final_model]
        calibrator, threshold, p_te = self._calibrate_and_threshold(
            final, X_va, y_va, prepared[va], X_te
        )
        raw_te = final.predict_proba(X_te)
        report = evaluate_predictions(
            y_te.to_numpy(), p_te, threshold, cfg.costs, frame_te, out_dir=self.reports / "plots"
        )
        plots.calibration_plot(
            y_te.to_numpy(),
            {"uncalibrated": raw_te, "calibrated": p_te},
            self.reports / "plots" / "calibration_before_after.png",
        )
        metrics = dict(report.metrics)
        report.to_json(self.reports / "metrics.json")
        report.grades.to_csv(self.reports / "grades.csv", index=False)
        self.tracker.metrics(metrics)
        self.tracker.artifacts(self.reports / "plots", "plots")
        self.tracker.artifact(self.reports / "grades.csv")

        # ------------------------------------------------- robustness (OOT)
        if cfg.robustness_check and "lightgbm" in fitted:
            oot = self._robustness_check(
                prepared, y, encoder, selector, params_by_model["lightgbm"], fitted["lightgbm"]
            )
            metrics.update(oot)
            self.tracker.metrics(oot)

        # ------------------------------------------------------ explanation
        importance = self._explain(final, X_te)

        # -------------------------------------------------------- fairness
        if cfg.fairness.enabled:
            sens_cols = [c for c in cfg.fairness.sensitive_features if c in prepared.columns]
            fair = fairness_report(
                y_te.to_numpy(),
                p_te,
                threshold,
                prepared.loc[te, sens_cols].reset_index(drop=True),
                mitigate=cfg.fairness.mitigate,
                gate_features=cfg.fairness.gate_features,
            )
            fm = fair.metrics()
            metrics.update(fm)
            self.tracker.metrics(fm)
            (self.reports / "fairness.md").write_text(fair.to_markdown(), encoding="utf-8")
            self.tracker.artifact(self.reports / "fairness.md")

        # ------------------------------------------------- scorer bundle
        reference = self._reference_sample(
            prepared[te], p_te, importance, input_columns, history_columns
        )
        scorer = Scorer(
            task=cfg.task,
            encoder=encoder,
            feature_names=list(selector.selected_),
            model=final,
            calibrator=calibrator,
            threshold=threshold,
            grade_bands=cfg.costs.grade_bands,
            explain_model=cfg.explain.model,
            top_k=cfg.explain.top_k,
            input_columns=input_columns,
            history_columns=history_columns,
            metadata={
                "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                "trained_at": datetime.now(UTC).isoformat(),
                "config": config_to_dict(cfg),
                **tags,
            },
            reference=reference,
        )
        scorer_path = scorer.save(Path(cfg.model_dir) / "scorer.joblib")
        (self.reports / "metrics.json").write_text(
            json.dumps(scorer.metadata["metrics"], indent=2), encoding="utf-8"
        )

        example = frame.loc[te, input_columns + history_columns].head(5).reset_index(drop=True)
        logged = self.tracker.log_scorer(scorer_path, example, cfg.registered_model)
        if logged.get("version"):
            scorer.version = str(logged["version"])
            scorer.save(scorer_path)
            self.tracker.tags({"registered_version": logged["version"]})
        self.tracker.artifact(self.reports / "metrics.json")
        return TrainResult(
            scorer=scorer,
            scorer_path=scorer_path,
            test_metrics=metrics,
            model_table=table,
            model_version=logged.get("version"),
        )

    # ------------------------------------------------------------ helpers
    def _calibrate_and_threshold(
        self,
        model: Classifier,
        X_va: pd.DataFrame,
        y_va: pd.Series,
        frame_va: pd.DataFrame,
        X_te: pd.DataFrame,
    ) -> tuple[Calibrator, float, np.ndarray]:
        cfg = self.cfg
        p_va_raw = model.predict_proba(X_va)
        calibrator = Calibrator(cfg.calibration).fit(p_va_raw, y_va.to_numpy())
        p_va = calibrator.transform(p_va_raw)
        if cfg.costs.target_fpr is not None:
            threshold = threshold_at_fpr(y_va.to_numpy(), p_va, cfg.costs.target_fpr)
        else:
            amounts = amounts_for(frame_va, cfg.costs, len(y_va))
            threshold, _ = optimal_threshold(y_va.to_numpy(), p_va, amounts, cfg.costs)
        p_te = calibrator.transform(model.predict_proba(X_te))
        return calibrator, float(threshold), p_te

    def _robustness_check(
        self,
        prepared: pd.DataFrame,
        y: pd.Series,
        encoder: TabularEncoder,
        selector: FeatureSelector,
        lgb_params: dict[str, Any],
        lgb_model: Classifier,
    ) -> dict[str, float]:
        """Train on the lowest 85% ids, test on the newest 15% (time proxy)."""
        from sklearn.metrics import roc_auc_score

        ids = prepared[self.cfg.id_column]
        part = ordered_split(ids, holdout=0.15).to_numpy()
        X_all = selector.transform(encoder.transform(prepared))
        params = {k: v for k, v in lgb_params.items() if k != "early_stopping_rounds"}
        params["n_estimators"] = getattr(lgb_model, "best_iteration_", 1000)
        m = create_model("lightgbm", **params).fit(X_all[part == TRAIN], y[part == TRAIN])
        auc = roc_auc_score(y[part == TEST], m.predict_proba(X_all[part == TEST]))
        log.info("robustness check (ordered split) AUC=%.4f", auc)
        return {"oot_roc_auc": float(auc)}

    def _explain(self, final: Classifier, X_te: pd.DataFrame) -> pd.Series:
        cfg = self.cfg
        n = min(len(X_te), cfg.explain.shap_sample)
        sample = X_te.sample(n, random_state=cfg.seed)
        try:
            explainer = make_explainer(final, cfg.explain.model)
        except (TypeError, StopIteration):
            log.warning("final model has no tree explainer; skipping SHAP")
            imp = final.feature_importance()
            return imp if imp is not None else pd.Series(dtype=float)
        expl = explainer.explain(sample)
        importance = global_importance(expl)
        importance.rename("mean_abs_shap").to_csv(self.reports / "shap_importance.csv")
        shap_summary_plot(expl, self.reports / "plots" / "shap_summary.png")
        for feat in importance.index[:3]:
            dependence_plot(expl, feat, self.reports / "plots" / f"shap_dependence_{feat}.png")
        self.tracker.artifact(self.reports / "shap_importance.csv", "shap")
        self.tracker.artifacts(self.reports / "plots", "plots")
        return importance

    def _reference_sample(
        self,
        prepared_te: pd.DataFrame,
        p_te: np.ndarray,
        importance: pd.Series,
        input_columns: list[str],
        history_columns: list[str],
    ) -> pd.DataFrame:
        """Reference distribution for drift monitoring: top raw-level features + score."""
        allowed = set(input_columns) | set(history_columns)
        if self.cfg.task == "credit":  # application-derived columns are recomputable from the input
            allowed |= set(prepared_te.columns)
        allowed -= {self.cfg.id_column, self.cfg.target}
        cols: list[str] = []
        for f in importance.index:
            b = base_feature(f)
            if b in allowed and b in prepared_te.columns and b not in cols:
                cols.append(b)
            if len(cols) >= 30:
                break
        ref = prepared_te[cols].copy()
        ref["score"] = p_te
        return ref.sample(min(len(ref), 5000), random_state=self.cfg.seed).reset_index(drop=True)


def encoder_size(encoder: TabularEncoder) -> int:
    return len(encoder.get_feature_names_out())
