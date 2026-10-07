"""Typed configuration objects (Rule 3 + Rule 6: config, not code; typed boundaries).

Every tunable lives in ``ml/configs/*.yaml``. YAML is parsed into these pydantic
models so a typo fails loudly at load time instead of deep inside training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ImbalanceStrategy = Literal["none", "class_weight", "scale_pos_weight", "undersample", "smote"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class DataConfig(StrictModel):
    raw_dir: Path
    processed_path: Path
    processed_test_path: Path | None = None
    splits_path: Path
    nrows: int | None = Field(None, description="Limit main-table rows for quick runs")
    synthetic: bool = Field(False, description="Use generated data (CI / no Kaggle access)")
    synthetic_rows: int = 5000


class SplitConfig(StrictModel):
    strategy: Literal["stratified", "time"] = "stratified"
    valid_size: float = 0.15
    test_size: float = 0.15
    time_column: str | None = None


class SelectionConfig(StrictModel):
    enabled: bool = True
    max_missing: float = 0.95
    corr_threshold: float = 0.98
    corr_sample: int = 50_000
    top_k: int | None = 300
    permutation: bool = False
    permutation_sample: int = 20_000


class EncodingConfig(StrictModel):
    target_encode: list[str] = Field(default_factory=list)
    one_hot_max_cardinality: int = 10
    n_folds: int = 5
    smoothing: float = 20.0


class ModelSpec(StrictModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class TuningConfig(StrictModel):
    enabled: bool = False
    model: str = "lightgbm"
    n_trials: int = 100
    timeout: int | None = None
    storage: str | None = Field(None, description="e.g. postgresql://... so studies can resume")
    study_name: str | None = None


class ExplainConfig(StrictModel):
    top_k: int = 5
    model: str = Field(
        "lightgbm", description="base model used for reasons when final is an ensemble"
    )
    shap_sample: int = 2000


class FairnessConfig(StrictModel):
    enabled: bool = True
    sensitive_features: list[str] = Field(default_factory=lambda: ["CODE_GENDER", "AGE_GROUP"])
    gate_features: list[str] | None = Field(
        None,
        description="attributes whose gap gates promotion (default: all sensitive_features); others are reported only",
    )
    max_gap: float = 0.10
    mitigate: bool = False


def _default_guards() -> dict[str, Literal["lower", "higher"]]:
    return {"expected_cost": "lower"}


class PromotionConfig(StrictModel):
    primary_metric: str = "roc_auc"
    min_improvement: float = 0.002
    guard_metrics: dict[str, Literal["lower", "higher"]] = Field(default_factory=_default_guards)
    max_fairness_gap: float = 0.10


class CostConfig(StrictModel):
    """Business cost matrix. Amount-dependent costs use ``amount_column``."""

    amount_column: str | None = None
    default_amount: float = 500_000.0
    fn_cost_rate: float = Field(0.45, description="credit: LGD; fraud: share of amount lost")
    fp_cost_rate: float = Field(0.08, description="credit: lost margin; fraud: unused")
    fp_cost_fixed: float = Field(0.0, description="fixed friction cost per false positive")
    target_fpr: float | None = Field(None, description="fraud: choose threshold at this FPR")
    grade_bands: dict[str, float] = Field(
        default_factory=lambda: {"A": 0.03, "B": 0.06, "C": 0.12, "D": 0.25, "E": 1.0}
    )


class TrainConfig(StrictModel):
    task: Literal["credit", "fraud"]
    experiment: str
    registered_model: str
    seed: int = 42
    id_column: str
    target: str
    data: DataConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    feature_sources: list[str] = Field(default_factory=list)
    exclude_features: list[str] = Field(default_factory=list)
    encoding: EncodingConfig = Field(default_factory=EncodingConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    imbalance: ImbalanceStrategy = "scale_pos_weight"
    models: list[ModelSpec]
    final_model: str
    calibration: Literal["isotonic", "sigmoid", "none"] = "isotonic"
    cv_folds: int = 5
    robustness_check: bool = True
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    explain: ExplainConfig = Field(default_factory=ExplainConfig)
    fairness: FairnessConfig = Field(default_factory=FairnessConfig)
    promotion: PromotionConfig = Field(default_factory=PromotionConfig)
    costs: CostConfig = Field(default_factory=CostConfig)
    reports_dir: Path = Path("reports")
    model_dir: Path = Path("models")
    mlflow: bool = True

    @model_validator(mode="after")
    def _final_model_is_declared(self) -> TrainConfig:
        names = {m.name for m in self.models if m.enabled}
        if self.final_model not in names:
            raise ValueError(f"final_model '{self.final_model}' not in enabled models {names}")
        return self

    @property
    def enabled_models(self) -> list[ModelSpec]:
        return [m for m in self.models if m.enabled]


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _set_dotted(d: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> TrainConfig:
    """Load a YAML config. ``costs`` may be a path to ``costs.yaml`` (section = task).

    ``overrides`` are ``dotted.key=value`` strings, value parsed as YAML
    (e.g. ``data.nrows=20000`` or ``tuning.enabled=true``).
    """
    path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    costs = raw.get("costs")
    if isinstance(costs, str):
        costs_path = (path.parent / costs) if not Path(costs).is_absolute() else Path(costs)
        all_costs = yaml.safe_load(costs_path.read_text(encoding="utf-8"))
        raw["costs"] = all_costs[raw["task"]]
    for item in overrides or []:
        key, _, value = item.partition("=")
        _set_dotted(raw, key.strip(), yaml.safe_load(value))
    return TrainConfig.model_validate(raw)


def config_to_dict(cfg: TrainConfig) -> dict[str, Any]:
    return cfg.model_dump(mode="json")


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    return _deep_update(base, override)
