"""MLflow tracking helpers (Rule 7: reproducibility).

Every run logs: git SHA, DVC data hash, the full config, params, metrics,
plots, feature list, SHAP summary and the model (pyfunc + signature +
input example). When tracking is disabled every call is a no-op, so the same
pipeline runs in CI without an MLflow server.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

log = logging.getLogger(__name__)


def git_sha() -> str:
    if sha := os.getenv("GITHUB_SHA") or os.getenv("GIT_SHA"):
        return sha
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def dvc_data_hash(paths: list[Path], lock_file: Path = Path("dvc.lock")) -> str:
    """Hash of the data versions: md5s recorded by DVC if available, else file stats."""
    if lock_file.exists():
        lock = yaml.safe_load(lock_file.read_text(encoding="utf-8")) or {}
        md5s = []
        for stage in (lock.get("stages") or {}).values():
            for dep in stage.get("deps", []) + stage.get("outs", []):
                if any(str(p).replace("\\", "/") in dep.get("path", "") for p in paths):
                    md5s.append(dep.get("md5", ""))
        if md5s:
            return hashlib.md5("".join(sorted(md5s)).encode()).hexdigest()
    h = hashlib.md5()
    for p in paths:
        p = Path(p)
        files = sorted(p.rglob("*")) if p.is_dir() else [p]
        for f in files:
            if f.is_file():
                st = f.stat()
                h.update(f"{f.name}:{st.st_size}".encode())
    return h.hexdigest()


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{key}."))
        elif isinstance(v, list):
            out[key] = json.dumps(v)[:500]
        else:
            out[key] = v
    return out


class Tracker:
    def __init__(self, enabled: bool, experiment: str) -> None:
        self.enabled = enabled
        self.experiment = experiment
        self.run_id: str | None = None
        if enabled:
            import mlflow

            self.mlflow = mlflow
            if uri := os.getenv("MLFLOW_TRACKING_URI"):
                mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment)

    @contextlib.contextmanager
    def run(
        self, name: str, nested: bool = False, tags: dict[str, str] | None = None
    ) -> Iterator[Tracker]:
        if not self.enabled:
            yield self
            return
        with self.mlflow.start_run(run_name=name, nested=nested, tags=tags) as r:
            prev = self.run_id
            if not nested:
                self.run_id = r.info.run_id
            try:
                yield self
            finally:
                if not nested:
                    self.run_id = prev or self.run_id

    def params(self, params: dict[str, Any]) -> None:
        if self.enabled:
            flat = {
                k: (str(v)[:500] if v is not None else "None") for k, v in flatten(params).items()
            }
            items = list(flat.items())
            for i in range(0, len(items), 100):
                self.mlflow.log_params(dict(items[i : i + 100]))

    def metrics(self, metrics: dict[str, float], prefix: str = "", step: int | None = None) -> None:
        if self.enabled:
            clean = {
                f"{prefix}{k}": float(v) for k, v in metrics.items() if isinstance(v, (int, float))
            }
            self.mlflow.log_metrics(clean, step=step)

    def tags(self, tags: dict[str, Any]) -> None:
        if self.enabled:
            self.mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def artifact(self, path: Path, artifact_path: str | None = None) -> None:
        if self.enabled and Path(path).exists():
            self.mlflow.log_artifact(str(path), artifact_path)

    def artifacts(self, directory: Path, artifact_path: str | None = None) -> None:
        if self.enabled and Path(directory).exists():
            self.mlflow.log_artifacts(str(directory), artifact_path)

    def log_dict(self, d: dict[str, Any], name: str) -> None:
        if self.enabled:
            self.mlflow.log_dict(d, name)

    def text(self, text: str, name: str) -> None:
        if self.enabled:
            self.mlflow.log_text(text, name)

    def log_scorer(
        self, scorer_path: Path, input_example: pd.DataFrame, registered_model: str | None
    ) -> dict[str, Any]:
        """Log the Scorer as an MLflow pyfunc model and register a new version."""
        if not self.enabled:
            return {}
        from mlflow.models import infer_signature

        from risklens.pyfunc import RiskLensPyfunc, output_frame

        scorer = RiskLensPyfunc.load_scorer(scorer_path)
        # integer columns may be missing at inference -> declare them as float in the signature
        ints = input_example.select_dtypes(include=["integer"]).columns
        input_example = input_example.astype(dict.fromkeys(ints, "float64"))
        output = output_frame(scorer.score(input_example, explain=False))
        signature = infer_signature(input_example, output)
        kwargs: dict[str, Any] = {
            "python_model": RiskLensPyfunc(),
            "artifacts": {"scorer": str(scorer_path)},
            "signature": signature,
            "input_example": input_example,
            "pip_requirements": _pip_requirements(),
            "registered_model_name": registered_model,
        }
        try:
            info = self.mlflow.pyfunc.log_model(name="model", **kwargs)
        except TypeError:  # MLflow < 3
            info = self.mlflow.pyfunc.log_model(artifact_path="model", **kwargs)
        version = getattr(info, "registered_model_version", None)
        return {"model_uri": info.model_uri, "version": version}


def _pip_requirements() -> list[str]:
    import lightgbm
    import pandas
    import sklearn
    import xgboost

    reqs = [
        f"lightgbm=={lightgbm.__version__}",
        f"xgboost=={xgboost.__version__}",
        f"scikit-learn=={sklearn.__version__}",
        f"pandas=={pandas.__version__}",
        "risklens",
    ]
    try:
        import catboost

        reqs.append(f"catboost=={catboost.__version__}")
    except ImportError:
        pass
    return reqs
