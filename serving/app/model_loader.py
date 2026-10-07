"""Load Scorers from the MLflow Model Registry by alias (champion / challenger),
with a local-file fallback and hot reload."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import structlog

from app.config import Settings
from risklens.scorer import Scorer

log = structlog.get_logger(__name__)

TASKS = ("credit", "fraud")


class ModelNotLoadedError(RuntimeError):
    def __init__(self, task: str) -> None:
        super().__init__(f"no {task} model loaded")
        self.task = task


@dataclass
class LoadedModel:
    task: str
    scorer: Scorer
    version: str
    source: str
    loaded_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class ModelRegistry:
    """Thread-safe holder of the live models (one champion + optional challenger per task)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._models: dict[str, LoadedModel] = {}
        self._challengers: dict[str, LoadedModel] = {}
        self._lock = threading.RLock()

    # ----------------------------------------------------------- access
    def get(self, task: str) -> LoadedModel:
        with self._lock:
            if task not in self._models:
                raise ModelNotLoadedError(task)
            return self._models[task]

    def challenger(self, task: str) -> LoadedModel | None:
        return self._challengers.get(task)

    def loaded(self) -> dict[str, LoadedModel]:
        return dict(self._models)

    def model_name(self, task: str) -> str:
        return (
            self.settings.credit_model_name if task == "credit" else self.settings.fraud_model_name
        )

    def set(self, model: LoadedModel) -> None:
        """Inject a model (tests, notebooks)."""
        with self._lock:
            self._models[model.task] = model

    # ---------------------------------------------------------- loading
    def load_all(self) -> dict[str, str]:
        return {task: self.reload(task) for task in TASKS}

    def reload(self, task: str) -> str:
        model = self._load(task, self.settings.model_alias)
        if model is None:
            return "unavailable"
        with self._lock:
            self._models[task] = model
        if self.settings.shadow_enabled:
            chall = self._load(task, self.settings.challenger_alias, fallback=False)
            with self._lock:
                if chall:
                    self._challengers[task] = chall
                else:
                    self._challengers.pop(task, None)
        log.info("model_loaded", task=task, version=model.version, source=model.source)
        return model.version

    def _load(self, task: str, alias: str, fallback: bool = True) -> LoadedModel | None:
        s = self.settings
        if s.model_source == "mlflow":
            try:
                return self._load_mlflow(task, alias)
            except Exception as exc:
                log.warning("mlflow_load_failed", task=task, alias=alias, error=str(exc))
        if fallback and (s.model_source == "local" or s.local_fallback):
            path = Path(s.credit_model_path if task == "credit" else s.fraud_model_path)
            if path.exists():
                scorer = Scorer.load(path)
                return LoadedModel(task, scorer, f"{scorer.version}", f"file:{path}")
            log.warning("model_file_missing", task=task, path=str(path))
        return None

    def _load_mlflow(self, task: str, alias: str) -> LoadedModel:
        import mlflow
        from mlflow import MlflowClient

        if self.settings.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        name = self.model_name(task)
        version = MlflowClient().get_model_version_by_alias(name, alias).version
        uri = f"models:/{name}/{version}"
        pyfunc = mlflow.pyfunc.load_model(uri)
        scorer: Scorer = pyfunc.unwrap_python_model().scorer
        scorer.version = str(version)
        return LoadedModel(task, scorer, str(version), uri)

    # ------------------------------------------------------ hot reload
    def registry_version(self, task: str) -> str | None:
        if self.settings.model_source != "mlflow":
            return None
        try:
            from mlflow import MlflowClient

            return (
                MlflowClient()
                .get_model_version_by_alias(self.model_name(task), self.settings.model_alias)
                .version
            )
        except Exception:
            return None

    def check_for_updates(self) -> dict[str, str]:
        """Reload any task whose registry alias now points to a different version."""
        changed = {}
        for task in TASKS:
            latest = self.registry_version(task)
            current = self._models.get(task)
            if latest and (current is None or current.version != str(latest)):
                changed[task] = self.reload(task)
        return changed
