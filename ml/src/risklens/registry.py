"""Name -> class registries (plug-ins).

    @register_model("lightgbm")
    class LGBMClassifier(Classifier): ...

    model = create_model("lightgbm", **params)

Model modules are imported lazily so optional libraries (catboost, torch)
are only required when that model is actually used.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, TypeVar

from risklens.interfaces import Classifier

T = TypeVar("T", bound=type[Classifier])

MODELS: dict[str, type[Classifier]] = {}

# name -> module that registers it (lazy import)
_MODEL_MODULES: dict[str, str] = {
    "logreg_woe": "risklens.models.logreg",
    "lightgbm": "risklens.models.lgbm",
    "xgboost": "risklens.models.xgb",
    "catboost": "risklens.models.catboost",
    "stacking": "risklens.models.stacking",
}


def register_model(name: str) -> Callable[[T], T]:
    def deco(cls: T) -> T:
        cls.name = name
        MODELS[name] = cls
        return cls

    return deco


def get_model_class(name: str) -> type[Classifier]:
    if name not in MODELS:
        module = _MODEL_MODULES.get(name)
        if module is None:
            raise KeyError(f"Unknown model '{name}'. Known: {sorted(_MODEL_MODULES)}")
        importlib.import_module(module)
    return MODELS[name]


def create_model(name: str, **params: Any) -> Classifier:
    return get_model_class(name)(**params)


def available_models() -> list[str]:
    return sorted(_MODEL_MODULES)
