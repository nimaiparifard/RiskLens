"""MLflow pyfunc wrapper around ``Scorer`` (requires mlflow).

model = mlflow.pyfunc.load_model("models:/risklens-credit@champion")
model.predict(df)                       # generic: probability / grade / decision
scorer = model.unwrap_python_model().scorer   # full API incl. SHAP reasons
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow.pyfunc
import pandas as pd

from risklens.interfaces import ScoreResult
from risklens.scorer import Scorer


def output_frame(results: list[ScoreResult]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "probability": [r.probability for r in results],
            "grade": [r.grade for r in results],
            "decision": [r.decision for r in results],
        }
    )


class RiskLensPyfunc(mlflow.pyfunc.PythonModel):  # type: ignore[name-defined,misc]
    scorer: Scorer

    @staticmethod
    def load_scorer(path: str | Path) -> Scorer:
        return Scorer.load(Path(path))

    def load_context(self, context: Any) -> None:
        self.scorer = self.load_scorer(context.artifacts["scorer"])

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        return output_frame(self.scorer.score(model_input, explain=False))
