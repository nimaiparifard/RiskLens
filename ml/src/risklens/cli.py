"""Typer CLI.

risklens download credit
risklens validate  --config ml/configs/credit.yaml
risklens features  --config ml/configs/credit.yaml
risklens train     --config ml/configs/credit.yaml [-o data.nrows=30000]
risklens evaluate  --config ml/configs/credit.yaml
risklens promote   --config ml/configs/credit.yaml [--approve]
risklens benchmark --config ml/configs/credit.yaml
risklens synth     --task credit --out data/synthetic
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import typer

from risklens.config import load_config

app = typer.Typer(add_completion=False, help="RiskLens ML pipeline")

ConfigOpt = typer.Option(Path("ml/configs/credit.yaml"), "--config", "-c", help="YAML config")
OverrideOpt = typer.Option(None, "--override", "-o", help="dotted.key=value overrides")


def _setup_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("matplotlib", "PIL", "urllib3", "botocore", "alembic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@app.callback()
def main() -> None:
    _setup_logging()


@app.command()
def download(
    task: str = typer.Argument("credit"), raw_dir: Path | None = None, force: bool = False
) -> None:
    """Download Kaggle data (skips if already present)."""
    from risklens.data.download import download as dl

    default = Path("data/raw/home-credit" if task == "credit" else "data/raw/ieee-fraud")
    dl(task, raw_dir or default, force)


@app.command()
def validate(config: Path = ConfigOpt, override: list[str] = OverrideOpt) -> None:
    """Validate raw data against pandera schemas (non-zero exit on failure)."""
    from risklens.pipelines import features_stage

    cfg = load_config(config, override)
    summary = features_stage.validate(cfg, report=Path(cfg.reports_dir) / "validation.json")
    typer.echo(json.dumps(summary, indent=2))


@app.command()
def features(
    config: Path = ConfigOpt, override: list[str] = OverrideOpt, skip_validation: bool = False
) -> None:
    """Build the modelling dataset parquet (DVC stage 'features')."""
    from risklens.pipelines import features_stage

    cfg = load_config(config, override)
    features_stage.run(cfg, validate_first=not skip_validation)


@app.command()
def train(config: Path = ConfigOpt, override: list[str] = OverrideOpt) -> None:
    """Train all configured models, log to MLflow, register the final Scorer."""
    cfg = load_config(config, override)
    if cfg.task == "credit":
        from risklens.pipelines.train_credit import run
    else:
        from risklens.pipelines.train_fraud import run
    result = run(cfg)
    typer.echo(result.model_table.to_string(index=False))
    typer.echo(
        f"\nscorer: {result.scorer_path}  version: {result.model_version}  run: {result.run_id}"
    )


@app.command()
def evaluate(config: Path = ConfigOpt, override: list[str] = OverrideOpt) -> None:
    """Re-evaluate the saved Scorer on the frozen test split."""
    from risklens.pipelines.evaluate import run

    metrics = run(load_config(config, override))
    typer.echo(json.dumps({k: round(v, 5) for k, v in metrics.items()}, indent=2))


@app.command()
def promote(
    config: Path = ConfigOpt,
    version: str | None = typer.Option(None, help="registered version (default: latest)"),
    approve: bool = typer.Option(False, help="challenger -> champion (human approval step)"),
    bootstrap: bool = typer.Option(
        False, help="if no champion exists, promote straight to champion"
    ),
    summary_file: Path | None = typer.Option(
        None, help="write markdown summary (GitHub job summary)"
    ),
) -> None:
    """Champion/challenger promotion against the MLflow Model Registry."""
    from risklens.pipelines import promote as p

    cfg = load_config(config)
    if approve:
        v = p.approve_challenger(cfg.registered_model)
        typer.echo(f"approved: version {v} is now champion")
        return
    decision = p.evaluate_candidate(cfg.registered_model, cfg.promotion, version)
    md = decision.to_markdown()
    typer.echo(md)
    if summary_file:
        summary_file.write_text(md, encoding="utf-8")
    if bootstrap and decision.promote and p.alias_version(cfg.registered_model, p.CHAMPION) is None:
        p.approve_challenger(cfg.registered_model)
        typer.echo("bootstrap: challenger promoted to champion")
    if not decision.promote:
        raise typer.Exit(code=0)


@app.command()
def benchmark(config: Path = ConfigOpt, n: int = 200) -> None:
    """Single-request latency of the saved Scorer (features + model + SHAP)."""
    import pandas as pd

    from risklens.pipelines.evaluate import latency_benchmark
    from risklens.scorer import Scorer

    cfg = load_config(config)
    scorer = Scorer.load(Path(cfg.model_dir) / "scorer.joblib")
    df = pd.read_parquet(cfg.data.processed_path).head(n)
    cols = [c for c in scorer.input_columns + scorer.history_columns if c in df.columns]
    typer.echo(json.dumps(latency_benchmark(scorer, df[cols], n), indent=2))


@app.command()
def synth(
    task: str = "credit", out: Path = Path("data/synthetic"), rows: int = 2000, seed: int = 0
) -> None:
    """Write synthetic CSVs with the Kaggle file layout (no Kaggle account needed)."""
    from risklens.data import synthetic
    from risklens.data.load import CREDIT_FILES, FRAUD_FILES

    out.mkdir(parents=True, exist_ok=True)
    if task == "credit":
        for name, df in synthetic.make_credit_tables(rows, seed).items():
            df.to_csv(out / CREDIT_FILES[name], index=False)
    else:
        synthetic.make_fraud_transactions(rows, seed).to_csv(
            out / FRAUD_FILES["transaction"], index=False
        )
    typer.echo(f"wrote synthetic {task} data to {out}")


if __name__ == "__main__":
    app()
