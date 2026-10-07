"""Download Kaggle competition data (requires ~/.kaggle/kaggle.json and
accepting the competition rules on the website first)."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

from risklens.data.load import CREDIT_FILES, FRAUD_FILES

log = logging.getLogger(__name__)

COMPETITIONS = {
    "credit": ("home-credit-default-risk", CREDIT_FILES),
    "fraud": ("ieee-fraud-detection", FRAUD_FILES),
}


def is_complete(raw_dir: Path, files: dict[str, str]) -> bool:
    return all((raw_dir / f).exists() for f in files.values())


def download(task: str, raw_dir: Path, force: bool = False) -> Path:
    competition, files = COMPETITIONS[task]
    raw_dir = Path(raw_dir)
    if is_complete(raw_dir, files) and not force:
        log.info("%s already present in %s – skipping download", competition, raw_dir)
        return raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    from kaggle.api.kaggle_api_extended import KaggleApi  # lazy: needs credentials

    api = KaggleApi()
    api.authenticate()
    log.info("downloading %s ...", competition)
    api.competition_download_files(competition, path=str(raw_dir), quiet=False)
    for archive in raw_dir.glob("*.zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw_dir)
        archive.unlink()
    return raw_dir
