"""RiskLens ML package.

Pure Python (no Django). Used from the CLI, notebooks, the Celery worker and
the FastAPI serving app. Heavy dependencies are imported lazily inside the
modules that need them so that ``import risklens`` stays cheap.
"""

__version__ = "0.1.0"
