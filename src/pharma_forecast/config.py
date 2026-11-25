"""Central configuration for the forecasting pipeline.

Every other module reads its paths, horizons and defaults from here so there
is exactly one place to change them. Values can be overridden with
environment variables (see ``.env.example``), which is how the Docker image
and the CI smoke run shrink the workload without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Forecast horizons in days. The purchasing team plans orders one week,
# two weeks and one month ahead, so we train one model per horizon.
HORIZONS: tuple[int, ...] = (7, 15, 30)

# Name under which the forecaster is stored in the MLflow Model Registry.
REGISTERED_MODEL_NAME = "pharma-demand-forecaster"

# Registry aliases used by the promote / rollback flow (see registry.py).
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"
PREVIOUS_ALIAS = "previous"


def _project_root() -> Path:
    """Return the repository root (two levels above this file's package)."""
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Runtime settings resolved from environment variables with defaults.

    Attributes:
        data_path: CSV produced by ``generate-data`` and read by every step.
        mlflow_tracking_uri: MLflow backend store. SQLite is used because the
            Model Registry (aliases, versions) needs a database backend.
        mlflow_artifact_root: Directory where MLflow stores model files.
        experiment_name: MLflow experiment that groups all runs.
        ray_results_dir: Where Ray Tune writes trial checkpoints/logs. Kept
            short on purpose: Windows has a 260-char path limit and Ray trial
            directories are deeply nested.
        seed: Global random seed so every run is reproducible.
    """

    data_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("PF_DATA_PATH", _project_root() / "data" / "consumption.csv")
        )
    )
    mlflow_tracking_uri: str = field(
        default_factory=lambda: os.getenv(
            "MLFLOW_TRACKING_URI",
            # as_posix() keeps forward slashes, which SQLAlchemy URIs require on Windows.
            f"sqlite:///{(_project_root() / 'mlflow.db').as_posix()}",
        )
    )
    mlflow_artifact_root: Path = field(
        default_factory=lambda: Path(os.getenv("PF_ARTIFACT_ROOT", _project_root() / "mlartifacts"))
    )
    experiment_name: str = field(
        default_factory=lambda: os.getenv("PF_EXPERIMENT", "pharma-demand-forecasting")
    )
    ray_results_dir: Path = field(
        default_factory=lambda: Path(os.getenv("PF_RAY_RESULTS", _project_root() / "ray_results"))
    )
    seed: int = field(default_factory=lambda: int(os.getenv("PF_SEED", "42")))


def get_settings() -> Settings:
    """Build a fresh :class:`Settings` (re-reads env vars on every call)."""
    return Settings()
