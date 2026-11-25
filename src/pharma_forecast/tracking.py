"""MLflow setup helpers.

All runs go to one experiment in a local SQLite backend store. SQLite (not the
plain ``mlruns/`` file store) is used because the Model Registry -- versions,
aliases, tags -- requires a database-backed store.
"""

from __future__ import annotations

import mlflow

from pharma_forecast.config import Settings


def setup_mlflow(settings: Settings) -> str:
    """Point MLflow at the configured store and ensure the experiment exists.

    Returns:
        The experiment id.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    settings.mlflow_artifact_root.mkdir(parents=True, exist_ok=True)
    exp = mlflow.get_experiment_by_name(settings.experiment_name)
    if exp is None:
        # as_uri() yields file:///C:/... on Windows; raw backslash paths break
        # MLflow's artifact repository resolution.
        mlflow.create_experiment(
            settings.experiment_name,
            artifact_location=settings.mlflow_artifact_root.resolve().as_uri(),
        )
    return mlflow.set_experiment(settings.experiment_name).experiment_id
