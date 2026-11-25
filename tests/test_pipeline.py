"""End-to-end smoke test on a tiny dataset with a temporary MLflow store.

Covers train -> register -> promote -> rollback -> predict, and asserts the
registry aliases move exactly as documented in ``registry.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from mlflow import MlflowClient

from pharma_forecast.config import REGISTERED_MODEL_NAME, Settings
from pharma_forecast.data import generate_consumption
from pharma_forecast.predict import predict_latest
from pharma_forecast.registry import promote, register, rollback
from pharma_forecast.train import run_training

SMALL = {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1}
WORSE = {"n_estimators": 2, "max_depth": 1, "learning_rate": 0.01}


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_path=tmp_path / "data.csv",
        mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        mlflow_artifact_root=tmp_path / "artifacts",
        experiment_name="test",
        ray_results_dir=tmp_path / "rt",
        seed=0,
    )


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return generate_consumption(n_days=260, ingredients=("a", "b"), seed=11)


def _aliases(uri: str) -> dict[str, str]:
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    model = client.get_registered_model(REGISTERED_MODEL_NAME)
    return {alias: str(v) for alias, v in model.aliases.items()}


def test_train_register_promote_rollback_predict(raw: pd.DataFrame, settings: Settings) -> None:
    uri = settings.mlflow_tracking_uri
    good = run_training(raw, settings, SMALL, n_folds=3, test_days=14)
    assert set(good.holdout["forecaster"]) == {"xgboost", "naive", "seasonal_naive"}

    # First version becomes champion directly (bootstrap).
    v1 = register(good.run_id, uri)
    assert _aliases(uri) == {"champion": v1}

    # A clearly worse model is registered as challenger and promotion is refused.
    bad = run_training(raw, settings, WORSE, n_folds=3, test_days=14)
    v2 = register(bad.run_id, uri)
    assert _aliases(uri) == {"champion": v1, "challenger": v2}
    decision = promote(uri)
    assert not decision.promoted
    assert _aliases(uri)["champion"] == v1

    # Forced promotion keeps the old champion as @previous ...
    assert promote(uri, force=True).promoted
    assert _aliases(uri) == {"champion": v2, "previous": v1}

    # ... so rollback can restore it.
    rolled_from, now = rollback(uri)
    assert (rolled_from, now) == (v2, v1)
    assert _aliases(uri) == {"champion": v1}

    preds = predict_latest(raw, settings)
    assert len(preds) == 2
    assert (preds["model_version"] == v1).all()
    assert (preds[["forecast_7", "forecast_15", "forecast_30"]] > 0).all().all()
    # Longer horizons should forecast more total grams.
    assert (preds["forecast_30"] > preds["forecast_7"]).all()
