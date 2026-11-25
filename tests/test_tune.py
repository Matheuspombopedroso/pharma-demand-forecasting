"""Smoke test for the Ray Tune search (2 trials, 2 local CPUs)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pharma_forecast.config import Settings
from pharma_forecast.data import generate_consumption

ray = pytest.importorskip("ray")


def test_tune_returns_best_params_and_logs_trials(tmp_path: Path) -> None:
    from pharma_forecast.tune import SEARCH_SPACE, run_tuning

    settings = Settings(
        data_path=tmp_path / "data.csv",
        mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        mlflow_artifact_root=tmp_path / "artifacts",
        experiment_name="tune-test",
        ray_results_dir=tmp_path / "rt",
        seed=0,
    )
    raw = generate_consumption(n_days=220, ingredients=("a", "b"), seed=3)
    res = run_tuning(raw, settings, num_samples=2, max_concurrent=2, n_folds=3, test_days=14)

    assert set(res.best_params) == set(SEARCH_SPACE)
    assert len(res.trials) == 2
    assert 0 < res.best_cv_wape < 1
    assert res.best_cv_wape == pytest.approx(res.trials["cv_wape"].min())
