"""Distributed hyper-parameter search with Ray Tune.

How it works
------------
* Ray starts a local "cluster" of worker processes (one per CPU we allow).
* Each **trial** is one hyper-parameter set sampled from :data:`SEARCH_SPACE`.
  It runs :func:`cross_validate` over the tuning folds (never the hold-out)
  and reports the running mean WAPE after every fold.
* The **ASHA** scheduler compares trials at the same fold count and stops the
  clearly worse ones early, so the budget goes to promising regions.
* Trials run in parallel, each pinned to 1 CPU with XGBoost single-threaded,
  so ``max_concurrent`` really is the degree of parallelism.

The same code scales from a laptop to a multi-node Ray cluster: only the
``ray.init`` address changes, not the trainable.

After the search, every trial is logged to MLflow from the driver process as
a nested run under one parent ``tune`` run. Logging from the driver (rather
than from each worker) avoids concurrent writes to the SQLite backend.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import mlflow
import pandas as pd
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from pharma_forecast.config import HORIZONS, Settings
from pharma_forecast.evaluation import cross_validate, prepare_table
from pharma_forecast.tracking import setup_mlflow
from pharma_forecast.validation import make_folds, split_tuning_holdout

# Search space around DEFAULT_PARAMS. Log-uniform for learning rate and L2
# because their effect is multiplicative; integer ranges for tree shape.
SEARCH_SPACE: dict[str, Any] = {
    "n_estimators": tune.randint(100, 600),
    "max_depth": tune.randint(3, 9),
    "learning_rate": tune.loguniform(0.01, 0.3),
    "subsample": tune.uniform(0.6, 1.0),
    "colsample_bytree": tune.uniform(0.5, 1.0),
    "min_child_weight": tune.uniform(1.0, 20.0),
    "reg_lambda": tune.loguniform(0.1, 10.0),
}


@dataclass
class TuneResult:
    """Outcome of a search.

    Attributes:
        best_params: Hyper-parameters of the best trial (by ``cv_wape``).
        best_cv_wape: Its mean WAPE over the tuning folds.
        trials: One row per trial (params + metrics).
        wall_seconds: Wall-clock time of ``tuner.fit()`` only.
        parent_run_id: MLflow run grouping the trials.
    """

    best_params: dict[str, Any]
    best_cv_wape: float
    trials: pd.DataFrame
    wall_seconds: float
    parent_run_id: str


def _trainable(config: dict[str, Any], table: pd.DataFrame, folds: list, seed: int) -> None:
    """Ray Tune trainable: walk-forward CV for one hyper-parameter set.

    ``table`` and ``folds`` arrive through ``tune.with_parameters``, which puts
    them in Ray's object store once instead of pickling them into every
    trial.
    """
    params = dict(config)
    # randint returns numpy ints; XGBoost and MLflow prefer plain Python ints.
    params["n_estimators"] = int(params["n_estimators"])
    params["max_depth"] = int(params["max_depth"])

    def report(fold_number: int, metrics: dict[str, float]) -> None:
        tune.report({**metrics, "folds_done": fold_number})

    cross_validate(table, folds, params, HORIZONS, seed, on_fold_end=report)


def _short_trial_name(trial: Any) -> str:
    """Short trial directory names: Windows paths are limited to 260 chars."""
    return f"t_{trial.trial_id}"


def run_tuning(
    raw: pd.DataFrame,
    settings: Settings,
    num_samples: int = 20,
    max_concurrent: int = 4,
    n_folds: int = 5,
    test_days: int = 28,
) -> TuneResult:
    """Run the Ray Tune search and log all trials to MLflow.

    Args:
        raw: Long consumption DataFrame.
        settings: Paths, seed and MLflow configuration.
        num_samples: Number of hyper-parameter sets to try.
        max_concurrent: Trials allowed to run at the same time (= CPUs used).
        n_folds: Walk-forward folds including the hold-out (excluded here).
        test_days: Days per test window.

    Returns:
        A :class:`TuneResult` with the best parameters.
    """
    table = prepare_table(raw, HORIZONS)
    folds = make_folds(raw["date"].max(), max(HORIZONS), n_folds, test_days)
    tuning_folds, _holdout = split_tuning_holdout(folds)

    # Ray's default 30 s raylet start-up timeout was too short on the Windows
    # laptop used for development (start-up measured at ~30 s under load).
    os.environ.setdefault("RAY_raylet_start_wait_time_s", "120")
    # Dashboard off: it is not needed locally and adds startup time and ports.
    ray.init(num_cpus=max_concurrent, include_dashboard=False, ignore_reinit_error=True)
    try:
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(
                    _trainable, table=table, folds=tuning_folds, seed=settings.seed
                ),
                {"cpu": 1},
            ),
            param_space=SEARCH_SPACE,
            tune_config=tune.TuneConfig(
                metric="cv_wape",
                mode="min",
                num_samples=num_samples,
                max_concurrent_trials=max_concurrent,
                # grace_period=1: every trial gets at least one fold before
                # ASHA may stop it; max_t = number of folds it can report.
                scheduler=ASHAScheduler(
                    time_attr="folds_done",
                    max_t=len(tuning_folds),
                    grace_period=1,
                    reduction_factor=2,
                ),
                trial_dirname_creator=_short_trial_name,
                trial_name_creator=_short_trial_name,
            ),
            run_config=tune.RunConfig(
                name="xgb_search",
                storage_path=str(settings.ray_results_dir.resolve()),
                verbose=0,
            ),
        )
        start = time.perf_counter()
        grid = tuner.fit()
        wall = time.perf_counter() - start
    finally:
        ray.shutdown()

    trials = grid.get_dataframe()
    best = grid.get_best_result(metric="cv_wape", mode="min", scope="last")
    best_params = {k: best.config[k] for k in SEARCH_SPACE}
    best_params["n_estimators"] = int(best_params["n_estimators"])
    best_params["max_depth"] = int(best_params["max_depth"])
    best_cv = float(best.metrics["cv_wape"])

    parent_id = _log_trials_to_mlflow(
        settings, trials, best_params, best_cv, wall, num_samples, max_concurrent
    )
    return TuneResult(best_params, best_cv, trials, wall, parent_id)


def _log_trials_to_mlflow(
    settings: Settings,
    trials: pd.DataFrame,
    best_params: dict[str, Any],
    best_cv: float,
    wall: float,
    num_samples: int,
    max_concurrent: int,
) -> str:
    """Log one parent run plus a nested run per trial; return the parent id."""
    setup_mlflow(settings)
    with mlflow.start_run(run_name="tune") as parent:
        mlflow.set_tag("stage", "tune")
        mlflow.log_params(
            {"num_samples": num_samples, "max_concurrent": max_concurrent, "scheduler": "ASHA"}
        )
        mlflow.log_metrics(
            {
                "best_cv_wape": best_cv,
                "tune_wall_seconds": wall,
                "trials_stopped_early": int(
                    (trials["folds_done"] < trials["folds_done"].max()).sum()
                ),
            }
        )
        mlflow.log_dict(best_params, "best_params.json")
        # to_dict("records") keeps names like "config/max_depth" intact
        # (itertuples would mangle them because of the slash).
        for row_d in trials.to_dict("records"):
            with mlflow.start_run(run_name=str(row_d["trial_id"]), nested=True):
                mlflow.log_params({k: row_d[f"config/{k}"] for k in SEARCH_SPACE})
                mlflow.log_metrics(
                    {"cv_wape": float(row_d["cv_wape"]), "folds_done": int(row_d["folds_done"])}
                )
        return parent.info.run_id


def save_best_params(result: TuneResult, path: Any) -> None:
    """Persist the best parameters so ``train --params`` can reuse them."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result.best_params, fh, indent=2)
