"""Walk-forward evaluation shared by ``train`` and ``tune``.

This module answers two questions:

1. *How good is a given hyper-parameter set?* -- :func:`cross_validate` fits
   on each tuning fold's training window and scores the following test
   window. Ray Tune calls it once per trial.
2. *Is the final model better than the baselines?* --
   :func:`holdout_report` scores XGBoost, naive and seasonal-naive on exactly
   the same hold-out rows with the same metric functions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from pharma_forecast.baselines import add_baselines
from pharma_forecast.config import HORIZONS
from pharma_forecast.features import build_modelling_table
from pharma_forecast.metrics import all_metrics
from pharma_forecast.model import MultiHorizonForecaster
from pharma_forecast.validation import Fold

FORECASTERS = ("xgboost", "naive", "seasonal_naive")


def prepare_table(raw: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Raw consumption -> features + targets + baseline forecasts.

    Baselines are computed on the raw series first so they ride along in the
    same rows as the model inputs; that is what guarantees model and baseline
    are scored on identical origins.
    """
    return build_modelling_table(add_baselines(raw, horizons), horizons)


def evaluate_fold(
    table: pd.DataFrame,
    fold: Fold,
    params: dict[str, Any],
    horizon: int,
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Train on the fold's training window and score its test window.

    Returns:
        ``(metrics, predictions)`` where ``predictions`` has the test rows'
        ``date``, ``ingredient``, actual, model and baseline forecasts.
    """
    train = table[fold.train_mask(table["date"], horizon)]
    test = table[fold.test_mask(table["date"])]
    test = test[test[f"target_{horizon}"].notna()]

    model = MultiHorizonForecaster(params, (horizon,), seed)
    model.fit_horizon(train, horizon)
    pred = model.predict_horizon(test, horizon)

    preds = pd.DataFrame(
        {
            "fold": fold.index,
            "horizon": horizon,
            "date": test["date"].to_numpy(),
            "ingredient": test["ingredient"].to_numpy(),
            "actual": test[f"target_{horizon}"].to_numpy(),
            "xgboost": pred,
            "naive": test[f"naive_{horizon}"].to_numpy(),
            "seasonal_naive": test[f"snaive_{horizon}"].to_numpy(),
        }
    )
    return all_metrics(preds["actual"], preds["xgboost"]), preds


def cross_validate(
    table: pd.DataFrame,
    folds: list[Fold],
    params: dict[str, Any],
    horizons: tuple[int, ...] = HORIZONS,
    seed: int = 42,
    on_fold_end: Callable[[int, dict[str, float]], None] | None = None,
) -> dict[str, float]:
    """Average model metrics over tuning folds, per horizon and overall.

    Args:
        table: Output of :func:`prepare_table`.
        folds: Tuning folds (the hold-out must NOT be included).
        params: XGBoost hyper-parameters.
        horizons: Horizons to evaluate.
        seed: XGBoost seed.
        on_fold_end: Optional callback ``(fold_number, running_metrics)``
            called after each fold. Ray Tune uses it to report intermediate
            results so the ASHA scheduler can stop bad trials early.

    Returns:
        Flat dict such as ``{"wape_h7": ..., "mae_h7": ..., "cv_wape": ...}``
        where ``cv_wape`` is the mean WAPE over all horizons (tuning objective).
    """
    per_fold: list[dict[str, float]] = []
    for i, fold in enumerate(folds):
        row: dict[str, float] = {}
        for h in horizons:
            m, _ = evaluate_fold(table, fold, params, h, seed)
            for name, value in m.items():
                row[f"{name}_h{h}"] = value
        per_fold.append(row)
        running = _summarise(per_fold, horizons)
        if on_fold_end is not None:
            on_fold_end(i + 1, running)
    return _summarise(per_fold, horizons)


def _summarise(per_fold: list[dict[str, float]], horizons: tuple[int, ...]) -> dict[str, float]:
    """Mean of each metric across folds, plus the overall ``cv_wape``."""
    keys = per_fold[0].keys()
    out = {k: float(np.mean([f[k] for f in per_fold])) for k in keys}
    out["cv_wape"] = float(np.mean([out[f"wape_h{h}"] for h in horizons]))
    return out


def holdout_report(
    table: pd.DataFrame,
    holdout: Fold,
    params: dict[str, Any],
    horizons: tuple[int, ...] = HORIZONS,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score model and baselines on the untouched hold-out fold.

    Returns:
        ``(report, predictions)``. ``report`` has one row per
        (horizon, forecaster) with MAE, WAPE, MAPE and row count.
    """
    rows = []
    all_preds = []
    for h in horizons:
        _, preds = evaluate_fold(table, holdout, params, h, seed)
        all_preds.append(preds)
        for name in FORECASTERS:
            m = all_metrics(preds["actual"], preds[name])
            rows.append({"horizon": h, "forecaster": name, **m, "n_rows": len(preds)})
    return pd.DataFrame(rows), pd.concat(all_preds, ignore_index=True)
