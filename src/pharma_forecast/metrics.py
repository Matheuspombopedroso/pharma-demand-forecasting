"""Forecast error metrics shared by the model and the baselines.

Using one implementation for every forecaster guarantees that a reported gain
comes from the forecasts, not from differences in how errors were computed.

* **MAE** (grams): average absolute error per origin. Easy to read, but not
  comparable across ingredients of different size.
* **WAPE**: sum|error| / sum|actual|. Scale-free and robust when individual
  actuals are small; this is the headline metric and the tuning objective.
* **MAPE**: mean of |error| / |actual| over rows with actual > 0. Reported
  because it is commonly asked for, but it over-weights low-volume rows.
"""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error in the units of the target (grams)."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted absolute percentage error (0.10 = 10 %)."""
    y_true = np.asarray(y_true, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - np.asarray(y_pred))) / denom)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error over rows with a non-zero actual."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])))


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MAE, WAPE and MAPE in one dict."""
    return {
        "mae": mae(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }
