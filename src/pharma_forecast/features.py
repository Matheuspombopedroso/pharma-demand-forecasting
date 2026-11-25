"""Leakage-safe feature and target engineering.

The forecasting setup
---------------------
Each row of the modelling table is a *forecast origin*: the end of day ``t``
for one ingredient. At that moment we know every consumption value up to and
including day ``t`` and nothing after it. For each horizon ``H`` the target is
the total consumption over the next ``H`` days::

    target_H(t) = qty(t+1) + qty(t+2) + ... + qty(t+H)

which is exactly the quantity purchasing needs ("how many grams will we use
in the next 7/15/30 days?").

The leakage rule
----------------
**Features at row t may only use qty(<= t).** Everything here is built from
``shift`` (looking back) and trailing ``rolling`` windows, computed inside a
per-ingredient ``groupby`` so one ingredient never sees another's history.
``tests/test_features.py`` enforces this: it scrambles all values after a
cut-off date and asserts that no feature at or before that date changes.
Targets are the only columns that look forward, and they live in separate
``target_*`` columns that are never passed to the model as inputs.

Scale normalisation
-------------------
Ingredients differ by an order of magnitude in volume (40 g/day vs 600 g/day).
To let one global XGBoost model share patterns across ingredients we divide
lags and rolling stats by a trailing ``scale`` (the 28-day mean) and predict
the *ratio* ``target_H / (H * scale)``. :func:`unscale_prediction` converts
back to grams. ``log_scale`` is also given to the model so it can still learn
level-dependent effects.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from pharma_forecast.config import HORIZONS

# Lags (days back from the origin; 0 = the origin day itself). 6/7, 13/14 and
# 27/28 pick up "same weekday last week / two weeks / four weeks ago".
LAGS: tuple[int, ...] = (0, 1, 2, 3, 6, 7, 13, 14, 27, 28)

# Trailing window lengths for rolling mean / std.
WINDOWS: tuple[int, ...] = (7, 14, 28, 56, 91)

# Window used as the normalisation scale (see module docstring).
SCALE_WINDOW = 28

# Small constant to avoid division by zero for an idle ingredient.
EPS = 1e-6


def stable_ingredient_id(name: str) -> int:
    """Map an ingredient name to a deterministic integer id.

    A CRC32 hash (unlike Python's ``hash``) is stable across processes and
    machines, so the same ingredient always gets the same id at training and
    at prediction time, without storing a lookup table next to the model.
    """
    return zlib.crc32(name.encode("utf-8")) % 10_000


def _per_ingredient(df: pd.DataFrame) -> pd.core.groupby.SeriesGroupBy:
    """Group the quantity column by ingredient (helper for rolling ops)."""
    return df.groupby("ingredient", sort=False)["qty_g"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute model inputs for every (ingredient, date) origin.

    Args:
        df: Long DataFrame with ``date``, ``ingredient``, ``qty_g`` and
            ``promo``; one row per ingredient per day, no gaps.

    Returns:
        A copy of ``df`` (sorted by ingredient, date) with feature columns
        added. Early rows have NaN features until enough history exists;
        callers drop them with :func:`feature_columns`.
    """
    out = df.sort_values(["ingredient", "date"]).reset_index(drop=True).copy()
    g = _per_ingredient(out)

    # Trailing scale: mean of the last 28 days *including* the origin day.
    # rolling() is right-aligned by default, so the window ends at t.
    scale = g.transform(lambda s: s.rolling(SCALE_WINDOW).mean()) + EPS
    out["scale"] = scale
    out["log_scale"] = np.log(scale)

    for lag in LAGS:
        # shift(lag) moves values *forward* in time: row t receives qty(t-lag).
        out[f"lag_{lag}"] = g.shift(lag) / scale

    for w in WINDOWS:
        out[f"roll_mean_{w}"] = g.transform(lambda s, w=w: s.rolling(w).mean()) / scale
        out[f"roll_std_{w}"] = g.transform(lambda s, w=w: s.rolling(w).std()) / scale

    # Momentum: is the last week above or below the last quarter?
    out["momentum_7_91"] = out["roll_mean_7"] / (out["roll_mean_91"] + EPS)

    # Calendar features of the *first forecast day* (t+1). The calendar is
    # known in advance, so using t+1 is not leakage.
    next_day = out["date"] + pd.Timedelta(days=1)
    out["dow_next"] = next_day.dt.day_of_week
    doy = next_day.dt.day_of_year
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Promotion flag observed on the origin day (known at time t).
    out["promo_today"] = out["promo"]
    # How many of the last 7 days were promo days (campaign momentum).
    out["promo_last7"] = out.groupby("ingredient", sort=False)["promo"].transform(
        lambda s: s.rolling(7).sum()
    )

    out["ingredient_id"] = out["ingredient"].map(stable_ingredient_id)
    return out


def feature_columns() -> list[str]:
    """Return the ordered list of model input columns.

    Kept as a function (not inferred from the DataFrame) so training and
    serving always agree on names *and* order.
    """
    cols = ["log_scale"]
    cols += [f"lag_{lag}" for lag in LAGS]
    for w in WINDOWS:
        cols += [f"roll_mean_{w}", f"roll_std_{w}"]
    cols += [
        "momentum_7_91",
        "dow_next",
        "doy_sin",
        "doy_cos",
        "promo_today",
        "promo_last7",
        "ingredient_id",
    ]
    return cols


def add_targets(df: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Add forward-looking targets ``target_{H}`` (grams over next H days).

    The trailing H-day sum at row t+H equals qty(t+1)+...+qty(t+H); shifting
    it back by H aligns that future sum with origin t. The last H rows of
    every ingredient get NaN because their future is not observed yet.

    Also adds ``target_ratio_{H}`` = target / (H * scale), the quantity the
    model actually learns (see module docstring).
    """
    out = df.copy()
    g = _per_ingredient(out)
    for h in horizons:
        fwd = g.transform(lambda s, h=h: s.rolling(h).sum().shift(-h))
        out[f"target_{h}"] = fwd
        out[f"target_ratio_{h}"] = fwd / (h * out["scale"])
    return out


def unscale_prediction(ratio: np.ndarray, scale: np.ndarray, horizon: int) -> np.ndarray:
    """Convert a predicted ratio back to grams over the horizon."""
    return np.asarray(ratio) * horizon * np.asarray(scale)


def build_modelling_table(df: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Features + targets, with warm-up rows (incomplete history) removed.

    Rows whose *targets* are NaN (the most recent days) are kept: they are not
    usable for training, but they are exactly the rows we predict from in
    production. Training code filters on the target per horizon.
    """
    table = add_targets(build_features(df), horizons)
    return table.dropna(subset=feature_columns()).reset_index(drop=True)
