"""Leakage tests for features and baselines.

The core idea: if a feature at origin t only depends on qty(<= t), then
replacing every value *after* t with garbage must leave that feature
unchanged. Any column that changes is reading the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pharma_forecast.baselines import add_baselines
from pharma_forecast.data import generate_consumption
from pharma_forecast.features import (
    add_targets,
    build_features,
    feature_columns,
)


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return generate_consumption(n_days=200, ingredients=("a", "b", "c"), seed=7)


def _scramble_after(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Replace consumption and promo after ``cutoff`` with random values."""
    out = df.copy()
    future = out["date"] > cutoff
    rng = np.random.default_rng(0)
    out.loc[future, "qty_g"] = rng.uniform(0, 10_000, future.sum())
    out.loc[future, "promo"] = rng.integers(0, 2, future.sum())
    return out


def test_features_do_not_use_future_values(raw: pd.DataFrame) -> None:
    cutoff = pd.Timestamp("2023-05-15")
    cols = feature_columns() + ["scale"]
    original = build_features(raw)
    scrambled = build_features(_scramble_after(raw, cutoff))

    past = original["date"] <= cutoff
    pd.testing.assert_frame_equal(
        original.loc[past, cols].reset_index(drop=True),
        scrambled.loc[past, cols].reset_index(drop=True),
    )


def test_baselines_do_not_use_future_values(raw: pd.DataFrame) -> None:
    cutoff = pd.Timestamp("2023-05-15")
    cols = ["naive_7", "snaive_7", "smean4_7", "naive_30", "snaive_30", "smean4_30"]
    original = add_baselines(raw)
    scrambled = add_baselines(_scramble_after(raw, cutoff))
    past = original["date"] <= cutoff
    pd.testing.assert_frame_equal(
        original.loc[past, cols].reset_index(drop=True),
        scrambled.loc[past, cols].reset_index(drop=True),
    )


def test_feature_columns_exclude_targets_and_raw_quantity() -> None:
    cols = feature_columns()
    assert not any(c.startswith("target") for c in cols)
    # qty_g itself is only allowed via the scale-normalised lag_0.
    assert "qty_g" not in cols


def test_target_is_sum_of_next_h_days(raw: pd.DataFrame) -> None:
    table = add_targets(build_features(raw), (7,))
    one = table[table["ingredient"] == "a"].reset_index(drop=True)
    i = 120
    expected = one.loc[i + 1 : i + 7, "qty_g"].sum()
    assert one.loc[i, "target_7"] == pytest.approx(expected)
    # The last 7 origins have no complete future -> NaN target.
    assert one["target_7"].tail(7).isna().all()


def test_seasonal_naive_repeats_last_week(raw: pd.DataFrame) -> None:
    b = add_baselines(raw, (7, 9))
    one = b[b["ingredient"] == "a"].reset_index(drop=True)
    i = 50
    last_week = one.loc[i - 6 : i, "qty_g"].to_numpy()
    assert one.loc[i, "snaive_7"] == pytest.approx(last_week.sum())
    # 9 days = one full week + the first 2 days of the pattern (t-6, t-5).
    assert one.loc[i, "snaive_9"] == pytest.approx(last_week.sum() + last_week[:2].sum())


def test_seasonal_mean_averages_four_weeks(raw: pd.DataFrame) -> None:
    b = add_baselines(raw, (7,))
    one = b[b["ingredient"] == "a"].reset_index(drop=True)
    i = 60
    four_weeks = one.loc[i - 27 : i, "qty_g"].to_numpy()
    # 7-day forecast = sum of the 7 weekday averages = (sum of 28 days) / 4.
    assert one.loc[i, "smean4_7"] == pytest.approx(four_weeks.sum() / 4)
