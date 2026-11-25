"""Synthetic data reproducibility and metric sanity checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pharma_forecast.data import generate_consumption
from pharma_forecast.metrics import mae, mape, wape


def test_same_seed_same_data() -> None:
    a = generate_consumption(n_days=60, ingredients=("x", "y"), seed=1)
    b = generate_consumption(n_days=60, ingredients=("x", "y"), seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_data_shape_and_positivity() -> None:
    df = generate_consumption(n_days=90, ingredients=("x", "y", "z"), seed=3)
    assert len(df) == 270
    assert (df["qty_g"] >= 0).all()
    # Contiguous daily dates per ingredient (features rely on no gaps).
    for _, g in df.groupby("ingredient"):
        assert (g["date"].diff().dropna() == pd.Timedelta(days=1)).all()


def test_weekly_seasonality_present() -> None:
    df = generate_consumption(n_days=364, ingredients=("x",), seed=5)
    by_dow = df.groupby(df["date"].dt.day_of_week)["qty_g"].mean()
    assert by_dow[6] < 0.5 * by_dow[0]  # Sundays much quieter than Mondays


def test_metrics() -> None:
    y = np.array([10.0, 20.0, 30.0])
    p = np.array([12.0, 18.0, 30.0])
    assert mae(y, p) == 4 / 3
    assert wape(y, p) == 4 / 60
    assert mape(y, p) == np.mean([0.2, 0.1, 0.0])
