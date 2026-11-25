"""Naive and seasonal-naive baselines.

A forecasting model is only worth its complexity if it beats the simple rules
a purchasing analyst would use in a spreadsheet. We compare against two:

* **Naive**: "tomorrow looks like today", so the next H days total
  ``H * qty(t)``. Very sensitive to weekday effects (a Sunday origin
  predicts near-zero demand for the whole month).
* **Seasonal naive (weekly)**: repeat the last 7 days as a pattern. Day t+1
  has the same weekday as day t-6, so the forecast for the next H days is
  ``q * sum(last 7 days) + sum(first r days of last week)`` with
  ``H = 7q + r``. This is a strong baseline for weekly-seasonal data.

Both use only qty(<= t), exactly like the model's features, so the
comparison is fair (same information, same origins, same metric code).
"""

from __future__ import annotations

import pandas as pd

from pharma_forecast.config import HORIZONS


def add_baselines(df: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Add ``naive_{H}``, ``snaive_{H}`` and ``smean4_{H}`` forecasts (grams).

    Args:
        df: Long DataFrame with ``date``, ``ingredient``, ``qty_g``.
        horizons: Horizons to produce forecasts for.

    Returns:
        Copy of ``df`` sorted by ingredient/date with baseline columns added.
    """
    out = df.sort_values(["ingredient", "date"]).reset_index(drop=True).copy()
    g = out.groupby("ingredient", sort=False)["qty_g"]

    # last_week[k] = qty(t-6+k) for k = 0..6, i.e. the weekday matching t+1+k.
    last_week = [g.shift(6 - k) for k in range(7)]
    week_sum = sum(last_week)

    # avg_week[k] = mean of qty(t-6+k-7j) for j = 0..3: same weekday, 4 weeks.
    avg_week = [sum(g.shift(6 - k + 7 * j) for j in range(4)) / 4 for k in range(7)]
    avg_week_sum = sum(avg_week)

    for h in horizons:
        out[f"naive_{h}"] = h * out["qty_g"]
        q, r = divmod(h, 7)
        partial = sum(last_week[:r]) if r else 0.0
        out[f"snaive_{h}"] = q * week_sum + partial
        partial4 = sum(avg_week[:r]) if r else 0.0
        out[f"smean4_{h}"] = q * avg_week_sum + partial4
    return out
