"""Synthetic daily raw-material consumption for a compounding pharmacy.

Why this module exists
----------------------
A compounding pharmacy consumes active ingredients (in grams) every day as it
prepares prescriptions. Purchasing needs to know how much of each ingredient
will be consumed over the next 7, 15 and 30 days. Real consumption data is
private, so this module *simulates* it with the patterns that matter for
forecasting:

* a per-ingredient base level and a slow linear **trend** (growing or
  shrinking demand);
* **weekly seasonality** (the lab is busy Monday-Friday, half-staffed on
  Saturday and almost idle on Sunday);
* a mild **yearly seasonality** (e.g. vitamin D sells more in winter);
* **promotions**: multi-day campaigns that lift demand;
* rare **spikes**: one-off bulk orders;
* multiplicative **noise** drawn from a Gamma distribution, which keeps
  values positive and makes the variance grow with the level.

Everything is driven by a single ``numpy`` Generator seeded once, so the same
seed always yields the same CSV. No real pharmacy, patient or customer data is
used anywhere in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Fictional catalogue of active ingredients. The names are only labels; the
# demand parameters below are random, not based on any real pharmacy.
DEFAULT_INGREDIENTS: tuple[str, ...] = (
    "minoxidil",
    "melatonin",
    "vitamin_d3",
    "caffeine",
    "hyaluronic_acid",
    "omega_3",
    "biotin",
    "finasteride",
)

# Relative activity per weekday (Mon=0 ... Sun=6). The lab works full days
# on weekdays, a short shift on Saturday and nearly nothing on Sunday.
WEEKLY_PROFILE = np.array([1.10, 1.05, 1.05, 1.00, 1.10, 0.55, 0.15])


@dataclass(frozen=True)
class IngredientParams:
    """Random demand parameters drawn once per ingredient.

    Attributes:
        base: Average grams/day at the start of the series.
        yearly_growth: Relative change in level per year (0.2 = +20 %/year).
        yearly_amplitude: Strength of the annual sine wave (0.1 = +/-10 %).
        yearly_phase: Phase shift of that sine wave, in radians.
        noise_cv: Coefficient of variation of the multiplicative Gamma noise.
    """

    base: float
    yearly_growth: float
    yearly_amplitude: float
    yearly_phase: float
    noise_cv: float


def _draw_params(rng: np.random.Generator) -> IngredientParams:
    """Draw one ingredient's demand parameters from plausible ranges."""
    return IngredientParams(
        base=float(rng.uniform(40, 600)),
        yearly_growth=float(rng.uniform(-0.10, 0.40)),
        yearly_amplitude=float(rng.uniform(0.0, 0.20)),
        yearly_phase=float(rng.uniform(0, 2 * np.pi)),
        noise_cv=float(rng.uniform(0.15, 0.35)),
    )


def _promotion_multiplier(n_days: int, rng: np.random.Generator) -> np.ndarray:
    """Return a per-day multiplier that is > 1 while a promotion is running.

    Campaigns start on average every ~60 days, last 5-14 days and lift demand
    by 30-80 %. Overlapping campaigns simply keep the larger uplift.
    """
    mult = np.ones(n_days)
    day = int(rng.integers(10, 60))
    while day < n_days:
        length = int(rng.integers(5, 15))
        uplift = float(rng.uniform(1.3, 1.8))
        end = min(day + length, n_days)
        mult[day:end] = np.maximum(mult[day:end], uplift)
        # Gap until the next campaign: exponential-ish spacing around 60 days.
        day = end + int(rng.integers(30, 90))
    return mult


def generate_ingredient_series(
    name: str,
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate one ingredient's daily consumption.

    Args:
        name: Ingredient label written to the ``ingredient`` column.
        dates: Contiguous daily index to simulate.
        rng: Shared random generator (drives every random choice).

    Returns:
        DataFrame with columns ``date``, ``ingredient``, ``qty_g`` (grams
        consumed that day) and ``promo`` (1 if a promotion was running).
    """
    p = _draw_params(rng)
    n = len(dates)
    t = np.arange(n)

    trend = 1.0 + p.yearly_growth * t / 365.0
    weekly = WEEKLY_PROFILE[dates.day_of_week.to_numpy()]
    yearly = 1.0 + p.yearly_amplitude * np.sin(
        2 * np.pi * dates.day_of_year.to_numpy() / 365.25 + p.yearly_phase
    )
    promo = _promotion_multiplier(n, rng)
    # Bulk orders: ~1 % of days get a 2x-4x spike on top of everything else.
    spikes = np.where(rng.random(n) < 0.01, rng.uniform(2.0, 4.0, n), 1.0)

    mean = p.base * trend * weekly * yearly * promo * spikes
    # Gamma(shape=k, scale=mean/k) has expectation ``mean`` and CV 1/sqrt(k),
    # so k = 1/cv^2 gives exactly the requested noise level.
    k = 1.0 / p.noise_cv**2
    qty = rng.gamma(shape=k, scale=mean / k)

    return pd.DataFrame(
        {
            "date": dates,
            "ingredient": name,
            "qty_g": np.round(qty, 1),
            "promo": (promo > 1.0).astype(int),
        }
    )


def generate_consumption(
    n_days: int = 730,
    ingredients: tuple[str, ...] = DEFAULT_INGREDIENTS,
    start: str = "2023-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate the full long-format history for several ingredients.

    Args:
        n_days: Number of consecutive days to simulate.
        ingredients: Ingredient names (one series each).
        start: First calendar date (ISO format).
        seed: Seed for the single ``numpy`` Generator; same seed, same data.

    Returns:
        Long DataFrame sorted by ``ingredient`` then ``date``.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n_days, freq="D")
    frames = [generate_ingredient_series(name, dates, rng) for name in ingredients]
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["ingredient", "date"])
        .reset_index(drop=True)
    )


def save_consumption(df: pd.DataFrame, path: Path) -> Path:
    """Write the dataset to CSV, creating the parent folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_consumption(path: Path) -> pd.DataFrame:
    """Read the CSV written by :func:`save_consumption` with parsed dates."""
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["ingredient", "date"]).reset_index(drop=True)
