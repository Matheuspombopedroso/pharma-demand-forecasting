"""Walk-forward (expanding window) validation splits.

Why not random K-fold?
----------------------
Random folds put future days in the training set and past days in the test
set, which overstates accuracy for a time series. Walk-forward validation
replays history the way the model is used: train on everything known at a
cut-off date, then forecast the following weeks.

How a fold is built
-------------------
For a cut-off date ``c`` and horizon ``H``:

* **train** rows are origins ``t`` with ``t + H <= c``. The target of origin
  t spans days t+1..t+H, so this rule guarantees every training label was
  fully observed by the cut-off (no label leakage through overlapping
  target windows).
* **test** rows are origins with ``c < t <= c + test_days``.

Cut-offs step forward by ``test_days`` and the training window always starts
at the first date (expanding window). The cut-offs are derived from the
longest horizon so that every horizon is evaluated on the *same* origins,
which keeps 7/15/30-day results comparable.

Hold-out
--------
The last fold is reserved as a final hold-out: Ray Tune only sees the
earlier folds, and the model-vs-baseline numbers in the README come from the
hold-out. Tuning and reporting on the same folds would bias the comparison in
the model's favour.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """One walk-forward split.

    Attributes:
        index: 0-based fold number (chronological).
        cutoff: Last day whose consumption is known when training.
        test_start: First origin date in the test window (cutoff + 1 day).
        test_end: Last origin date in the test window (inclusive).
    """

    index: int
    cutoff: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def train_mask(self, dates: pd.Series, horizon: int) -> np.ndarray:
        """Boolean mask of training origins: label window ends by the cut-off."""
        return (dates + pd.Timedelta(days=horizon) <= self.cutoff).to_numpy()

    def test_mask(self, dates: pd.Series) -> np.ndarray:
        """Boolean mask of test origins inside (cutoff, test_end]."""
        return ((dates > self.cutoff) & (dates <= self.test_end)).to_numpy()


def make_folds(
    last_date: pd.Timestamp,
    max_horizon: int,
    n_folds: int,
    test_days: int,
) -> list[Fold]:
    """Build ``n_folds`` chronological walk-forward folds.

    Args:
        last_date: Last day of observed consumption in the data.
        max_horizon: Longest horizon; origins after ``last_date - max_horizon``
            have no complete target and cannot be evaluated.
        n_folds: Number of folds (the last one is the hold-out).
        test_days: Length of each test window in days (also the step size).

    Returns:
        Folds ordered oldest to newest; ``folds[-1]`` is the hold-out.
    """
    last_origin = pd.Timestamp(last_date) - pd.Timedelta(days=max_horizon)
    folds = []
    for i in range(n_folds):
        # Fold n_folds-1 ends at last_origin; each earlier fold is test_days earlier.
        steps_back = n_folds - 1 - i
        test_end = last_origin - pd.Timedelta(days=steps_back * test_days)
        cutoff = test_end - pd.Timedelta(days=test_days)
        folds.append(
            Fold(
                index=i,
                cutoff=cutoff,
                test_start=cutoff + pd.Timedelta(days=1),
                test_end=test_end,
            )
        )
    return folds


def split_tuning_holdout(folds: list[Fold]) -> tuple[list[Fold], Fold]:
    """Separate the tuning folds from the final hold-out fold."""
    if len(folds) < 2:
        raise ValueError("Need at least 2 folds: >=1 for tuning and 1 hold-out.")
    return folds[:-1], folds[-1]
