"""Walk-forward split tests: ordering, expansion and no label leakage."""

from __future__ import annotations

import pandas as pd
import pytest

from pharma_forecast.validation import make_folds, split_tuning_holdout

DATES = pd.Series(pd.date_range("2023-01-01", periods=400, freq="D"))
LAST = DATES.max()


@pytest.mark.parametrize("horizon", [7, 15, 30])
def test_train_labels_end_before_test_starts(horizon: int) -> None:
    for fold in make_folds(LAST, max_horizon=30, n_folds=5, test_days=28):
        train_dates = DATES[fold.train_mask(DATES, horizon)]
        test_dates = DATES[fold.test_mask(DATES)]
        # Every training target window (t+1..t+H) is fully observed by the cutoff.
        assert (train_dates + pd.Timedelta(days=horizon)).max() <= fold.cutoff
        # Test origins come strictly after the cutoff.
        assert test_dates.min() > fold.cutoff
        assert test_dates.min() == fold.test_start
        assert len(test_dates) == 28


def test_folds_are_chronological_and_expanding() -> None:
    folds = make_folds(LAST, max_horizon=30, n_folds=5, test_days=28)
    train_sizes = [fold.train_mask(DATES, 7).sum() for fold in folds]
    assert train_sizes == sorted(train_sizes)
    assert len(set(train_sizes)) == len(train_sizes)  # strictly growing
    for a, b in zip(folds, folds[1:], strict=False):
        assert b.cutoff == a.test_end  # test windows tile without gaps/overlap
    # All train windows start at the first date (expanding, not sliding).
    for fold in folds:
        assert DATES[fold.train_mask(DATES, 7)].min() == DATES.min()


def test_last_fold_ends_where_longest_target_is_still_observed() -> None:
    folds = make_folds(LAST, max_horizon=30, n_folds=3, test_days=14)
    assert folds[-1].test_end + pd.Timedelta(days=30) == LAST


def test_holdout_is_last_fold_and_excluded_from_tuning() -> None:
    folds = make_folds(LAST, max_horizon=30, n_folds=4, test_days=28)
    tuning, holdout = split_tuning_holdout(folds)
    assert holdout == folds[-1]
    assert holdout not in tuning
    assert all(f.test_end <= holdout.cutoff for f in tuning)


def test_needs_at_least_two_folds() -> None:
    with pytest.raises(ValueError):
        split_tuning_holdout(make_folds(LAST, 30, 1, 28))
