# 3. Walk-forward validation and the hold-out

**Code:** [`validation.py`](../src/pharma_forecast/validation.py),
[`evaluation.py`](../src/pharma_forecast/evaluation.py),
[`metrics.py`](../src/pharma_forecast/metrics.py) ·
**Tests:** [`tests/test_validation.py`](../tests/test_validation.py)

## What problem it solves

Random K-fold puts tomorrow in the training set and yesterday in the test
set, so the scores look better than what you get in production. Walk-forward
validation replays history: train with everything known at a cut-off date,
forecast the following weeks, move the cut-off forward, repeat.

## How a fold is defined

```
          cutoff c                  test_end
 ─────────────┼──────────────────────┼──────────►
 train origins t with t + H ≤ c │ test origins c < t ≤ c + test_days
```

- **Train rows**: `t + H ≤ c`. An origin's target covers days t+1..t+H, so a
  training row near the cut-off would otherwise have a label that includes
  days *after* the cut-off. This is a subtle leak that plain date splits
  miss, and it gets worse as H grows.
- **Test rows**: `c < t ≤ c + test_days`.
- **Expanding window**: training always starts at the first date.
- Cut-offs are computed from the **longest** horizon (30), so 7/15/30-day
  results are measured on the same origins.

```python
def train_mask(self, dates, horizon):
    return (dates + pd.Timedelta(days=horizon) <= self.cutoff).to_numpy()
```

## Tuning folds vs hold-out

Default: 5 folds × 28 days. Folds 0–3 are **tuning folds** (Ray Tune
optimises on them). Fold 4 is the **hold-out**: never seen during the search
and used only for the final model-vs-baseline comparison. Choosing
hyper-parameters and reporting on the same folds would bias the result
towards the model.

## Metrics

| Metric | Formula | Use |
|---|---|---|
| MAE | mean \|y − ŷ\| (grams) | readable but scale-dependent |
| WAPE | Σ\|y − ŷ\| / Σ\|y\| | headline metric and tuning objective |
| MAPE | mean \|y − ŷ\| / \|y\| (y > 0) | common, over-weights small rows |

The same functions score XGBoost and both baselines.

## Run just this part

```bash
pytest tests/test_validation.py -v
```

The tests assert: training labels end by the cut-off for every horizon, test
windows tile without gaps, training sets strictly grow, and the hold-out is
excluded from the tuning folds.
