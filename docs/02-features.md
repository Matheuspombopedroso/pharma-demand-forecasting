# 2. Leakage-safe features, targets and baselines

**Code:** [`features.py`](../src/pharma_forecast/features.py),
[`baselines.py`](../src/pharma_forecast/baselines.py) ·
**Tests:** [`tests/test_features.py`](../tests/test_features.py)

## What problem it solves

The model needs inputs that summarise recent history, and a target that is
what purchasing actually needs. The usual way to get impressive but fake
accuracy in time series is **leakage**: a feature that silently peeks at the
future. This part is built so that cannot happen, and a test proves it.

## The setup: one row = one forecast origin

A row is "the end of day *t* for ingredient *i*". At that moment we know
`qty(≤ t)` and nothing else. The target for horizon *H* is the total over the
next *H* days:

```
target_H(t) = qty(t+1) + ... + qty(t+H)        H ∈ {7, 15, 30}
```

```python
# trailing H-sum at row t+H == qty(t+1..t+H); shift(-H) aligns it with origin t
fwd = g.transform(lambda s, h=h: s.rolling(h).sum().shift(-h))
```

## Features (all look backwards only)

| Group | Columns | Why |
|---|---|---|
| Lags | `lag_0,1,2,3,6,7,13,14,27,28` | recent level + same weekday 1/2/4 weeks ago |
| Rolling | `roll_mean_*`, `roll_std_*` for 7/14/28/56/91 days | level and volatility at several speeds |
| Momentum | `momentum_7_91` | last week vs last quarter (trend) |
| Calendar | `dow_next`, `doy_sin`, `doy_cos` | weekday of the first forecast day, season. The calendar is known in advance, so using t+1 is legitimate |
| Promotion | `promo_today`, `promo_last7` | campaign running at the origin |
| Identity | `ingredient_id` | CRC32 of the name: stable across processes, no lookup table |
| Scale | `log_scale` | absolute volume level |

Everything is computed inside `groupby("ingredient")` with `shift(k ≥ 0)` and
right-aligned `rolling()` windows ending at *t*.

### Scale normalisation (trade-off)

Ingredients differ ~15× in volume. Lags and rolling stats are divided by
`scale` = the trailing 28-day mean, and the model predicts
`target / (H × scale)`. That lets **one global model** learn weekday and
promotion effects from all ingredients together instead of eight small
models with ~600 rows each. The cost: when the 28-day mean is distorted (e.g.
right after a spike), all inputs are distorted with it.

## The no-leakage test

```python
original = build_features(raw)
scrambled = build_features(_scramble_after(raw, cutoff))  # random future values
assert_frame_equal(original[past, cols], scrambled[past, cols])
```

If any feature at or before the cut-off changed when only the future changed,
that feature reads the future and the test fails. The same test runs on the
baseline forecasts. `test_feature_columns_exclude_targets_and_raw_quantity`
additionally checks no `target_*` column is in the model's input list.

## Baselines

| Name | Forecast for the next H days | Notes |
|---|---|---|
| naive | `H × qty(t)` | weak on weekly data: a Sunday origin predicts ~0 |
| seasonal naive | repeat the last 7 days as a pattern | strong baseline for weekly seasonality |

```python
last_week = [g.shift(6 - k) for k in range(7)]  # qty(t-6) ... qty(t)
q, r = divmod(h, 7)
snaive = q * sum(last_week) + sum(last_week[:r])
```

Baselines are columns in the *same* table as the features, so the model and
baselines are scored on identical rows.

## Run just this part

```bash
pytest tests/test_features.py -v
```

## What to look at

All five tests pass; try changing a `shift(lag)` to `shift(-1)` in
`build_features` and watch the leakage test fail.
