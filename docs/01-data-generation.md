# 1. Synthetic data generation

**Code:** [`src/pharma_forecast/data.py`](../src/pharma_forecast/data.py)

## What problem it solves

A compounding pharmacy uses active ingredients (in grams) every day to
prepare prescriptions. To build and test a forecaster we need years of daily
consumption per ingredient. Real data is private, so this project generates
it. **All data in this repository is synthetic**, created by a fixed-seed script.

## How it works

Each ingredient's expected daily consumption is a product of factors:

```
mean(t) = base × trend(t) × weekly(t) × yearly(t) × promo(t) × spike(t)
qty(t)  ~ Gamma(shape=k, scale=mean(t)/k),  k = 1 / cv²
```

| Factor | What it simulates | Parameters (random per ingredient) |
|---|---|---|
| `base` | typical grams/day | 40–600 |
| `trend` | slow growth or decline | −10 % to +40 % per year |
| `weekly` | busy weekdays, half Saturday, almost-idle Sunday | fixed profile `WEEKLY_PROFILE` |
| `yearly` | seasonal products (e.g. vitamin D in winter) | amplitude 0–20 %, random phase |
| `promo` | marketing campaigns lifting demand for 5–14 days | uplift 1.3–1.8×, every ~30–90 days |
| `spike` | rare bulk orders | ~1 % of days, 2–4× |
| Gamma noise | day-to-day randomness, always positive | CV 0.15–0.35 |

Why Gamma noise? It is positive (no negative grams) and its variance grows
with the mean, which is what real consumption looks like: a 500 g/day
ingredient fluctuates by more grams than a 50 g/day one.

```python
k = 1.0 / p.noise_cv**2
qty = rng.gamma(shape=k, scale=mean / k)  # E[qty] = mean, CV = noise_cv
```

One `numpy.random.Generator` seeded once drives every random choice, so the
same seed always produces exactly the same CSV (`tests/test_data_metrics.py`
checks this).

## Output

`data/consumption.csv`, long format, one row per ingredient per day:

| date | ingredient | qty_g | promo |
|---|---|---|---|
| 2023-01-01 | biotin | 41.7 | 0 |

Default: 730 days (2023-01-01 to 2024-12-30) × 8 ingredients = 5,840 rows.

## Run just this part

```bash
pharma-forecast generate-data                     # defaults
pharma-forecast generate-data --days 260 --ingredients 2   # tiny (CI)
```

## What to look at

- Open the CSV and plot one ingredient: you should see the weekly saw-tooth,
  promotion plateaus and an occasional spike.
- `test_weekly_seasonality_present` asserts Sundays average less than half of
  Mondays.
