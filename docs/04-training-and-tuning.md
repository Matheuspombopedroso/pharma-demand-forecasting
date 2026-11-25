# 4. Training, Ray Tune search and MLflow tracking

**Code:** [`model.py`](../src/pharma_forecast/model.py),
[`train.py`](../src/pharma_forecast/train.py),
[`tune.py`](../src/pharma_forecast/tune.py),
[`tracking.py`](../src/pharma_forecast/tracking.py) ·
**Tests:** [`tests/test_tune.py`](../tests/test_tune.py),
[`tests/test_pipeline.py`](../tests/test_pipeline.py)

## The model

`MultiHorizonForecaster` holds one `XGBRegressor` per horizon (the "direct"
multi-horizon strategy). The alternative, a recursive one-day model fed with
its own predictions, compounds errors across 30 steps. The cost of the direct
strategy is three models instead of one.

- `objective="reg:absoluteerror"`: L1 loss, aligned with MAE/WAPE.
- `tree_method="hist"`: fast histogram algorithm on CPU.
- Predictions are clipped at 0 and multiplied back by `H × scale` to grams.

## `train`: one MLflow run

```
prepare_table → cross_validate(tuning folds) → holdout_report(hold-out)
             → refit on all labelled rows → log to MLflow
```

What gets logged:

| Kind | Examples |
|---|---|
| params | XGBoost params, `n_folds`, `test_days`, hold-out dates, data size, seed |
| metrics | `cv_wape`, `cv_wape_h7`…, `holdout_xgboost_wape_h30`, `holdout_seasonal_naive_mae_h7`, `wape_gain_vs_snaive_h15`, `holdout_xgboost_wape_mean` |
| artifacts | `reports/holdout_report.csv`, `holdout_predictions.csv`, `feature_importance.csv`, `params.json` |
| model | MLflow pyfunc `model/` with boosters as JSON, signature and input example |

The model is refit on all labelled data after evaluation. The hold-out is
for *measuring*; the deployed model should still learn from the latest
weeks.

## `tune`: Ray Tune distributed search

```python
tune.Tuner(
    tune.with_resources(tune.with_parameters(_trainable, table=table, folds=tuning_folds, ...),
                        {"cpu": 1}),
    param_space=SEARCH_SPACE,
    tune_config=tune.TuneConfig(metric="cv_wape", mode="min", num_samples=...,
                                max_concurrent_trials=...,
                                scheduler=ASHAScheduler(time_attr="folds_done", ...)),
)
```

Key choices:

- **`tune.with_parameters`** puts the feature table into Ray's object store
  once; trials get a reference instead of a pickled copy each.
- **1 CPU per trial and XGBoost `n_jobs=1`**. Parallelism comes from running
  trials side by side. If XGBoost also used every core, 4 trials × 8 threads
  would oversubscribe the CPU and the measured speed-up would mean nothing.
- **ASHA early stopping**: each trial reports its running mean WAPE after
  every walk-forward fold (`folds_done` = 1, 2, 3, 4). ASHA stops trials that
  are clearly worse than their peers at the same fold count, so the budget
  goes to promising configurations.
- **Only tuning folds**: the hold-out stays untouched (see part 3).
- **Short trial directory names** (`t_<id>`): Ray's default names include
  the whole config and can exceed Windows' 260-character path limit.
- **MLflow logging from the driver**: after `tuner.fit()`, one parent run
  `tune` gets a nested child run per trial. Writing from the parallel workers
  would mean concurrent writers on SQLite.

The same trainable runs on a multi-node Ray cluster by changing only the
`ray.init` address. In this repo it was only run on a single machine.

After the search, `tune` saves `outputs/best_params.json` and runs `train`
with those params (tagged `source=tune`), producing the run to register.

### Ray Train

Ray Train's `XGBoostTrainer` (data-parallel training of *one* model split
across workers) was **not** used. The whole table is ~5,800 rows and fits
easily in one process, so sharding a single fit would only add coordination
overhead. At this size the useful parallelism is across trials, which is what
Ray Tune provides. Ray Train becomes relevant when one training set no longer
fits on one machine (listed under Future work in the README).

## Run just this part

```bash
pharma-forecast train                                      # default params
pharma-forecast tune --num-samples 20 --max-concurrent 4   # search + train
make mlflow-ui                                             # http://localhost:5000
```

## What to look at

- In the MLflow UI, the `tune` run expands into one child per trial, with
  `folds_done < 4` for the trials ASHA stopped early.
- In the `train-tuned` run, compare `holdout_xgboost_wape_h*` against
  `holdout_seasonal_naive_wape_h*`, and open `reports/holdout_report.csv`.
