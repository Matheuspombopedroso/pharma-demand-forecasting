# Benchmarks

Every number in the README comes from the runs below, executed once on the
machine described here. Nothing is rounded up; where the README shows fewer
decimals it truncates the values printed here.

## Hardware and software

| | |
|---|---|
| CPU | 11th Gen Intel Core i7-1185G7 @ 3.00 GHz, 4 cores / 8 logical processors |
| RAM | 39.7 GB |
| OS | Windows 11 Pro (10.0.26200) |
| Python | 3.12.14 |
| Key packages | ray 2.60.1, xgboost 3.4.1, mlflow 3.18.0, pandas 3.1.1, numpy 2.5.5 |

**Conditions:** laptop on AC power, with other unrelated processes running at
the same time (including other Python package installs). Timings are
single runs, not averages, and include that noise. Treat them as indicative.

## Dataset

`pharma-forecast generate-data` (seed 42): 730 days (2023-01-01 to
2024-12-30) × 8 ingredients = 5,840 rows. Walk-forward: 5 folds × 28 days;
folds 0–3 used for CV/tuning, fold 4 is the hold-out (origins 2024-11-03 to
2024-11-30, 28 days × 8 ingredients = 224 rows per horizon).

## Summary

### Forecast accuracy on the hold-out (default params, registry version 1)

Same 224 origins and the same metric code for every forecaster.

| Horizon | XGBoost WAPE | Seasonal mean 4w WAPE | Seasonal naive WAPE | Naive WAPE | XGBoost MAE (g) | Seasonal mean 4w MAE (g) |
|---|---|---|---|---|---|---|
| 7 d | 0.1063 | 0.1485 | 0.1720 | 0.3770 | 289.8 | 405.0 |
| 15 d | 0.0903 | 0.1259 | 0.1693 | 0.3803 | 534.8 | 745.2 |
| 30 d | 0.0765 | 0.1200 | 0.1704 | 0.3714 | 915.5 | 1435.7 |

WAPE gap vs the strongest baseline (seasonal mean of 4 weeks):
0.0422 (7 d), 0.0356 (15 d), 0.0435 (30 d).

### Ray Tune: parallel speed-up

Same 8 seeded random configurations, FIFO scheduler (every trial runs all 4
tuning folds × 3 horizons = 12 XGBoost fits), XGBoost single-threaded per
trial. Wall time is `tuner.fit()` only (Ray start-up excluded).

| `max_concurrent` | Trial-folds | `tuner.fit()` wall time | Speed-up | Best CV WAPE |
|---|---|---|---|---|
| 1 | 32 | 679.7 s | 1.00× | 0.0932 |
| 4 | 32 | 251.3 s | 2.70× | 0.0932 |

Both runs found the same best configuration (as expected from the seeded
search), so the parallel run did the same work 2.70× faster on 4 physical cores.

### Ray Tune with ASHA early stopping (24 configurations, 4 concurrent)

| | |
|---|---|
| Trials | 24 |
| Trial-folds actually run | 55 of 96 possible (ASHA stopped weak trials early) |
| `tuner.fit()` wall time | 382.4 s |
| Best CV WAPE | 0.0904 (default params CV WAPE: 0.0960) |
| Hold-out mean WAPE of the retrained tuned model | 0.0960 (default params: 0.0910) |

**Honest result:** the tuned parameters were better on the tuning folds but
worse on the hold-out. The promotion gate rejected that version (see the
registry flow below), and version 1 (default params) stayed champion. With
5,840 rows and 4 tuning folds, the CV differences between good configurations
are about as large as the fold-to-fold noise, so tuning here mainly shows the
infrastructure working, not a reliable accuracy gain.

### Registry flow

`register` v1 → champion (bootstrap) · `register` v2 → challenger ·
`promote` → **rejected** (0.0960 ≥ 0.0910) · `promote --version 2 --force` →
v2 champion, v1 previous · `rollback` → v1 champion again, v2 tagged
`rolled_back=true` · `predict` serves v1. Outputs below.

## Raw output

Local absolute paths were replaced with `<repo>` / `<tmp>` and ANSI colour
codes were stripped; nothing else was edited. The `(raylet) Stack (most
recent call first)` lines are printed by Ray's worker processes on shutdown
on Windows; they did not affect results. The `real` times from `time` include
Ray start-up, feature building and MLflow logging, which is why they exceed
the `tuner.fit()` wall times.

Notes on `07_registry_flow`, which ran with an earlier revision of the code:

- `status` showed `-` in the aliases column: `search_model_versions()` does not
  return aliases. Fixed before `08` (aliases are now read from the registered
  model); the alias state itself was correct, as `08` shows.
- `rollback` raised a traceback. That outcome is expected: the promotion had
  been rejected, so there was no `@previous` version to roll back to. The CLI
  now prints `Rollback not possible: ...` and exits with code 1 instead.
- In `08`, the first `predict` printed an MLflow warning that `psutil` was not
  installed (MLflow records it as a model requirement). It is now pinned in
  `requirements.txt`.

### `01_generate`

```text
Wrote 5840 rows (8 ingredients) to <repo>\data\consumption.csv
```

### `02_train_default`

```text
+ .venv/Scripts/pharma-forecast train
MLflow run: 2863e01bff294976b2863a0e00ad91fd  cv_wape=0.0960
Hold-out comparison (lower is better):
 horizon       forecaster       mae   wape   mape  n_rows
       7          xgboost  289.8165 0.1063 0.1041     224
       7            naive 1028.1397 0.3770 0.3909     224
       7   seasonal_naive  468.9388 0.1720 0.1703     224
       7 seasonal_mean_4w  404.9819 0.1485 0.1403     224
      15          xgboost  534.7796 0.0903 0.0894     224
      15            naive 2251.9254 0.3803 0.3924     224
      15   seasonal_naive 1002.3429 0.1693 0.1658     224
      15 seasonal_mean_4w  745.2092 0.1259 0.1182     224
      30          xgboost  915.5362 0.0765 0.0757     224
      30            naive 4444.4112 0.3714 0.3957     224
      30   seasonal_naive 2039.3911 0.1704 0.1692     224
      30 seasonal_mean_4w 1435.6798 0.1200 0.1142     224
real	2m52.263s
user	0m0.000s
sys	0m0.015s
```

### `03_register_v1`

```text
Successfully registered model 'pharma-demand-forecaster'.
Created version '1' of model 'pharma-demand-forecaster'.
Registered run 2863e01bff294976b2863a0e00ad91fd as version 1
```

### `04_tune_fifo_c1`

```text
+ .venv/Scripts/pharma-forecast tune --num-samples 8 --max-concurrent 1 --no-asha --no-train
(raylet) Stack (most recent call first):
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\worker.py", line 629 in job_logging_config
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\worker.py", line 2201 in _should_ignore_worker_log_prefix
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\worker.py", line 2830 in disconnect
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\worker.py", line 2132 in shutdown
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\worker.py", line 1178 in wrapper
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\client_mode_hook.py", line 134 in wrapper
+--------------------------------------------------------+
| Configuration for experiment     xgb_search            |
+--------------------------------------------------------+
| Search algorithm                 BasicVariantGenerator |
| Scheduler                        FIFOScheduler         |
| Number of trials                 8                     |
+--------------------------------------------------------+
View detailed results here: <repo>/ray_results/xgb_search
To visualize your results with TensorBoard, run: `tensorboard --logdir <tmp>/ray/session_2026-11-25_15-58-05_788504_19060/artifacts/2026-11-25_15-58-21/xgb_search/driver_artifacts`
Tuned 8 trials (32 trial-folds) in 679.7s (max_concurrent=1); best cv_wape=0.0932
Best params -> outputs\best_params.json: {"n_estimators": 331, "max_depth": 3, "learning_rate": 0.050428245390731465, "subsample": 0.6907637396203536, "colsample_bytree": 0.8349069973412552, "min_child_weight": 9.305886458574284, "reg_lambda": 4.627606177115527}
real	12m20.590s
user	0m0.031s
sys	0m0.016s
```

### `05_tune_fifo_c4`

```text
+ .venv/Scripts/pharma-forecast tune --num-samples 8 --max-concurrent 4 --no-asha --no-train
(raylet) Stack (most recent call first):
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\worker.py", line 629 in job_logging_config
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\worker.py", line 2201 in _should_ignore_worker_log_prefix
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\worker.py", line 2830 in disconnect
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\worker.py", line 2132 in shutdown
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\worker.py", line 1178 in wrapper
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\client_mode_hook.py", line 134 in wrapper
(raylet) Stack (most recent call first): [repeated 3x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\worker.py", line 629 in job_logging_config [repeated 3x across cluster]
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\worker.py", line 2201 in _should_ignore_worker_log_prefix [repeated 3x across cluster]
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\worker.py", line 2830 in disconnect [repeated 3x across cluster]
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\worker.py", line 2132 in shutdown [repeated 3x across cluster]
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\worker.py", line 1178 in wrapper [repeated 3x across cluster]
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\client_mode_hook.py", line 134 in wrapper [repeated 3x across cluster]
+--------------------------------------------------------+
| Configuration for experiment     xgb_search            |
+--------------------------------------------------------+
| Search algorithm                 BasicVariantGenerator |
| Scheduler                        FIFOScheduler         |
| Number of trials                 8                     |
+--------------------------------------------------------+
View detailed results here: <repo>/ray_results/xgb_search
To visualize your results with TensorBoard, run: `tensorboard --logdir <tmp>/ray/session_2026-11-25_16-10-03_089462_8400/artifacts/2026-11-25_16-10-22/xgb_search/driver_artifacts`
Tuned 8 trials (32 trial-folds) in 251.3s (max_concurrent=4); best cv_wape=0.0932
Best params -> outputs\best_params.json: {"n_estimators": 331, "max_depth": 3, "learning_rate": 0.050428245390731465, "subsample": 0.6907637396203536, "colsample_bytree": 0.8349069973412552, "min_child_weight": 9.305886458574284, "reg_lambda": 4.627606177115527}
real	4m52.951s
user	0m0.000s
sys	0m0.015s
```

### `06_tune_asha_c4_and_train`

```text
+ .venv/Scripts/pharma-forecast tune --num-samples 24 --max-concurrent 4
(raylet) Stack (most recent call first):
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\worker.py", line 629 in job_logging_config
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\worker.py", line 2201 in _should_ignore_worker_log_prefix
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\worker.py", line 2830 in disconnect
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\worker.py", line 2132 in shutdown
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\worker.py", line 1178 in wrapper
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\client_mode_hook.py", line 134 in wrapper
(raylet) Stack (most recent call first): [repeated 3x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\worker.py", line 629 in job_logging_config [repeated 3x across cluster]
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\worker.py", line 2201 in _should_ignore_worker_log_prefix [repeated 3x across cluster]
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\worker.py", line 2830 in disconnect [repeated 3x across cluster]
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\worker.py", line 2132 in shutdown [repeated 3x across cluster]
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\worker.py", line 1178 in wrapper [repeated 3x across cluster]
(raylet)   File "<repo>\.venv\Lib\site-packages\ray\_private\client_mode_hook.py", line 134 in wrapper [repeated 3x across cluster]
+----------------------------------------------------------+
| Configuration for experiment     xgb_search              |
+----------------------------------------------------------+
| Search algorithm                 BasicVariantGenerator   |
| Scheduler                        AsyncHyperBandScheduler |
| Number of trials                 24                      |
+----------------------------------------------------------+
View detailed results here: <repo>/ray_results/xgb_search
To visualize your results with TensorBoard, run: `tensorboard --logdir <tmp>/ray/session_2026-11-25_16-14-59_097776_27088/artifacts/2026-11-25_16-15-13/xgb_search/driver_artifacts`
Tuned 24 trials (55 trial-folds) in 382.4s (max_concurrent=4); best cv_wape=0.0904
Best params -> outputs\best_params.json: {"n_estimators": 194, "max_depth": 4, "learning_rate": 0.016348569248088956, "subsample": 0.6413611870890207, "colsample_bytree": 0.793822286088856, "min_child_weight": 4.241266402200836, "reg_lambda": 7.083375039486971}
MLflow run (tuned params): 948e08e9b1604d08b801eed7560fc5b8
 horizon       forecaster       mae   wape   mape  n_rows
       7          xgboost  317.1258 0.1163 0.1104     224
       7            naive 1028.1397 0.3770 0.3909     224
       7   seasonal_naive  468.9388 0.1720 0.1703     224
       7 seasonal_mean_4w  404.9819 0.1485 0.1403     224
      15          xgboost  557.8044 0.0942 0.0902     224
      15            naive 2251.9254 0.3803 0.3924     224
      15   seasonal_naive 1002.3429 0.1693 0.1658     224
      15 seasonal_mean_4w  745.2092 0.1259 0.1182     224
      30          xgboost  928.9571 0.0776 0.0754     224
      30            naive 4444.4112 0.3714 0.3957     224
      30   seasonal_naive 2039.3911 0.1704 0.1692     224
      30 seasonal_mean_4w 1435.6798 0.1200 0.1142     224
real	8m1.000s
user	0m0.015s
sys	0m0.031s
```

### `07_registry_flow`

```text
+ .venv/Scripts/pharma-forecast register
Registered model 'pharma-demand-forecaster' already exists. Creating a new version of this model...
Created version '2' of model 'pharma-demand-forecaster'.
Registered run 948e08e9b1604d08b801eed7560fc5b8 as version 2
+ .venv/Scripts/pharma-forecast status
version aliases holdout_xgboost_wape_mean rolled_back                           run_id
      1       -                  0.091035       false 2863e01bff294976b2863a0e00ad91fd
      2       -                  0.096043       false 948e08e9b1604d08b801eed7560fc5b8
+ .venv/Scripts/pharma-forecast promote
Rejected: hold-out WAPE 0.0960 is not better than champion's 0.0910. Use --force to override.
candidate v2 wape=0.0960; previous champion v1 wape=0.09103530397714642
+ .venv/Scripts/pharma-forecast status
version aliases holdout_xgboost_wape_mean rolled_back                           run_id
      1       -                  0.091035       false 2863e01bff294976b2863a0e00ad91fd
      2       -                  0.096043       false 948e08e9b1604d08b801eed7560fc5b8
+ .venv/Scripts/pharma-forecast predict --out outputs/forecast.csv
     ingredient origin_date model_version  forecast_7  forecast_15  forecast_30
         biotin  2024-12-30             1      1621.7       3745.2       7055.2
       caffeine  2024-12-30             1      4670.5      10919.6      21314.7
    finasteride  2024-12-30             1       861.4       1547.8       2950.5
hyaluronic_acid  2024-12-30             1      1978.0       4348.6       8776.9
      melatonin  2024-12-30             1      1654.7       3262.9       6755.9
      minoxidil  2024-12-30             1      3352.9       7603.7      14949.5
        omega_3  2024-12-30             1      5558.3      12172.8      25468.7
     vitamin_d3  2024-12-30             1      1747.7       3647.5       7819.3
Saved to outputs/forecast.csv
+ .venv/Scripts/pharma-forecast rollback
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "<repo>\.venv\Scripts\pharma-forecast.exe\__main__.py", line 5, in <module>
  File "<repo>\src\pharma_forecast\cli.py", line 218, in main
    args.func(args, get_settings())
  File "<repo>\src\pharma_forecast\cli.py", line 145, in cmd_rollback
    old, new = rollback(s.mlflow_tracking_uri, args.version)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<repo>\src\pharma_forecast\registry.py", line 174, in rollback
    raise ValueError("Nothing to roll back to: no @champion or no @previous version.")
ValueError: Nothing to roll back to: no @champion or no @previous version.
+ .venv/Scripts/pharma-forecast status
version aliases holdout_xgboost_wape_mean rolled_back                           run_id
      1       -                  0.091035       false 2863e01bff294976b2863a0e00ad91fd
      2       -                  0.096043       false 948e08e9b1604d08b801eed7560fc5b8
+ .venv/Scripts/pharma-forecast predict
     ingredient origin_date model_version  forecast_7  forecast_15  forecast_30
         biotin  2024-12-30             1      1621.7       3745.2       7055.2
       caffeine  2024-12-30             1      4670.5      10919.6      21314.7
    finasteride  2024-12-30             1       861.4       1547.8       2950.5
hyaluronic_acid  2024-12-30             1      1978.0       4348.6       8776.9
      melatonin  2024-12-30             1      1654.7       3262.9       6755.9
      minoxidil  2024-12-30             1      3352.9       7603.7      14949.5
        omega_3  2024-12-30             1      5558.3      12172.8      25468.7
     vitamin_d3  2024-12-30             1      1747.7       3647.5       7819.3
```

### `08_registry_force_and_rollback`

```text
++ .venv/Scripts/pharma-forecast status
version    aliases holdout_xgboost_wape_mean rolled_back                           run_id
      1   champion                  0.091035       false 2863e01bff294976b2863a0e00ad91fd
      2 challenger                  0.096043       false 948e08e9b1604d08b801eed7560fc5b8
++ .venv/Scripts/pharma-forecast promote --version 2 --force
Promoted (forced).
candidate v2 wape=0.0960; previous champion v1 wape=0.09103530397714642
++ .venv/Scripts/pharma-forecast status
version  aliases holdout_xgboost_wape_mean rolled_back                           run_id
      1 previous                  0.091035       false 2863e01bff294976b2863a0e00ad91fd
      2 champion                  0.096043       false 948e08e9b1604d08b801eed7560fc5b8
++ .venv/Scripts/pharma-forecast predict
 - psutil (current: uninstalled, required: psutil==7.2.2)
To fix the mismatches, call `mlflow.pyfunc.get_model_dependencies(model_uri)` to fetch the model's environment and install dependencies using the resulting environment file.
     ingredient origin_date model_version  forecast_7  forecast_15  forecast_30
         biotin  2024-12-30             2      1580.7       3446.2       7021.8
       caffeine  2024-12-30             2      4764.6      10557.0      21268.0
    finasteride  2024-12-30             2       793.0       1471.7       2815.8
hyaluronic_acid  2024-12-30             2      1957.6       4342.4       9086.2
      melatonin  2024-12-30             2      1633.3       3466.7       6929.3
      minoxidil  2024-12-30             2      3489.6       7781.6      15593.1
        omega_3  2024-12-30             2      5489.5      12197.9      24925.1
     vitamin_d3  2024-12-30             2      1644.8       3639.3       7549.3
++ .venv/Scripts/pharma-forecast rollback
Rolled back: champion v2 -> v1
++ .venv/Scripts/pharma-forecast status
version  aliases holdout_xgboost_wape_mean rolled_back                           run_id
      1 champion                  0.091035       false 2863e01bff294976b2863a0e00ad91fd
      2        -                  0.096043        true 948e08e9b1604d08b801eed7560fc5b8
++ .venv/Scripts/pharma-forecast predict
     ingredient origin_date model_version  forecast_7  forecast_15  forecast_30
         biotin  2024-12-30             1      1621.7       3745.2       7055.2
       caffeine  2024-12-30             1      4670.5      10919.6      21314.7
    finasteride  2024-12-30             1       861.4       1547.8       2950.5
hyaluronic_acid  2024-12-30             1      1978.0       4348.6       8776.9
      melatonin  2024-12-30             1      1654.7       3262.9       6755.9
      minoxidil  2024-12-30             1      3352.9       7603.7      14949.5
        omega_3  2024-12-30             1      5558.3      12172.8      25468.7
     vitamin_d3  2024-12-30             1      1747.7       3647.5       7819.3
```
