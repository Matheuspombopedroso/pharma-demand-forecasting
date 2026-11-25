"""Training step: evaluate, fit the final model and log everything to MLflow.

Flow of :func:`run_training`:

1. Build the feature table (features, targets, baseline forecasts).
2. Walk-forward cross-validation on the tuning folds -> ``cv_*`` metrics.
3. Hold-out evaluation of XGBoost **and** both baselines on the last fold
   -> ``holdout_*`` metrics and a comparison table.
4. Refit on every labelled row (the hold-out is only for measurement; the
   deployed model should learn from the most recent data too).
5. Log params, metrics, CSV artifacts and the model (as an MLflow pyfunc)
   to a single MLflow run, and return its id for ``register``.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.models import infer_signature

from pharma_forecast.config import HORIZONS, Settings
from pharma_forecast.evaluation import cross_validate, holdout_report, prepare_table
from pharma_forecast.features import feature_columns
from pharma_forecast.model import DEFAULT_PARAMS, ForecasterPyfunc, MultiHorizonForecaster
from pharma_forecast.tracking import setup_mlflow
from pharma_forecast.validation import make_folds, split_tuning_holdout


@dataclass
class TrainResult:
    """What a training run produced.

    Attributes:
        run_id: MLflow run id (input for ``register``).
        cv_metrics: Mean walk-forward metrics on the tuning folds.
        holdout: Hold-out comparison table (model vs baselines).
    """

    run_id: str
    cv_metrics: dict[str, float]
    holdout: pd.DataFrame


def holdout_metrics_flat(report: pd.DataFrame) -> dict[str, float]:
    """Flatten the hold-out table into MLflow metric names.

    Produces e.g. ``holdout_xgboost_wape_h7`` and, for convenience,
    ``holdout_xgboost_wape_mean`` (the number used to compare registry
    versions during promotion) and ``wape_gain_vs_snaive_h7`` /
    ``wape_gain_vs_smean4_h7`` (positive = model better than that baseline,
    in WAPE points).
    """
    out: dict[str, float] = {}
    for r in report.itertuples():
        for metric in ("mae", "wape", "mape"):
            out[f"holdout_{r.forecaster}_{metric}_h{r.horizon}"] = float(getattr(r, metric))
    for name in report["forecaster"].unique():
        out[f"holdout_{name}_wape_mean"] = float(
            report.loc[report["forecaster"] == name, "wape"].mean()
        )
    for h in report["horizon"].unique():
        out[f"wape_gain_vs_snaive_h{h}"] = (
            out[f"holdout_seasonal_naive_wape_h{h}"] - out[f"holdout_xgboost_wape_h{h}"]
        )
        out[f"wape_gain_vs_smean4_h{h}"] = (
            out[f"holdout_seasonal_mean_4w_wape_h{h}"] - out[f"holdout_xgboost_wape_h{h}"]
        )
    return out


def run_training(
    raw: pd.DataFrame,
    settings: Settings,
    params: dict[str, Any] | None = None,
    n_folds: int = 5,
    test_days: int = 28,
    run_name: str = "train",
    extra_tags: dict[str, str] | None = None,
) -> TrainResult:
    """Evaluate and fit a forecaster, logging the run to MLflow.

    Args:
        raw: Long consumption DataFrame (``data.load_consumption``).
        settings: Paths and MLflow configuration.
        params: XGBoost hyper-parameters; defaults to ``DEFAULT_PARAMS``.
        n_folds: Walk-forward folds including the final hold-out.
        test_days: Days per test window.
        run_name: MLflow run name.
        extra_tags: Extra MLflow tags (e.g. ``source=tune``).

    Returns:
        A :class:`TrainResult`.
    """
    params = dict(params or DEFAULT_PARAMS)
    table = prepare_table(raw, HORIZONS)
    folds = make_folds(raw["date"].max(), max(HORIZONS), n_folds, test_days)
    tuning_folds, holdout = split_tuning_holdout(folds)

    cv = cross_validate(table, tuning_folds, params, HORIZONS, settings.seed)
    report, preds = holdout_report(table, holdout, params, HORIZONS, settings.seed)

    # Final model: all labelled rows, all CPU threads (no other job competes).
    final = MultiHorizonForecaster(params, HORIZONS, settings.seed, n_jobs=os.cpu_count() or 1)
    final.fit(table)

    setup_mlflow(settings)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"stage": "train", **(extra_tags or {})})
        mlflow.log_params(params)
        mlflow.log_params(
            {
                "n_folds": n_folds,
                "test_days": test_days,
                "holdout_start": str(holdout.test_start.date()),
                "holdout_end": str(holdout.test_end.date()),
                "n_ingredients": raw["ingredient"].nunique(),
                "n_days": raw["date"].nunique(),
                "seed": settings.seed,
            }
        )
        mlflow.log_metrics({f"cv_{k}" if k != "cv_wape" else k: v for k, v in cv.items()})
        mlflow.log_metrics(holdout_metrics_flat(report))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report.to_csv(tmp_path / "holdout_report.csv", index=False)
            preds.to_csv(tmp_path / "holdout_predictions.csv", index=False)
            importance = pd.DataFrame(
                {
                    f"gain_h{h}": pd.Series(
                        final.models[h].get_booster().get_score(importance_type="gain")
                    )
                    for h in HORIZONS
                }
            ).sort_values(f"gain_h{HORIZONS[0]}", ascending=False)
            importance.to_csv(tmp_path / "feature_importance.csv")
            (tmp_path / "params.json").write_text(json.dumps(params, indent=2))
            mlflow.log_artifacts(tmp, artifact_path="reports")

            model_dir = final.save(tmp_path / "model_dir")
            example = table[feature_columns() + ["scale"]].tail(5)
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=ForecasterPyfunc(),
                artifacts={"model_dir": str(model_dir)},
                signature=infer_signature(example, final.predict_grams(example)),
                input_example=example,
            )
            # Lets `register` point at this exact model (see registry.register).
            mlflow.set_tag("model_uri", info.model_uri)
        return TrainResult(run.info.run_id, cv, report)
