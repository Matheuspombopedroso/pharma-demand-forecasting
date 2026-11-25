"""Batch prediction with the registered champion model.

Purchasing runs this once a day (e.g. after the lab closes): it takes the
latest consumption history, builds features for the most recent day of every
ingredient and asks the ``@champion`` model for the next 7/15/30-day totals.
"""

from __future__ import annotations

import mlflow
import pandas as pd

from pharma_forecast.config import CHAMPION_ALIAS, REGISTERED_MODEL_NAME, Settings
from pharma_forecast.features import build_features, feature_columns


def latest_feature_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Feature row for the last observed date of each ingredient.

    Only the history up to that date is used, i.e. exactly what would be known
    in production at the end of the day.
    """
    feats = build_features(raw)
    last = feats.groupby("ingredient")["date"].transform("max")
    return feats[feats["date"] == last].dropna(subset=feature_columns()).reset_index(drop=True)


def predict_latest(
    raw: pd.DataFrame,
    settings: Settings,
    alias: str = CHAMPION_ALIAS,
    name: str = REGISTERED_MODEL_NAME,
) -> pd.DataFrame:
    """Forecast next-H-day consumption (grams) per ingredient.

    Args:
        raw: Long consumption history.
        settings: MLflow configuration.
        alias: Registry alias to load (``champion`` by default; pass
            ``challenger`` to shadow-test a candidate).
        name: Registered model name.

    Returns:
        DataFrame with ``ingredient``, ``origin_date``, ``model_version`` and
        ``forecast_{H}`` columns.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    uri = f"models:/{name}@{alias}"
    model = mlflow.pyfunc.load_model(uri)
    rows = latest_feature_rows(raw)
    preds = model.predict(rows[feature_columns() + ["scale"]])
    version = str(mlflow.MlflowClient().get_model_version_by_alias(name, alias).version)
    out = pd.concat(
        [rows[["ingredient", "date"]].rename(columns={"date": "origin_date"}), preds], axis=1
    )
    out.insert(2, "model_version", version)
    return out.round(1)
