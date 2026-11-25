"""Multi-horizon XGBoost forecaster and its MLflow ``pyfunc`` wrapper.

Design
------
One XGBoost regressor per horizon (7, 15, 30 days), all sharing the same
hyper-parameters and the same feature table. A separate model per horizon
(the "direct" strategy) avoids feeding predictions back as inputs, which
would compound errors over 30 steps, at the cost of training three models.

Each regressor predicts the scale-free ratio described in ``features.py``;
:meth:`MultiHorizonForecaster.predict_grams` converts back to grams.

Serving contract
----------------
The MLflow model takes a DataFrame that already contains the feature columns
(plus ``scale``) and returns ``forecast_7``, ``forecast_15``, ``forecast_30`` in
grams. Feature engineering stays in this package (``predict.py`` calls it)
rather than inside the pyfunc. That keeps the registered artifact small and
the feature code in one place, with the trade-off that callers must install
this package to build inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd
import xgboost as xgb

from pharma_forecast.config import HORIZONS
from pharma_forecast.features import feature_columns, unscale_prediction

# Defaults used by `train` when no tuned parameters are supplied. Chosen to be
# reasonable, not optimal; `tune` searches around them.
DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5.0,
    "reg_lambda": 1.0,
}


def make_regressor(params: dict[str, Any], seed: int, n_jobs: int = 1) -> xgb.XGBRegressor:
    """Create one XGBoost regressor.

    ``n_jobs`` defaults to 1 on purpose: Ray Tune runs several trials in
    parallel, one CPU each, so letting XGBoost also spawn threads would
    oversubscribe the cores and make the parallel speed-up meaningless.
    ``tree_method="hist"`` is the fast histogram algorithm on CPU.
    """
    return xgb.XGBRegressor(
        objective="reg:absoluteerror",  # L1 loss matches the MAE/WAPE metrics.
        tree_method="hist",
        random_state=seed,
        n_jobs=n_jobs,
        **params,
    )


class MultiHorizonForecaster:
    """A bundle of per-horizon regressors with a grams-level ``predict``.

    Args:
        params: XGBoost hyper-parameters shared by all horizons.
        horizons: Horizons (days) to train one model for.
        seed: Random seed for XGBoost's row/column subsampling.
        n_jobs: Threads per XGBoost model.
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        horizons: tuple[int, ...] = HORIZONS,
        seed: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.params = dict(params or DEFAULT_PARAMS)
        self.horizons = tuple(horizons)
        self.seed = seed
        self.n_jobs = n_jobs
        self.models: dict[int, xgb.XGBRegressor] = {}

    def fit(self, table: pd.DataFrame) -> MultiHorizonForecaster:
        """Fit every horizon on rows whose target for that horizon is known."""
        for h in self.horizons:
            self.fit_horizon(table, h)
        return self

    def fit_horizon(self, table: pd.DataFrame, horizon: int) -> xgb.XGBRegressor:
        """Fit and store a single horizon (used by walk-forward evaluation)."""
        rows = table[f"target_ratio_{horizon}"].notna()
        model = make_regressor(self.params, self.seed, self.n_jobs)
        model.fit(table.loc[rows, feature_columns()], table.loc[rows, f"target_ratio_{horizon}"])
        self.models[horizon] = model
        return model

    def predict_horizon(self, features: pd.DataFrame, horizon: int) -> np.ndarray:
        """Predict grams for one horizon from a feature DataFrame."""
        ratio = self.models[horizon].predict(features[feature_columns()])
        # Demand cannot be negative; clip in ratio space before unscaling.
        return unscale_prediction(np.clip(ratio, 0, None), features["scale"], horizon)

    def predict_grams(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict grams for every horizon; columns ``forecast_{H}``."""
        return pd.DataFrame(
            {f"forecast_{h}": self.predict_horizon(features, h) for h in self.horizons},
            index=features.index,
        )

    def save(self, directory: Path) -> Path:
        """Save each booster as JSON plus a small metadata file."""
        directory.mkdir(parents=True, exist_ok=True)
        for h, model in self.models.items():
            model.save_model(directory / f"xgb_h{h}.json")
        meta = {"params": self.params, "horizons": list(self.horizons), "seed": self.seed}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))
        return directory

    @classmethod
    def load(cls, directory: Path) -> MultiHorizonForecaster:
        """Load a forecaster written by :meth:`save`."""
        meta = json.loads((directory / "meta.json").read_text())
        obj = cls(meta["params"], tuple(meta["horizons"]), meta["seed"])
        for h in obj.horizons:
            model = xgb.XGBRegressor()
            model.load_model(directory / f"xgb_h{h}.json")
            obj.models[h] = model
        return obj


class ForecasterPyfunc(mlflow.pyfunc.PythonModel):
    """MLflow wrapper so the registry can serve the forecaster generically.

    The boosters are stored as plain artifacts (``model_dir``) instead of being
    pickled, so the saved model does not depend on pickle compatibility
    between XGBoost versions.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load the boosters from the artifact directory MLflow provides."""
        self.forecaster = MultiHorizonForecaster.load(Path(context.artifacts["model_dir"]))

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Return ``forecast_{H}`` columns (grams) for each input row."""
        return self.forecaster.predict_grams(model_input)
