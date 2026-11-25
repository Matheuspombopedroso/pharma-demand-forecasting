"""Raw-material demand forecasting for compounding pharmacies (demo).

Pipeline: synthetic data -> leakage-safe features -> walk-forward validation
-> Ray Tune search over XGBoost -> MLflow tracking and Model Registry
(champion / challenger / previous aliases) -> batch prediction.
"""

__version__ = "0.1.0"
