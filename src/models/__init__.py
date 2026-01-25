"""Model implementations for groundwater quality forecasting."""

from .base import BaseForecaster, ForecasterConfig
from .trees.catboost_model import CatBoostForecaster
from .trees.lightgbm_model import LightGBMForecaster

__all__ = [
    "BaseForecaster",
    "ForecasterConfig",
    "CatBoostForecaster",
    "LightGBMForecaster",
]
