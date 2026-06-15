"""Tree-based model implementations."""

from .catboost_model import CatBoostForecaster
from .lightgbm_model import LightGBMForecaster

try:
    from .xgboost_model import XGBoostForecaster
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    XGBoostForecaster = None

__all__ = ["CatBoostForecaster", "LightGBMForecaster", "XGBoostForecaster"]
