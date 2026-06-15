"""Model implementations for groundwater quality forecasting."""

from .base import BaseForecaster, ForecasterConfig
from .trees.catboost_model import CatBoostForecaster
from .trees.lightgbm_model import LightGBMForecaster

try:
    from .trees.xgboost_model import XGBoostForecaster
except ImportError:
    XGBoostForecaster = None

try:
    from .advanced.ft_transformer import FTTransformerForecaster
except ImportError:
    FTTransformerForecaster = None

try:
    from .advanced.gnn_model import SpatialGNNForecaster
except ImportError:
    SpatialGNNForecaster = None

try:
    from .advanced.coral_model import CORALForecaster
except ImportError:
    CORALForecaster = None

__all__ = [
    "BaseForecaster",
    "ForecasterConfig",
    "CatBoostForecaster",
    "LightGBMForecaster",
    "XGBoostForecaster",
    "FTTransformerForecaster",
    "SpatialGNNForecaster",
    "CORALForecaster",
]
