"""Advanced model implementations (FT-Transformer, Spatial GNN, CORAL)."""

try:
    from .ft_transformer import FTTransformerForecaster
    FT_AVAILABLE = True
except ImportError:
    FT_AVAILABLE = False
    FTTransformerForecaster = None

try:
    from .gnn_model import SpatialGNNForecaster
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    SpatialGNNForecaster = None

try:
    from .coral_model import CORALForecaster
    CORAL_AVAILABLE = True
except ImportError:
    CORAL_AVAILABLE = False
    CORALForecaster = None

__all__ = [
    "FTTransformerForecaster",
    "SpatialGNNForecaster",
    "CORALForecaster",
]
