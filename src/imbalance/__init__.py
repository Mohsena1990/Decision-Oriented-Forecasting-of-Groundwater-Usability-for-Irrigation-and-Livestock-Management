"""Class imbalance handling module."""

from .handlers import ImbalanceHandler, compute_class_weights
from .enhanced_handlers import (
    EnhancedImbalanceHandler,
    compute_enhanced_class_weights,
    compute_sample_weights,
    optimize_threshold_for_recall,
    apply_threshold_adjustment,
)

# Import PyTorch losses if available
try:
    from .enhanced_handlers import FocalLoss, ClassBalancedLoss, create_loss_function
    TORCH_LOSSES_AVAILABLE = True
except ImportError:
    TORCH_LOSSES_AVAILABLE = False
    FocalLoss = None
    ClassBalancedLoss = None
    create_loss_function = None

__all__ = [
    "ImbalanceHandler",
    "compute_class_weights",
    "EnhancedImbalanceHandler",
    "compute_enhanced_class_weights",
    "compute_sample_weights",
    "optimize_threshold_for_recall",
    "apply_threshold_adjustment",
    "FocalLoss",
    "ClassBalancedLoss",
    "create_loss_function",
    "TORCH_LOSSES_AVAILABLE",
]
