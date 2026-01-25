"""
Base Model Module
=================

Abstract base class defining the interface for all forecasting models.
Ensures consistent API across tree-based and deep learning models.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ForecasterConfig:
    """Configuration container for forecaster hyperparameters."""

    # Model-specific parameters
    params: Dict[str, Any] = field(default_factory=dict)

    # Feature selection mask (binary)
    feature_mask: Optional[np.ndarray] = None

    # Training configuration
    n_classes: int = 0
    class_weights: Optional[Dict[int, float]] = None

    # Random seed
    random_state: int = 42

    def get_param(self, name: str, default: Any = None) -> Any:
        """Get a parameter with default value."""
        return self.params.get(name, default)

    def set_param(self, name: str, value: Any):
        """Set a parameter."""
        self.params[name] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'params': self.params.copy(),
            'feature_mask': self.feature_mask.tolist() if self.feature_mask is not None else None,
            'n_classes': self.n_classes,
            'random_state': self.random_state
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ForecasterConfig':
        """Create from dictionary."""
        config = cls(
            params=d.get('params', {}),
            n_classes=d.get('n_classes', 0),
            random_state=d.get('random_state', 42)
        )
        if d.get('feature_mask') is not None:
            config.feature_mask = np.array(d['feature_mask'])
        return config


class BaseForecaster(ABC):
    """
    Abstract base class for groundwater quality forecasters.

    All model implementations must inherit from this class and
    implement the abstract methods to ensure consistent API.
    """

    def __init__(self, config: Optional[ForecasterConfig] = None):
        """
        Initialize forecaster.

        Args:
            config: Forecaster configuration
        """
        self.config = config or ForecasterConfig()
        self._is_fitted = False
        self._model = None
        self._n_features = None
        self._n_classes = None
        self._feature_names: Optional[List[str]] = None

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted

    @property
    def n_features(self) -> Optional[int]:
        """Number of features used in training."""
        return self._n_features

    @property
    def n_classes(self) -> Optional[int]:
        """Number of classes."""
        return self._n_classes

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None
    ) -> 'BaseForecaster':
        """
        Fit the model.

        Args:
            X: Training features
            y: Training targets
            X_val: Optional validation features (for early stopping)
            y_val: Optional validation targets
            feature_names: Optional feature names
            categorical_features: Optional indices of categorical features

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features

        Returns:
            Predicted class labels (integers)
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features

        Returns:
            Predicted probabilities (n_samples, n_classes)
        """
        pass

    def apply_feature_mask(self, X: np.ndarray) -> np.ndarray:
        """Apply feature selection mask if configured."""
        if self.config.feature_mask is not None:
            mask = self.config.feature_mask.astype(bool)
            return X[:, mask]
        return X

    def get_selected_features(self) -> Optional[np.ndarray]:
        """Get indices of selected features."""
        if self.config.feature_mask is not None:
            return np.where(self.config.feature_mask.astype(bool))[0]
        return None

    def get_n_selected_features(self) -> int:
        """Get number of selected features."""
        if self.config.feature_mask is not None:
            return int(np.sum(self.config.feature_mask))
        return self._n_features or 0

    @abstractmethod
    def get_model_complexity(self) -> Dict[str, Any]:
        """
        Get model complexity metrics.

        Returns:
            Dictionary with complexity metrics (e.g., n_params, depth)
        """
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]):
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: Union[str, Path]) -> 'BaseForecaster':
        """Load model from disk."""
        pass

    @classmethod
    @abstractmethod
    def get_model_name(cls) -> str:
        """Get the model name."""
        pass

    @classmethod
    @abstractmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Get default hyperparameters."""
        pass

    @classmethod
    @abstractmethod
    def get_param_space(cls) -> Dict[str, Tuple[Any, Any]]:
        """
        Get hyperparameter search space.

        Returns:
            Dictionary mapping param name to (min, max) bounds
        """
        pass


def compute_class_weights(
    y: np.ndarray,
    method: str = "balanced"
) -> Dict[int, float]:
    """
    Compute class weights for imbalanced classification.

    Args:
        y: Target array
        method: "balanced" or "sqrt"

    Returns:
        Dictionary mapping class index to weight
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(unique_classes)

    if method == "balanced":
        weights = n_samples / (n_classes * counts)
    elif method == "sqrt":
        weights = np.sqrt(n_samples / (n_classes * counts))
    else:
        weights = np.ones(n_classes)

    return {int(cls): float(w) for cls, w in zip(unique_classes, weights)}
