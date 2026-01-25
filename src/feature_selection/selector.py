"""
Feature Selector Module
=======================

Wrapper-based feature selection integrated with optimization.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureSelector:
    """
    Feature selection component for optimization.

    In wrapper mode, feature selection is part of the PSO-GWO search.
    Binary mask is encoded in the position vector.
    """

    n_features: int = 0
    mode: str = "wrapper"  # "wrapper" or "filter"
    min_features: int = 3
    parsimony_weight: float = 0.01

    _selected_mask: Optional[np.ndarray] = field(default=None, init=False)

    def initialize_mask(self, n_features: int) -> np.ndarray:
        """
        Initialize feature mask for optimization.

        Args:
            n_features: Number of features

        Returns:
            Initial binary mask
        """
        self.n_features = n_features

        # Start with all features selected
        return np.ones(n_features)

    def decode_mask(
        self,
        position: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Decode continuous position to binary mask.

        Args:
            position: Continuous values from optimizer
            threshold: Selection threshold

        Returns:
            Binary mask
        """
        mask = (position > threshold).astype(int)

        # Ensure minimum features
        if np.sum(mask) < self.min_features:
            # Select top features by position value
            top_indices = np.argsort(position)[-self.min_features:]
            mask = np.zeros_like(mask)
            mask[top_indices] = 1

        return mask

    def compute_parsimony_penalty(self, mask: np.ndarray) -> float:
        """
        Compute parsimony penalty for number of features.

        Args:
            mask: Binary feature mask

        Returns:
            Penalty value (to minimize)
        """
        n_selected = np.sum(mask)
        return self.parsimony_weight * (n_selected / self.n_features)

    def apply_mask(
        self,
        X: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Apply feature mask to data."""
        return X[:, mask.astype(bool)]

    def get_selected_indices(self, mask: np.ndarray) -> np.ndarray:
        """Get indices of selected features."""
        return np.where(mask.astype(bool))[0]

    def get_selected_names(
        self,
        mask: np.ndarray,
        feature_names: List[str]
    ) -> List[str]:
        """Get names of selected features."""
        indices = self.get_selected_indices(mask)
        return [feature_names[i] for i in indices if i < len(feature_names)]
