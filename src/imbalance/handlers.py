"""
Imbalance Handling Module
=========================

Strategies for handling class imbalance in groundwater classification.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_class_weights(
    y: np.ndarray,
    method: str = "balanced"
) -> Dict[int, float]:
    """
    Compute class weights for imbalanced classification.

    Args:
        y: Target array
        method: "balanced", "sqrt", or "none"

    Returns:
        Dictionary mapping class index to weight
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(unique_classes)

    if method == "balanced":
        # sklearn-style balanced weights
        weights = n_samples / (n_classes * counts)
    elif method == "sqrt":
        # Square root dampening
        weights = np.sqrt(n_samples / (n_classes * counts))
    else:
        weights = np.ones(n_classes)

    return {int(cls): float(w) for cls, w in zip(unique_classes, weights)}


@dataclass
class ImbalanceHandler:
    """
    Handle class imbalance through various strategies.

    Supports:
    - Class weights (compatible with all models)
    - SMOTE resampling (tabular data only)
    """

    strategy: str = "class_weights"  # "class_weights", "smote", "none"
    weight_method: str = "balanced"  # "balanced", "sqrt"
    smote_k_neighbors: int = 5

    _class_weights: Optional[Dict[int, float]] = field(default=None, init=False)

    def fit(self, y: np.ndarray) -> 'ImbalanceHandler':
        """
        Fit imbalance handler on target.

        Args:
            y: Target array

        Returns:
            self
        """
        if self.strategy == "class_weights":
            self._class_weights = compute_class_weights(y, self.weight_method)
            logger.info(f"Computed class weights: {self._class_weights}")

        return self

    def get_class_weights(self) -> Optional[Dict[int, float]]:
        """Get computed class weights."""
        return self._class_weights

    def resample(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample training data if using SMOTE.

        Args:
            X: Features
            y: Targets

        Returns:
            Resampled (X, y)
        """
        if self.strategy != "smote":
            return X, y

        try:
            from imblearn.over_sampling import SMOTE

            smote = SMOTE(
                k_neighbors=self.smote_k_neighbors,
                random_state=42
            )
            X_resampled, y_resampled = smote.fit_resample(X, y)

            logger.info(
                f"SMOTE resampling: {len(y)} -> {len(y_resampled)} samples"
            )
            return X_resampled, y_resampled

        except ImportError:
            logger.warning("imbalanced-learn not installed. Skipping SMOTE.")
            return X, y

    def get_sample_weights(self, y: np.ndarray) -> Optional[np.ndarray]:
        """
        Get per-sample weights for training.

        Args:
            y: Target array

        Returns:
            Sample weights array
        """
        if self._class_weights is None:
            return None

        return np.array([self._class_weights.get(int(label), 1.0) for label in y])
