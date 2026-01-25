"""
Leakage-Safe Scalers Module
===========================

Wrapper classes ensuring scalers are fitted only on training data.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

logger = logging.getLogger(__name__)


@dataclass
class LeakageSafeScaler:
    """
    Wrapper for sklearn scalers with leakage protection.

    Ensures statistics are computed only on training data and
    provides explicit tracking of fit/transform status.
    """

    method: str = "standard"  # "standard", "minmax", "robust", "none"
    feature_names: List[str] = field(default_factory=list)

    _scaler: Optional[Any] = field(default=None, init=False)
    _is_fitted: bool = field(default=False, init=False)
    _fit_stats: Dict[str, Dict[str, float]] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if self.method == "standard":
            self._scaler = StandardScaler()
        elif self.method == "minmax":
            self._scaler = MinMaxScaler()
        elif self.method == "robust":
            self._scaler = RobustScaler()
        elif self.method == "none":
            self._scaler = None
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'LeakageSafeScaler':
        """
        Fit scaler on training data.

        Args:
            X: Training data array
            feature_names: Optional list of feature names

        Returns:
            self
        """
        if self._scaler is None:
            self._is_fitted = True
            return self

        if feature_names:
            self.feature_names = feature_names

        # Handle NaN by filling with column median temporarily
        X_filled = X.copy()
        for i in range(X_filled.shape[1]):
            mask = np.isnan(X_filled[:, i])
            if mask.any():
                X_filled[mask, i] = np.nanmedian(X_filled[:, i])

        self._scaler.fit(X_filled)
        self._is_fitted = True

        # Store fit statistics for reference
        self._compute_fit_stats(X)

        logger.info(f"Fitted {self.method} scaler on {X.shape[1]} features")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted scaler.

        Args:
            X: Data array to transform

        Returns:
            Scaled data array
        """
        if not self._is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")

        if self._scaler is None:
            return X

        # Handle NaN by filling with fitted median
        X_filled = X.copy()
        for i in range(X_filled.shape[1]):
            mask = np.isnan(X_filled[:, i])
            if mask.any():
                if i < len(self.feature_names):
                    col_name = self.feature_names[i]
                    if col_name in self._fit_stats:
                        X_filled[mask, i] = self._fit_stats[col_name]['median']
                else:
                    X_filled[mask, i] = 0  # Fallback

        return self._scaler.transform(X_filled)

    def fit_transform(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, feature_names)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data."""
        if self._scaler is None:
            return X
        return self._scaler.inverse_transform(X)

    def _compute_fit_stats(self, X: np.ndarray):
        """Compute and store fit statistics."""
        for i in range(X.shape[1]):
            col_data = X[:, i]
            col_data = col_data[~np.isnan(col_data)]

            if len(col_data) > 0:
                name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
                self._fit_stats[name] = {
                    'mean': float(np.mean(col_data)),
                    'std': float(np.std(col_data)),
                    'median': float(np.median(col_data)),
                    'min': float(np.min(col_data)),
                    'max': float(np.max(col_data))
                }

    def get_fit_stats(self) -> Dict[str, Dict[str, float]]:
        """Get the fit statistics."""
        return self._fit_stats.copy()

    @property
    def is_fitted(self) -> bool:
        """Check if scaler is fitted."""
        return self._is_fitted
