"""
Preprocessing Pipeline Module
=============================

Leakage-safe preprocessing with fit/transform pattern ensuring
all fitting happens on training data only.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder as SKLabelEncoder

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingPipeline:
    """
    Leakage-safe preprocessing pipeline.

    Ensures all statistics are computed on training data only,
    then applied consistently to validation and test sets.

    Attributes:
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        spatial_features: List of spatial feature names
        scaling_method: Scaling method for numeric features
        target_column: Name of target column
    """

    numeric_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    spatial_features: List[str] = field(default_factory=list)
    scaling_method: str = "standard"  # "standard", "minmax", "robust", "none"
    target_column: str = "Classification_target"

    # Fitted components (populated during fit)
    _scaler: Optional[Any] = field(default=None, init=False)
    _label_encoder: Optional[Dict[str, int]] = field(default=None, init=False)
    _categorical_encoders: Dict[str, Dict[str, int]] = field(default_factory=dict, init=False)
    _numeric_stats: Dict[str, Dict[str, float]] = field(default_factory=dict, init=False)
    _is_fitted: bool = field(default=False, init=False)

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> 'PreprocessingPipeline':
        """
        Fit the preprocessing pipeline on training data.

        Args:
            df: Training DataFrame
            y: Target series (optional, for supervised encoding)

        Returns:
            self
        """
        logger.info("Fitting preprocessing pipeline")

        # Fit scaler on numeric features
        if self.scaling_method != "none" and self.numeric_features:
            self._fit_scaler(df)

        # Fit categorical encoders
        if self.categorical_features:
            self._fit_categorical_encoders(df)

        # Store numeric statistics for imputation fallback
        self._compute_numeric_stats(df)

        # Fit label encoder if target present
        if self.target_column in df.columns:
            self._fit_label_encoder(df[self.target_column])
        elif y is not None:
            self._fit_label_encoder(y)

        self._is_fitted = True
        logger.info("Preprocessing pipeline fitted successfully")
        return self

    def transform(
        self,
        df: pd.DataFrame,
        scale_features: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """
        Transform data using fitted pipeline.

        Args:
            df: DataFrame to transform
            scale_features: Whether to apply scaling (False for tree models)

        Returns:
            Tuple of (X array, y array if target present, feature names)
        """
        if not self._is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        df = df.copy()
        feature_names = []

        # Process numeric features
        X_numeric = None
        if self.numeric_features:
            available_numeric = [c for c in self.numeric_features if c in df.columns]
            if available_numeric:
                X_numeric = df[available_numeric].values.astype(np.float32)

                # Handle missing values using fitted statistics
                for i, col in enumerate(available_numeric):
                    mask = np.isnan(X_numeric[:, i])
                    if mask.any() and col in self._numeric_stats:
                        X_numeric[mask, i] = self._numeric_stats[col]['median']

                # Apply scaling if requested
                if scale_features and self._scaler is not None:
                    # Only scale columns that were in training
                    cols_to_scale = [i for i, c in enumerate(available_numeric)
                                    if c in self._scaler_columns]
                    if cols_to_scale:
                        X_numeric[:, cols_to_scale] = self._scaler.transform(
                            X_numeric[:, cols_to_scale]
                        )

                feature_names.extend(available_numeric)

        # Process categorical features (integer encoding)
        X_categorical = None
        if self.categorical_features:
            available_cat = [c for c in self.categorical_features if c in df.columns]
            if available_cat:
                X_categorical = np.zeros((len(df), len(available_cat)), dtype=np.int32)
                for i, col in enumerate(available_cat):
                    if col in self._categorical_encoders:
                        encoder = self._categorical_encoders[col]
                        # Map values, using -1 for unknown
                        X_categorical[:, i] = df[col].map(
                            lambda x: encoder.get(str(x).upper().strip(), -1)
                        ).values
                feature_names.extend(available_cat)

        # Process spatial features
        X_spatial = None
        if self.spatial_features:
            available_spatial = [c for c in self.spatial_features if c in df.columns]
            if available_spatial:
                X_spatial = df[available_spatial].values.astype(np.float32)
                feature_names.extend(available_spatial)

        # Combine all features
        arrays_to_concat = [a for a in [X_numeric, X_categorical, X_spatial]
                          if a is not None]
        if arrays_to_concat:
            # Convert all to float for concatenation
            arrays_float = [a.astype(np.float32) for a in arrays_to_concat]
            X = np.concatenate(arrays_float, axis=1)
        else:
            X = np.array([]).reshape(len(df), 0)

        # Process target
        y = None
        if self.target_column in df.columns and self._label_encoder is not None:
            y = df[self.target_column].map(self._label_encoder).values
            # Handle unmapped labels
            y = np.where(pd.isna(y), -1, y).astype(np.int32)

        return X, y, feature_names

    def fit_transform(
        self,
        df: pd.DataFrame,
        y: Optional[pd.Series] = None,
        scale_features: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """Fit and transform in one step."""
        self.fit(df, y)
        return self.transform(df, scale_features)

    def _fit_scaler(self, df: pd.DataFrame):
        """Fit scaler on numeric features."""
        available = [c for c in self.numeric_features if c in df.columns]
        if not available:
            return

        self._scaler_columns = available
        X = df[available].values.astype(np.float32)

        # Handle NaN by filling with median temporarily
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            if mask.any():
                X[mask, i] = np.nanmedian(X[:, i])

        if self.scaling_method == "standard":
            self._scaler = StandardScaler()
        elif self.scaling_method == "minmax":
            self._scaler = MinMaxScaler()
        elif self.scaling_method == "robust":
            self._scaler = RobustScaler()

        if self._scaler is not None:
            self._scaler.fit(X)
            logger.debug(f"Fitted {self.scaling_method} scaler on {len(available)} features")

    def _fit_categorical_encoders(self, df: pd.DataFrame):
        """Fit categorical encoders (label encoding)."""
        for col in self.categorical_features:
            if col not in df.columns:
                continue

            # Get unique values and create encoding
            values = df[col].astype(str).str.upper().str.strip().unique()
            self._categorical_encoders[col] = {v: i for i, v in enumerate(sorted(values))}
            logger.debug(f"Fitted encoder for {col}: {len(values)} unique values")

    def _compute_numeric_stats(self, df: pd.DataFrame):
        """Compute numeric statistics for imputation."""
        for col in self.numeric_features:
            if col not in df.columns:
                continue

            values = df[col].dropna()
            if len(values) > 0:
                self._numeric_stats[col] = {
                    'mean': float(values.mean()),
                    'median': float(values.median()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max())
                }

    def _fit_label_encoder(self, y: pd.Series):
        """Fit label encoder for target with correct ordering for tier names or C#S# labels."""
        import re
        unique_labels = y.dropna().unique()

        # Tier order for the 3-tier semantic system (T1 < T2 < T3)
        _tier_order = {'T1_Safe': 0, 'T2_Marginal': 1, 'T3_Restricted': 2, 'T4_Unsafe': 3}

        def sort_key(label):
            s = str(label)
            if s in _tier_order:
                return (_tier_order[s], 0)
            match = re.match(r'C(\d)S(\d)', s)
            if match:
                return (int(match.group(1)), int(match.group(2)))
            return (99, 99)

        sorted_labels = sorted(unique_labels, key=sort_key)
        self._label_encoder = {str(label): i for i, label in enumerate(sorted_labels)}
        logger.info(f"Label encoder: {self._label_encoder}")

    def get_label_encoder(self) -> Dict[str, int]:
        """Get the fitted label encoder."""
        return self._label_encoder.copy() if self._label_encoder else {}

    def get_inverse_label_encoder(self) -> Dict[int, str]:
        """Get inverse label encoder (int -> label)."""
        if self._label_encoder is None:
            return {}
        return {v: k for k, v in self._label_encoder.items()}

    def get_n_classes(self) -> int:
        """Get number of classes."""
        return len(self._label_encoder) if self._label_encoder else 0

    def get_categorical_cardinalities(self) -> Dict[str, int]:
        """Get cardinality of each categorical feature."""
        return {col: len(enc) for col, enc in self._categorical_encoders.items()}

    def get_feature_info(self) -> Dict[str, Any]:
        """Get comprehensive feature information."""
        return {
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'spatial_features': self.spatial_features,
            'n_numeric': len(self.numeric_features),
            'n_categorical': len(self.categorical_features),
            'n_spatial': len(self.spatial_features),
            'categorical_cardinalities': self.get_categorical_cardinalities(),
            'n_classes': self.get_n_classes(),
            'label_mapping': self.get_label_encoder()
        }


def get_feature_dataframe(
    X: np.ndarray,
    feature_names: List[str],
    y: Optional[np.ndarray] = None,
    y_name: str = 'target'
) -> pd.DataFrame:
    """Convert numpy arrays back to DataFrame."""
    df = pd.DataFrame(X, columns=feature_names)
    if y is not None:
        df[y_name] = y
    return df
