"""
Enhanced Preprocessing Pipeline
===============================

Professional data preprocessing addressing:
1. Severe class imbalance (234:1 ratio)
2. Covariate shift between train/test
3. Multicollinearity (EC-TDS perfect correlation)
4. Heavy-tailed distributions with outliers
5. Non-discriminative features
6. Missing value handling

Based on data distribution analysis results.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler, PowerTransformer, LabelEncoder
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE CONFIGURATION (Based on Analysis)
# =============================================================================

# Features to REMOVE (identified in analysis)
FEATURES_TO_REMOVE = [
    'gwl',   # Massive covariate shift (Cohen's d = -0.978), non-discriminative (F=0.75)
    'TDS',   # Perfectly correlated with EC (r=1.000) - redundant
]

# Highly skewed features needing transformation (skewness > 2)
HIGHLY_SKEWED_FEATURES = ['EC', 'Cl', 'F', 'NO3', 'SO4', 'Na', 'K', 'Ca', 'Mg', 'SAR', 'CO3']

# Features with significant outliers (>5%)
HIGH_OUTLIER_FEATURES = ['CO3', 'SO4', 'K', 'NO3', 'SAR', 'Na']

# Most discriminative features (ANOVA F > 100)
TOP_DISCRIMINATIVE_FEATURES = ['EC', 'Na', 'SAR', 'Cl', 'Mg', 'HCO3', 'Ca', 'NO3']

# High-risk classes for special weighting
HIGH_RISK_CLASSES = ['C4S1', 'C4S2', 'C4S3', 'C4S4', 'C3S3', 'C3S4']


@dataclass
class EnhancedPreprocessingConfig:
    """Configuration for enhanced preprocessing."""

    # Feature handling
    remove_redundant_features: bool = True
    remove_shifted_features: bool = True

    # Transformation
    apply_power_transform: bool = True
    power_transform_method: str = 'yeo-johnson'  # 'yeo-johnson' or 'box-cox'

    # Scaling
    scaling_method: str = 'robust'  # 'robust', 'standard', 'minmax'

    # Outlier handling
    outlier_method: str = 'winsorize'  # 'winsorize', 'clip', 'none'
    winsorize_limits: Tuple[float, float] = (0.02, 0.98)  # Stronger winsorization

    # Feature engineering
    add_current_class_feature: bool = True  # Use current year's class as feature
    add_interaction_features: bool = True   # Add key interactions

    # Class weighting
    class_weight_strategy: str = 'custom_high_risk'  # 'balanced', 'sqrt', 'custom_high_risk'
    high_risk_weight_multiplier: float = 3.0  # Extra weight for high-risk classes


@dataclass
class EnhancedPreprocessingPipeline:
    """
    Enhanced preprocessing pipeline with professional techniques.

    Addresses key data issues:
    - Removes redundant/shifted features
    - Applies robust scaling for outliers
    - Power transforms skewed features
    - Computes custom class weights for severe imbalance
    """

    config: EnhancedPreprocessingConfig = field(default_factory=EnhancedPreprocessingConfig)

    # Feature definitions
    numeric_features: List[str] = field(default_factory=lambda: [
        'pH', 'EC', 'CO3', 'HCO3', 'Cl', 'F', 'NO3', 'SO4', 'Na', 'K', 'Ca', 'Mg', 'TH', 'SAR', 'RSC'
    ])
    categorical_features: List[str] = field(default_factory=lambda: ['district', 'mandal', 'village'])
    spatial_features: List[str] = field(default_factory=lambda: ['lat_gis', 'long_gis'])
    target_column: str = 'Classification_target'

    # Fitted components
    _scaler: Optional[Any] = field(default=None, init=False)
    _power_transformer: Optional[PowerTransformer] = field(default=None, init=False)
    _label_encoder: Optional[Dict[str, int]] = field(default=None, init=False)
    _categorical_encoders: Dict[str, Dict[str, int]] = field(default_factory=dict, init=False)
    _numeric_stats: Dict[str, Dict[str, float]] = field(default_factory=dict, init=False)
    _winsorize_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict, init=False)
    _final_feature_names: List[str] = field(default_factory=list, init=False)
    _class_weights: Optional[Dict[int, float]] = field(default=None, init=False)
    _is_fitted: bool = field(default=False, init=False)

    def __post_init__(self):
        """Apply feature removal based on config."""
        if self.config.remove_redundant_features:
            # Remove TDS (redundant with EC)
            if 'TDS' in self.numeric_features:
                self.numeric_features = [f for f in self.numeric_features if f != 'TDS']
                logger.info("Removed TDS (redundant with EC, r=1.000)")

        if self.config.remove_shifted_features:
            # Remove gwl (massive covariate shift)
            if 'gwl' in self.numeric_features:
                self.numeric_features = [f for f in self.numeric_features if f != 'gwl']
                logger.info("Removed gwl (covariate shift Cohen's d=-0.978)")

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> 'EnhancedPreprocessingPipeline':
        """
        Fit the preprocessing pipeline on training data.

        Args:
            df: Training DataFrame
            y: Optional target series

        Returns:
            self
        """
        logger.info("Fitting enhanced preprocessing pipeline")

        # 1. Compute numeric statistics and bounds
        self._fit_numeric_stats(df)

        # 2. Fit power transformer for skewed features
        if self.config.apply_power_transform:
            self._fit_power_transformer(df)

        # 3. Fit scaler
        self._fit_scaler(df)

        # 4. Fit categorical encoders
        self._fit_categorical_encoders(df)

        # 5. Fit label encoder
        if self.target_column in df.columns:
            self._fit_label_encoder(df[self.target_column])
        elif y is not None:
            self._fit_label_encoder(y)

        # 6. Compute class weights
        if self.target_column in df.columns:
            self._compute_class_weights(df[self.target_column])
        elif y is not None:
            self._compute_class_weights(y)

        self._is_fitted = True
        logger.info("Enhanced preprocessing pipeline fitted successfully")
        return self

    def transform(
        self,
        df: pd.DataFrame,
        scale_features: bool = True,
        include_interactions: bool = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """
        Transform data using fitted pipeline.

        Args:
            df: DataFrame to transform
            scale_features: Whether to apply scaling (False for tree models)
            include_interactions: Whether to include interaction features

        Returns:
            Tuple of (X array, y array if target present, feature names)
        """
        if not self._is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        if include_interactions is None:
            include_interactions = self.config.add_interaction_features

        df = df.copy()
        feature_arrays = []
        feature_names = []

        # 1. Process numeric features
        X_numeric, num_names = self._transform_numeric(df, scale_features)
        if X_numeric is not None:
            feature_arrays.append(X_numeric)
            feature_names.extend(num_names)

        # 2. Process categorical features
        X_cat, cat_names = self._transform_categorical(df)
        if X_cat is not None:
            feature_arrays.append(X_cat.astype(np.float32))
            feature_names.extend(cat_names)

        # 3. Process spatial features
        X_spatial, spatial_names = self._transform_spatial(df)
        if X_spatial is not None:
            feature_arrays.append(X_spatial)
            feature_names.extend(spatial_names)

        # 4. Add current classification as feature (if available and configured)
        if self.config.add_current_class_feature and 'Classification' in df.columns:
            X_current, current_names = self._add_current_class_feature(df)
            if X_current is not None:
                feature_arrays.append(X_current)
                feature_names.extend(current_names)

        # 5. Add interaction features
        if include_interactions and X_numeric is not None:
            X_interact, interact_names = self._create_interaction_features(X_numeric, num_names)
            if X_interact is not None:
                feature_arrays.append(X_interact)
                feature_names.extend(interact_names)

        # Combine all features
        if feature_arrays:
            X = np.concatenate(feature_arrays, axis=1)
        else:
            X = np.array([]).reshape(len(df), 0)

        self._final_feature_names = feature_names

        # Process target
        y = None
        if self.target_column in df.columns and self._label_encoder is not None:
            y = df[self.target_column].map(self._label_encoder).values
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

    def _fit_numeric_stats(self, df: pd.DataFrame):
        """Compute statistics and winsorization bounds."""
        for col in self.numeric_features:
            if col not in df.columns:
                continue

            values = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(values) == 0:
                continue

            self._numeric_stats[col] = {
                'mean': float(values.mean()),
                'median': float(values.median()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'q01': float(values.quantile(0.01)),
                'q99': float(values.quantile(0.99)),
                'skewness': float(stats.skew(values))
            }

            # Winsorization bounds
            lower = values.quantile(self.config.winsorize_limits[0])
            upper = values.quantile(self.config.winsorize_limits[1])
            self._winsorize_bounds[col] = (float(lower), float(upper))

        logger.info(f"Computed statistics for {len(self._numeric_stats)} numeric features")

    def _fit_power_transformer(self, df: pd.DataFrame):
        """Fit power transformer for skewed features."""
        skewed_cols = [c for c in HIGHLY_SKEWED_FEATURES
                      if c in self.numeric_features and c in df.columns]

        if not skewed_cols:
            return

        # Prepare data
        X = df[skewed_cols].copy()
        X = X.apply(pd.to_numeric, errors='coerce')

        # Handle missing and apply winsorization first
        for col in skewed_cols:
            if col in self._numeric_stats:
                X[col] = X[col].fillna(self._numeric_stats[col]['median'])
                if col in self._winsorize_bounds:
                    lower, upper = self._winsorize_bounds[col]
                    X[col] = X[col].clip(lower=lower, upper=upper)

        # Shift data to be positive for Yeo-Johnson (handles zeros better)
        X_shifted = X + 1  # Small shift to avoid log(0) issues

        self._power_transformer = PowerTransformer(
            method=self.config.power_transform_method,
            standardize=False  # We'll use RobustScaler separately
        )
        self._power_transformer.fit(X_shifted)
        self._power_transform_cols = skewed_cols

        logger.info(f"Fitted power transformer for {len(skewed_cols)} skewed features")

    def _fit_scaler(self, df: pd.DataFrame):
        """Fit scaler on numeric features."""
        available = [c for c in self.numeric_features if c in df.columns]
        if not available:
            return

        # Prepare data
        X = df[available].copy()
        X = X.apply(pd.to_numeric, errors='coerce')

        # Handle missing values
        for col in available:
            if col in self._numeric_stats:
                X[col] = X[col].fillna(self._numeric_stats[col]['median'])

        # Apply winsorization
        if self.config.outlier_method == 'winsorize':
            for col in available:
                if col in self._winsorize_bounds:
                    lower, upper = self._winsorize_bounds[col]
                    X[col] = X[col].clip(lower=lower, upper=upper)

        # Apply power transform if fitted
        if self._power_transformer is not None and hasattr(self, '_power_transform_cols'):
            transform_cols = [c for c in self._power_transform_cols if c in available]
            if transform_cols:
                col_indices = [available.index(c) for c in transform_cols]
                X_transform = X[transform_cols].values + 1
                X_transformed = self._power_transformer.transform(X_transform)
                for i, col in enumerate(transform_cols):
                    X[col] = X_transformed[:, i]

        # Fit scaler
        self._scaler_columns = available

        if self.config.scaling_method == 'robust':
            self._scaler = RobustScaler(quantile_range=(5.0, 95.0))
        elif self.config.scaling_method == 'standard':
            from sklearn.preprocessing import StandardScaler
            self._scaler = StandardScaler()
        elif self.config.scaling_method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            self._scaler = MinMaxScaler()

        if self._scaler is not None:
            self._scaler.fit(X.values)
            logger.info(f"Fitted {self.config.scaling_method} scaler on {len(available)} features")

    def _fit_categorical_encoders(self, df: pd.DataFrame):
        """Fit categorical encoders."""
        for col in self.categorical_features:
            if col not in df.columns:
                continue

            values = df[col].astype(str).str.upper().str.strip().unique()
            self._categorical_encoders[col] = {v: i for i, v in enumerate(sorted(values))}
            logger.debug(f"Fitted encoder for {col}: {len(values)} unique values")

    def _fit_label_encoder(self, y: pd.Series):
        """Fit label encoder for target with ordinal ordering."""
        unique_labels = y.dropna().unique()

        # Sort by C and S components for ordinal ordering
        def sort_key(label):
            match = re.match(r'C(\d)S(\d)', str(label))
            if match:
                return (int(match.group(1)), int(match.group(2)))
            return (99, 99)

        sorted_labels = sorted(unique_labels, key=sort_key)
        self._label_encoder = {str(label): i for i, label in enumerate(sorted_labels)}
        logger.info(f"Label encoder: {self._label_encoder}")

    def _compute_class_weights(self, y: pd.Series):
        """Compute class weights with special handling for high-risk classes."""
        y_encoded = y.map(self._label_encoder).dropna()
        unique_classes, counts = np.unique(y_encoded, return_counts=True)
        n_samples = len(y_encoded)
        n_classes = len(unique_classes)

        # Get inverse label mapping
        idx_to_label = {v: k for k, v in self._label_encoder.items()}

        if self.config.class_weight_strategy == 'balanced':
            # Standard sklearn balanced weights
            weights = n_samples / (n_classes * counts)

        elif self.config.class_weight_strategy == 'sqrt':
            # Square root dampening
            weights = np.sqrt(n_samples / (n_classes * counts))

        elif self.config.class_weight_strategy == 'custom_high_risk':
            # Custom weighting with extra emphasis on high-risk classes
            # Start with balanced weights
            base_weights = n_samples / (n_classes * counts)
            weights = base_weights.copy()

            # Apply multiplier for high-risk classes
            for i, cls_idx in enumerate(unique_classes):
                label = idx_to_label.get(int(cls_idx), '')
                if label in HIGH_RISK_CLASSES:
                    weights[i] *= self.config.high_risk_weight_multiplier
                    logger.info(f"High-risk class {label}: weight {weights[i]:.2f} "
                               f"(base {base_weights[i]:.2f} x {self.config.high_risk_weight_multiplier})")
        else:
            weights = np.ones(n_classes)

        self._class_weights = {int(cls): float(w) for cls, w in zip(unique_classes, weights)}
        logger.info(f"Class weights computed: {self._class_weights}")

    def _transform_numeric(
        self,
        df: pd.DataFrame,
        scale_features: bool
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """Transform numeric features."""
        available = [c for c in self.numeric_features if c in df.columns]
        if not available:
            return None, []

        # Prepare data
        X = df[available].copy()
        X = X.apply(pd.to_numeric, errors='coerce')

        # 1. Handle missing values
        for col in available:
            if col in self._numeric_stats:
                X[col] = X[col].fillna(self._numeric_stats[col]['median'])
            else:
                X[col] = X[col].fillna(0)

        # 2. Apply winsorization
        if self.config.outlier_method == 'winsorize':
            for col in available:
                if col in self._winsorize_bounds:
                    lower, upper = self._winsorize_bounds[col]
                    X[col] = X[col].clip(lower=lower, upper=upper)

        # 3. Apply power transform
        if self._power_transformer is not None and hasattr(self, '_power_transform_cols'):
            transform_cols = [c for c in self._power_transform_cols if c in available]
            if transform_cols:
                X_transform = X[transform_cols].values + 1
                X_transformed = self._power_transformer.transform(X_transform)
                for i, col in enumerate(transform_cols):
                    X[col] = X_transformed[:, i]

        # 4. Apply scaling
        if scale_features and self._scaler is not None:
            cols_to_scale = [c for c in available if c in self._scaler_columns]
            if cols_to_scale:
                X_values = X.values.astype(np.float32)
                X_values = self._scaler.transform(X_values)
                X = pd.DataFrame(X_values, columns=available)

        return X.values.astype(np.float32), available

    def _transform_categorical(
        self,
        df: pd.DataFrame
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """Transform categorical features."""
        available = [c for c in self.categorical_features if c in df.columns]
        if not available:
            return None, []

        X = np.zeros((len(df), len(available)), dtype=np.int32)

        for i, col in enumerate(available):
            if col in self._categorical_encoders:
                encoder = self._categorical_encoders[col]
                X[:, i] = df[col].apply(
                    lambda x: encoder.get(str(x).upper().strip(), -1)
                ).values

        return X, available

    def _transform_spatial(
        self,
        df: pd.DataFrame
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """Transform spatial features."""
        available = [c for c in self.spatial_features if c in df.columns]
        if not available:
            return None, []

        X = df[available].values.astype(np.float32)
        return X, available

    def _add_current_class_feature(
        self,
        df: pd.DataFrame
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """Add current year's classification as a feature."""
        if 'Classification' not in df.columns or self._label_encoder is None:
            return None, []

        encoded = df['Classification'].map(self._label_encoder).fillna(-1).values
        return encoded.reshape(-1, 1).astype(np.float32), ['current_class']

    def _create_interaction_features(
        self,
        X_numeric: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """Create key interaction features based on domain knowledge."""
        interactions = []
        interaction_names = []

        # Key interactions based on water quality science
        interaction_pairs = [
            ('EC', 'SAR'),      # Salinity-sodium interaction
            ('Na', 'Ca'),       # Sodium-calcium balance
            ('Na', 'Mg'),       # Sodium-magnesium balance
            ('Cl', 'HCO3'),     # Chloride-bicarbonate ratio
            ('EC', 'Na'),       # EC-sodium relationship
        ]

        for feat1, feat2 in interaction_pairs:
            if feat1 in feature_names and feat2 in feature_names:
                idx1 = feature_names.index(feat1)
                idx2 = feature_names.index(feat2)

                # Ratio feature (with epsilon to avoid division by zero)
                epsilon = 1e-6
                ratio = X_numeric[:, idx1] / (X_numeric[:, idx2] + epsilon)
                interactions.append(ratio.reshape(-1, 1))
                interaction_names.append(f'{feat1}_div_{feat2}')

        # Add SAR-based features if EC and SAR available
        if 'EC' in feature_names and 'SAR' in feature_names:
            ec_idx = feature_names.index('EC')
            sar_idx = feature_names.index('SAR')

            # EC * SAR interaction (higher = more problematic)
            ec_sar_product = X_numeric[:, ec_idx] * X_numeric[:, sar_idx]
            interactions.append(ec_sar_product.reshape(-1, 1))
            interaction_names.append('EC_x_SAR')

        if interactions:
            X_interact = np.concatenate(interactions, axis=1).astype(np.float32)
            return X_interact, interaction_names

        return None, []

    def get_class_weights(self) -> Optional[Dict[int, float]]:
        """Get computed class weights."""
        return self._class_weights.copy() if self._class_weights else None

    def get_label_encoder(self) -> Dict[str, int]:
        """Get the fitted label encoder."""
        return self._label_encoder.copy() if self._label_encoder else {}

    def get_inverse_label_encoder(self) -> Dict[int, str]:
        """Get inverse label encoder."""
        if self._label_encoder is None:
            return {}
        return {v: k for k, v in self._label_encoder.items()}

    def get_n_classes(self) -> int:
        """Get number of classes."""
        return len(self._label_encoder) if self._label_encoder else 0

    def get_categorical_cardinalities(self) -> Dict[str, int]:
        """Get cardinality of each categorical feature."""
        return {col: len(enc) for col, enc in self._categorical_encoders.items()}

    def get_feature_names(self) -> List[str]:
        """Get final feature names after transformation."""
        return self._final_feature_names.copy()

    def get_categorical_indices(self) -> List[int]:
        """Get indices of categorical features in final feature array."""
        if not self._final_feature_names:
            return []

        indices = []
        for i, name in enumerate(self._final_feature_names):
            if name in self.categorical_features:
                indices.append(i)
        return indices


def compute_enhanced_class_weights(
    y: np.ndarray,
    label_encoder: Dict[str, int],
    strategy: str = 'custom_high_risk',
    high_risk_multiplier: float = 3.0
) -> Dict[int, float]:
    """
    Compute enhanced class weights for severe imbalance.

    Args:
        y: Target array (encoded)
        label_encoder: Label to index mapping
        strategy: Weight strategy ('balanced', 'sqrt', 'custom_high_risk')
        high_risk_multiplier: Extra multiplier for high-risk classes

    Returns:
        Dictionary mapping class index to weight
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(unique_classes)

    idx_to_label = {v: k for k, v in label_encoder.items()}

    if strategy == 'balanced':
        weights = n_samples / (n_classes * counts)
    elif strategy == 'sqrt':
        weights = np.sqrt(n_samples / (n_classes * counts))
    elif strategy == 'custom_high_risk':
        base_weights = n_samples / (n_classes * counts)
        weights = base_weights.copy()

        for i, cls_idx in enumerate(unique_classes):
            label = idx_to_label.get(int(cls_idx), '')
            if label in HIGH_RISK_CLASSES:
                weights[i] *= high_risk_multiplier
    else:
        weights = np.ones(n_classes)

    return {int(cls): float(w) for cls, w in zip(unique_classes, weights)}


def compute_sample_weights(
    y: np.ndarray,
    class_weights: Dict[int, float]
) -> np.ndarray:
    """
    Compute per-sample weights from class weights.

    Args:
        y: Target array
        class_weights: Class weight dictionary

    Returns:
        Array of sample weights
    """
    return np.array([class_weights.get(int(label), 1.0) for label in y])
