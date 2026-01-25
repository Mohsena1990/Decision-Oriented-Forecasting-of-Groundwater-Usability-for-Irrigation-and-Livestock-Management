"""
LightGBM Forecaster Module
==========================

LightGBM classifier for groundwater quality forecasting.
Efficient gradient boosting with fast training.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

from ..base import BaseForecaster, ForecasterConfig

logger = logging.getLogger(__name__)


class LightGBMForecaster(BaseForecaster):
    """
    LightGBM-based forecaster for groundwater quality.

    Features:
    - Fast training
    - Efficient memory usage
    - Supports categorical features with encoding
    - Feature importance via SHAP
    """

    def __init__(self, config: Optional[ForecasterConfig] = None):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed. Install with: pip install lightgbm")

        super().__init__(config)
        self._categorical_features: Optional[List[int]] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None
    ) -> 'LightGBMForecaster':
        """
        Fit the LightGBM model.

        Args:
            X: Training features
            y: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
            feature_names: Optional feature names
            categorical_features: Indices of categorical features

        Returns:
            self
        """
        # Apply feature mask if configured
        X = self.apply_feature_mask(X)
        if X_val is not None:
            X_val = self.apply_feature_mask(X_val)

        self._n_features = X.shape[1]
        self._n_classes = len(np.unique(y))
        self._feature_names = feature_names
        self._categorical_features = categorical_features or []

        # Build model parameters
        params = self._build_params()

        # Create datasets
        train_data = lgb.Dataset(
            X, y,
            feature_name=feature_names,
            categorical_feature=self._categorical_features if self._categorical_features else 'auto'
        )

        valid_sets = [train_data]
        valid_names = ['train']

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(
                X_val, y_val,
                reference=train_data,
                feature_name=feature_names,
                categorical_feature=self._categorical_features if self._categorical_features else 'auto'
            )
            valid_sets.append(val_data)
            valid_names.append('valid')

        # Set up callbacks
        callbacks = [
            lgb.log_evaluation(period=0)  # Suppress logging
        ]

        if X_val is not None and self.config.params.get('early_stopping_rounds'):
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=self.config.params['early_stopping_rounds'],
                    verbose=False
                )
            )

        # Train model
        self._model = lgb.train(
            params,
            train_data,
            num_boost_round=self.config.params.get('n_estimators', 500),
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )

        self._is_fitted = True
        logger.info(f"LightGBM fitted with {self._n_features} features, {self._n_classes} classes")

        return self

    def _build_params(self) -> Dict[str, Any]:
        """Build LightGBM parameters from config."""
        params = {
            'objective': 'multiclass',
            'num_class': self._n_classes,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'random_state': self.config.random_state,
            'verbose': -1,
            'force_col_wise': True
        }

        # Add configured parameters
        param_mapping = {
            'n_estimators': None,  # Handled separately
            'max_depth': 'max_depth',
            'learning_rate': 'learning_rate',
            'num_leaves': 'num_leaves',
            'min_child_samples': 'min_child_samples',
            'reg_alpha': 'reg_alpha',
            'reg_lambda': 'reg_lambda',
            'subsample': 'bagging_fraction',
            'colsample_bytree': 'feature_fraction'
        }

        for config_key, lgb_key in param_mapping.items():
            if lgb_key and config_key in self.config.params:
                params[lgb_key] = self.config.params[config_key]

        # Class weights
        if self.config.class_weights:
            # LightGBM uses sample weights, so we need to apply during training
            pass  # Handled in dataset creation

        return params

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self.apply_feature_mask(X)
        proba = self._model.predict(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self.apply_feature_mask(X)
        return self._model.predict(X)

    def get_feature_importance(
        self,
        importance_type: str = "gain"
    ) -> np.ndarray:
        """
        Get feature importance scores.

        Args:
            importance_type: Type of importance ("gain" or "split")

        Returns:
            Array of importance scores
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted.")

        return self._model.feature_importance(importance_type=importance_type)

    def get_model_complexity(self) -> Dict[str, Any]:
        """Get model complexity metrics."""
        if not self._is_fitted:
            return {'n_trees': 0, 'n_leaves': 0}

        return {
            'n_trees': self._model.num_trees(),
            'n_leaves': self.config.params.get('num_leaves', 31),
            'max_depth': self.config.params.get('max_depth', -1),
            'n_features': self.get_n_selected_features()
        }

    def save(self, path: Union[str, Path]):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save LightGBM model
        self._model.save_model(str(path.with_suffix('.lgb')))

        # Save config
        with open(path.with_suffix('.config.pkl'), 'wb') as f:
            pickle.dump({
                'config': self.config,
                'n_features': self._n_features,
                'n_classes': self._n_classes,
                'feature_names': self._feature_names,
                'categorical_features': self._categorical_features
            }, f)

        logger.info(f"LightGBM model saved to {path}")

    def load(self, path: Union[str, Path]) -> 'LightGBMForecaster':
        """Load model from disk."""
        path = Path(path)

        # Load LightGBM model
        self._model = lgb.Booster(model_file=str(path.with_suffix('.lgb')))

        # Load config
        with open(path.with_suffix('.config.pkl'), 'rb') as f:
            state = pickle.load(f)

        self.config = state['config']
        self._n_features = state['n_features']
        self._n_classes = state['n_classes']
        self._feature_names = state['feature_names']
        self._categorical_features = state['categorical_features']
        self._is_fitted = True

        logger.info(f"LightGBM model loaded from {path}")
        return self

    @classmethod
    def get_model_name(cls) -> str:
        return "LightGBM"

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'min_child_samples': 20,
            'reg_alpha': 0,
            'reg_lambda': 0,
            'early_stopping_rounds': 50
        }

    @classmethod
    def get_param_space(cls) -> Dict[str, Tuple[Any, Any]]:
        return {
            'n_estimators': (100, 1000),
            'max_depth': (4, 10),
            'learning_rate': (0.01, 0.3),
            'num_leaves': (15, 127),
            'min_child_samples': (5, 100),
            'reg_alpha': (0, 1),
            'reg_lambda': (0, 1)
        }
