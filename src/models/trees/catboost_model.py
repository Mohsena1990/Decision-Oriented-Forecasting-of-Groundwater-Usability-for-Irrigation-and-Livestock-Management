"""
CatBoost Forecaster Module
==========================

CatBoost classifier for groundwater quality forecasting.
Handles categorical features natively without encoding.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from catboost import CatBoostClassifier, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    CatBoostClassifier = None
    Pool = None

from ..base import BaseForecaster, ForecasterConfig

logger = logging.getLogger(__name__)


class CatBoostForecaster(BaseForecaster):
    """
    CatBoost-based forecaster for groundwater quality.

    Features:
    - Native categorical feature handling
    - Built-in class weighting
    - GPU support available
    - Feature importance via SHAP
    """

    def __init__(self, config: Optional[ForecasterConfig] = None):
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not installed. Install with: pip install catboost")

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
    ) -> 'CatBoostForecaster':
        """
        Fit the CatBoost model.

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

        # Create model
        self._model = CatBoostClassifier(**params)

        # Prepare data
        train_pool = Pool(
            X, y,
            cat_features=self._categorical_features if self._categorical_features else None,
            feature_names=feature_names
        )

        eval_pool = None
        if X_val is not None and y_val is not None:
            eval_pool = Pool(
                X_val, y_val,
                cat_features=self._categorical_features if self._categorical_features else None,
                feature_names=feature_names
            )

        # Fit model
        self._model.fit(
            train_pool,
            eval_set=eval_pool,
            verbose=False,
            plot=False
        )

        self._is_fitted = True
        logger.info(f"CatBoost fitted with {self._n_features} features, {self._n_classes} classes")

        return self

    def _build_params(self) -> Dict[str, Any]:
        """Build CatBoost parameters from config."""
        params = {
            'loss_function': 'MultiClass',
            'eval_metric': 'MultiClass',
            'random_seed': self.config.random_state,
            'verbose': False,
            'allow_writing_files': False
        }

        # Add configured parameters
        param_mapping = {
            'iterations': 'iterations',
            'depth': 'depth',
            'learning_rate': 'learning_rate',
            'l2_leaf_reg': 'l2_leaf_reg',
            'border_count': 'border_count',
            'min_data_in_leaf': 'min_data_in_leaf'
        }

        for config_key, catboost_key in param_mapping.items():
            if config_key in self.config.params:
                params[catboost_key] = self.config.params[config_key]

        # Class weights
        if self.config.class_weights:
            params['class_weights'] = list(self.config.class_weights.values())

        # Early stopping
        if self.config.params.get('early_stopping_rounds'):
            params['early_stopping_rounds'] = self.config.params['early_stopping_rounds']

        return params

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self.apply_feature_mask(X)
        return self._model.predict(X).flatten().astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self.apply_feature_mask(X)
        return self._model.predict_proba(X)

    def get_feature_importance(
        self,
        importance_type: str = "FeatureImportance"
    ) -> np.ndarray:
        """
        Get feature importance scores.

        Args:
            importance_type: Type of importance ("FeatureImportance" or "ShapValues")

        Returns:
            Array of importance scores
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted.")

        return self._model.get_feature_importance(type=importance_type)

    def get_model_complexity(self) -> Dict[str, Any]:
        """Get model complexity metrics."""
        if not self._is_fitted:
            return {'n_trees': 0, 'depth': 0}

        return {
            'n_trees': self._model.tree_count_,
            'depth': self.config.params.get('depth', 6),
            'n_features': self.get_n_selected_features()
        }

    def save(self, path: Union[str, Path]):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save CatBoost model
        self._model.save_model(str(path.with_suffix('.cbm')))

        # Save config
        with open(path.with_suffix('.config.pkl'), 'wb') as f:
            pickle.dump({
                'config': self.config,
                'n_features': self._n_features,
                'n_classes': self._n_classes,
                'feature_names': self._feature_names,
                'categorical_features': self._categorical_features
            }, f)

        logger.info(f"CatBoost model saved to {path}")

    def load(self, path: Union[str, Path]) -> 'CatBoostForecaster':
        """Load model from disk."""
        path = Path(path)

        # Load CatBoost model
        self._model = CatBoostClassifier()
        self._model.load_model(str(path.with_suffix('.cbm')))

        # Load config
        with open(path.with_suffix('.config.pkl'), 'rb') as f:
            state = pickle.load(f)

        self.config = state['config']
        self._n_features = state['n_features']
        self._n_classes = state['n_classes']
        self._feature_names = state['feature_names']
        self._categorical_features = state['categorical_features']
        self._is_fitted = True

        logger.info(f"CatBoost model loaded from {path}")
        return self

    @classmethod
    def get_model_name(cls) -> str:
        return "CatBoost"

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            'iterations': 500,
            'depth': 6,
            'learning_rate': 0.1,
            'l2_leaf_reg': 3,
            'border_count': 64,
            'early_stopping_rounds': 50
        }

    @classmethod
    def get_param_space(cls) -> Dict[str, Tuple[Any, Any]]:
        return {
            'iterations': (100, 1000),
            'depth': (4, 10),
            'learning_rate': (0.01, 0.3),
            'l2_leaf_reg': (1, 10),
            'border_count': (32, 255)
        }
