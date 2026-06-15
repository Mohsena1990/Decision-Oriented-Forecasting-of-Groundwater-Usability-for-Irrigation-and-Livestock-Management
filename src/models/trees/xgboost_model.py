"""
XGBoost Forecaster Module
=========================

XGBoost classifier for groundwater quality forecasting.
Regularized gradient boosting with depth-wise tree growth.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    xgb = None

from ..base import BaseForecaster, ForecasterConfig

logger = logging.getLogger(__name__)


class XGBoostForecaster(BaseForecaster):
    """
    XGBoost-based forecaster for groundwater quality.

    Features:
    - Regularized gradient boosting (L1 + L2)
    - Native SHAP support via xgboost.XGBClassifier
    - Class imbalance handled via sample_weight in fit()
    - Early stopping via eval set
    """

    def __init__(self, config: Optional[ForecasterConfig] = None):
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")

        super().__init__(config)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None
    ) -> 'XGBoostForecaster':
        """
        Fit the XGBoost model.

        Args:
            X: Training features
            y: Training targets
            X_val: Optional validation features (for early stopping)
            y_val: Optional validation targets
            feature_names: Optional feature names
            categorical_features: Ignored (XGBoost does not support native categorical in multi-class softprob)

        Returns:
            self
        """
        X = self.apply_feature_mask(X)
        if X_val is not None:
            X_val = self.apply_feature_mask(X_val)

        self._n_features = X.shape[1]
        self._n_classes = len(np.unique(y))
        self._feature_names = feature_names

        # Compute sample weights from class_weights config
        sample_weight = self._compute_sample_weight(y)

        params = self._build_params()

        self._model = xgb.XGBClassifier(**params)

        fit_kwargs: Dict[str, Any] = {
            'sample_weight': sample_weight,
        }

        if X_val is not None and y_val is not None:
            # Provide both train and val so evals_result_ has two named sets
            fit_kwargs['eval_set'] = [(X, y), (X_val, y_val)]
            fit_kwargs['verbose'] = False

        self._model.fit(X, y, **fit_kwargs)

        # Build training_history for animation (per-round logloss)
        evals = getattr(self._model, 'evals_result_', {})
        train_loss = list(evals.get('validation_0', {}).get('mlogloss', []))
        val_loss = list(evals.get('validation_1', {}).get('mlogloss', []))
        self.training_history = {
            'train_loss': train_loss,
            'val_loss': val_loss if val_loss else train_loss,
        }

        self._is_fitted = True
        logger.info(f"XGBoost fitted with {self._n_features} features, {self._n_classes} classes")

        return self

    def _compute_sample_weight(self, y: np.ndarray) -> np.ndarray:
        """Compute per-sample weights from class_weights config."""
        if self.config.class_weights:
            weights = np.ones(len(y), dtype=float)
            for cls_idx, w in self.config.class_weights.items():
                weights[y == cls_idx] = w
            return weights
        return np.ones(len(y), dtype=float)

    def _build_params(self) -> Dict[str, Any]:
        """Build XGBoost parameters from config."""
        n_classes = self._n_classes or self.config.n_classes or 3

        params: Dict[str, Any] = {
            'objective': 'multi:softprob',
            'num_class': n_classes,
            'eval_metric': 'mlogloss',
            'random_state': self.config.random_state,
            'verbosity': 0,
            'use_label_encoder': False,
        }

        param_mapping = {
            'n_estimators': 'n_estimators',
            'max_depth': 'max_depth',
            'learning_rate': 'learning_rate',
            'subsample': 'subsample',
            'colsample_bytree': 'colsample_bytree',
            'reg_alpha': 'reg_alpha',
            'reg_lambda': 'reg_lambda',
            'min_child_weight': 'min_child_weight',
            'early_stopping_rounds': 'early_stopping_rounds',
            'n_jobs': 'n_jobs',
            'tree_method': 'tree_method',
            'device': 'device',
        }

        for config_key, xgb_key in param_mapping.items():
            if config_key in self.config.params:
                params[xgb_key] = self.config.params[config_key]

        # Remove early_stopping_rounds from params if no eval_set will be provided
        # (it is passed at fit-time conditionally; keep it in params so XGBClassifier stores it)
        return params

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self.apply_feature_mask(X)
        return self._model.predict(X).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self.apply_feature_mask(X)
        return self._model.predict_proba(X)

    def get_feature_importance(self, importance_type: str = "gain") -> np.ndarray:
        """Get feature importance scores."""
        if not self._is_fitted:
            raise ValueError("Model not fitted.")

        scores = self._model.get_booster().get_score(importance_type=importance_type)
        # Return array aligned to feature order
        if self._feature_names:
            return np.array([scores.get(f, 0.0) for f in self._feature_names])
        n = self._n_features or len(scores)
        return np.array([scores.get(f'f{i}', 0.0) for i in range(n)])

    def get_model_complexity(self) -> Dict[str, Any]:
        """Get model complexity metrics."""
        if not self._is_fitted:
            return {'n_trees': 0, 'max_depth': 0}

        booster = self._model.get_booster()
        dump = booster.get_dump()
        return {
            'n_trees': len(dump),
            'max_depth': self.config.params.get('max_depth', 6),
            'n_features': self.get_n_selected_features(),
        }

    def save(self, path: Union[str, Path]):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save XGBoost booster
        self._model.get_booster().save_model(str(path.with_suffix('.xgb')))

        # Save sklearn wrapper + config
        with open(path.with_suffix('.config.pkl'), 'wb') as f:
            pickle.dump({
                'config': self.config,
                'n_features': self._n_features,
                'n_classes': self._n_classes,
                'feature_names': self._feature_names,
                'xgb_params': self._model.get_params(),
            }, f)

        logger.info(f"XGBoost model saved to {path}")

    def load(self, path: Union[str, Path]) -> 'XGBoostForecaster':
        """Load model from disk."""
        path = Path(path)

        with open(path.with_suffix('.config.pkl'), 'rb') as f:
            state = pickle.load(f)

        self.config = state['config']
        self._n_features = state['n_features']
        self._n_classes = state['n_classes']
        self._feature_names = state['feature_names']

        # Reconstruct XGBClassifier from saved params
        xgb_params = state.get('xgb_params', {})
        self._model = xgb.XGBClassifier(**xgb_params)

        # Build a dummy booster to load into
        booster = xgb.Booster()
        booster.load_model(str(path.with_suffix('.xgb')))
        self._model._Booster = booster

        self._is_fitted = True
        logger.info(f"XGBoost model loaded from {path}")
        return self

    @classmethod
    def get_model_name(cls) -> str:
        return "XGBoost"

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_weight': 3,
            'early_stopping_rounds': 50,
        }

    @classmethod
    def get_param_space(cls) -> Dict[str, Tuple[Any, Any]]:
        return {
            'n_estimators': (500, 5000),
            'max_depth': (3, 9),
            'learning_rate': (0.001, 0.1),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0),
            'reg_alpha': (0.0, 1.0),
            'reg_lambda': (0.5, 2.0),
            'min_child_weight': (1, 7),
        }
