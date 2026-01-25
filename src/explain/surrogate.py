"""
Surrogate SHAP Explainer Module
===============================

Explains deep learning models using surrogate tree models.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .shap_tree import TreeSHAPExplainer, SHAPResult
from .fidelity import FidelityChecker

logger = logging.getLogger(__name__)


@dataclass
class SurrogateResult:
    """Container for surrogate explanation results."""
    shap_result: SHAPResult
    fidelity_metrics: Dict[str, float]
    surrogate_model: Any
    is_reliable: bool


@dataclass
class SurrogateExplainer:
    """
    Explain deep learning models using surrogate tree models.

    Trains a tree model (LightGBM) to mimic the deep model's
    predictions, then uses TreeSHAP for explanations.
    """

    surrogate_type: str = "lightgbm"  # or "catboost"
    min_fidelity_r2: float = 0.85
    min_fidelity_corr: float = 0.90
    max_samples: int = 1000
    top_k_features: int = 10

    def explain(
        self,
        deep_model: Any,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> SurrogateResult:
        """
        Explain deep model using surrogate.

        Args:
            deep_model: Deep learning model (GRU/LSTM)
            X: Data to explain
            feature_names: Feature names

        Returns:
            SurrogateResult with SHAP values and fidelity metrics
        """
        # Get deep model predictions
        proba = deep_model.predict_proba(X)

        # Train surrogate model
        surrogate = self._train_surrogate(X, proba)

        # Check fidelity
        fidelity_checker = FidelityChecker()
        fidelity_metrics = fidelity_checker.compute_fidelity(
            deep_model, surrogate, X
        )

        is_reliable = (
            fidelity_metrics.get('r2', 0) >= self.min_fidelity_r2 and
            fidelity_metrics.get('correlation', 0) >= self.min_fidelity_corr
        )

        if not is_reliable:
            logger.warning(
                f"Surrogate fidelity below threshold: R2={fidelity_metrics.get('r2', 0):.3f}, "
                f"Corr={fidelity_metrics.get('correlation', 0):.3f}"
            )

        # Compute SHAP on surrogate
        tree_explainer = TreeSHAPExplainer(
            max_samples=self.max_samples,
            top_k_features=self.top_k_features
        )
        shap_result = tree_explainer.explain(surrogate, X, feature_names)

        return SurrogateResult(
            shap_result=shap_result,
            fidelity_metrics=fidelity_metrics,
            surrogate_model=surrogate,
            is_reliable=is_reliable
        )

    def _train_surrogate(
        self,
        X: np.ndarray,
        target_proba: np.ndarray
    ) -> Any:
        """Train surrogate tree model to mimic deep model."""
        if self.surrogate_type == "lightgbm":
            return self._train_lightgbm_surrogate(X, target_proba)
        elif self.surrogate_type == "catboost":
            return self._train_catboost_surrogate(X, target_proba)
        else:
            raise ValueError(f"Unknown surrogate type: {self.surrogate_type}")

    def _train_lightgbm_surrogate(
        self,
        X: np.ndarray,
        target_proba: np.ndarray
    ) -> Any:
        """Train LightGBM surrogate."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM required for surrogate. Install with: pip install lightgbm")

        # For multi-class, train on argmax (classification)
        y = np.argmax(target_proba, axis=1)
        n_classes = target_proba.shape[1]

        params = {
            'objective': 'multiclass' if n_classes > 2 else 'binary',
            'num_class': n_classes if n_classes > 2 else None,
            'metric': 'multi_logloss' if n_classes > 2 else 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'verbose': -1,
            'random_state': 42
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        train_data = lgb.Dataset(X, y)
        model = lgb.train(params, train_data, num_boost_round=100)

        return model

    def _train_catboost_surrogate(
        self,
        X: np.ndarray,
        target_proba: np.ndarray
    ) -> Any:
        """Train CatBoost surrogate."""
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            raise ImportError("CatBoost required for surrogate. Install with: pip install catboost")

        y = np.argmax(target_proba, axis=1)

        model = CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            loss_function='MultiClass',
            verbose=False,
            random_seed=42
        )
        model.fit(X, y)

        return model
