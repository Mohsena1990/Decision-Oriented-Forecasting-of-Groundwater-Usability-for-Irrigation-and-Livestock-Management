"""
Cross-Validation Module
=======================

Implements stratified cross-validation for hyperparameter optimization
within the training set.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


@dataclass
class CVResult:
    """Container for cross-validation results."""
    fold_scores: List[Dict[str, float]]
    mean_scores: Dict[str, float]
    std_scores: Dict[str, float]
    fold_predictions: Optional[List[np.ndarray]] = None
    fold_probabilities: Optional[List[np.ndarray]] = None


@dataclass
class CrossValidator:
    """
    Stratified cross-validator for model evaluation.

    Provides consistent CV evaluation across all models,
    ensuring fair comparison.
    """

    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42
    return_predictions: bool = False

    _indices: Optional[List[Tuple[np.ndarray, np.ndarray]]] = field(default=None, init=False)

    def fit(self, y: np.ndarray) -> 'CrossValidator':
        """
        Generate CV fold indices based on target.

        Args:
            y: Target array for stratification

        Returns:
            self
        """
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
        self._indices = list(skf.split(np.zeros(len(y)), y))

        logger.info(f"Created {self.n_splits} stratified CV folds")
        return self

    def get_indices(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get the pre-computed fold indices."""
        if self._indices is None:
            raise ValueError("CrossValidator not fitted. Call fit() first.")
        return self._indices

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_factory: Callable[[], Any],
        metrics_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], Dict[str, float]]
    ) -> CVResult:
        """
        Evaluate a model using cross-validation.

        Args:
            X: Feature array
            y: Target array
            model_factory: Function that creates a new model instance
            metrics_fn: Function that computes metrics from (y_true, y_pred, y_proba)

        Returns:
            CVResult with fold-wise and aggregated scores
        """
        if self._indices is None:
            self.fit(y)

        fold_scores = []
        fold_predictions = [] if self.return_predictions else None
        fold_probabilities = [] if self.return_predictions else None

        for fold_idx, (train_idx, val_idx) in enumerate(self._indices):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Create and train model
            model = model_factory()
            model.fit(X_train, y_train)

            # Get predictions
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)

            # Compute metrics
            scores = metrics_fn(y_val, y_pred, y_proba)
            scores['fold'] = fold_idx
            fold_scores.append(scores)

            if self.return_predictions:
                fold_predictions.append(y_pred)
                fold_probabilities.append(y_proba)

            logger.debug(f"Fold {fold_idx + 1}/{self.n_splits}: {scores}")

        # Aggregate results
        mean_scores = {}
        std_scores = {}

        metric_keys = [k for k in fold_scores[0].keys() if k != 'fold']
        for key in metric_keys:
            values = [s[key] for s in fold_scores]
            mean_scores[key] = float(np.mean(values))
            std_scores[key] = float(np.std(values))

        return CVResult(
            fold_scores=fold_scores,
            mean_scores=mean_scores,
            std_scores=std_scores,
            fold_predictions=fold_predictions,
            fold_probabilities=fold_probabilities
        )

    def evaluate_with_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_factory: Callable[[], Any],
        metrics_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], Dict[str, float]],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Tuple[CVResult, Optional[Dict[str, float]]]:
        """
        Evaluate with both CV and optional held-out validation set.

        Args:
            X: Training feature array
            y: Training target array
            model_factory: Function that creates a new model instance
            metrics_fn: Function that computes metrics
            X_val: Optional validation features
            y_val: Optional validation targets

        Returns:
            Tuple of (CV result, validation scores if provided)
        """
        # Run CV evaluation
        cv_result = self.evaluate(X, y, model_factory, metrics_fn)

        # Evaluate on held-out validation if provided
        val_scores = None
        if X_val is not None and y_val is not None:
            # Train on full training data
            model = model_factory()
            model.fit(X, y)

            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)
            val_scores = metrics_fn(y_val, y_pred, y_proba)

        return cv_result, val_scores


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        metric_fn: Function computing metric from (y_true, y_pred)
        n_iterations: Number of bootstrap iterations
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random seed

    Returns:
        Tuple of (point estimate, lower bound, upper bound)
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)
    bootstrap_scores = []

    for _ in range(n_iterations):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        score = metric_fn(y_true[indices], y_pred[indices])
        bootstrap_scores.append(score)

    bootstrap_scores = np.array(bootstrap_scores)
    point_estimate = metric_fn(y_true, y_pred)

    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
    upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))

    return point_estimate, lower, upper
