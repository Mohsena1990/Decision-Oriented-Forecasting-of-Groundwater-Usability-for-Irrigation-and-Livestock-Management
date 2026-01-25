"""
Fidelity Checker Module
=======================

Measures how well a surrogate model mimics the original model.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class FidelityChecker:
    """
    Check fidelity of surrogate models.

    Computes various metrics to assess how well the surrogate
    approximates the original model's predictions.
    """

    def compute_fidelity(
        self,
        original_model: Any,
        surrogate_model: Any,
        X: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute fidelity metrics between original and surrogate.

        Args:
            original_model: Original model (deep learning)
            surrogate_model: Surrogate model (tree-based)
            X: Data to evaluate on

        Returns:
            Dictionary of fidelity metrics
        """
        # Get predictions from both models
        original_proba = original_model.predict_proba(X)
        original_pred = np.argmax(original_proba, axis=1)

        # Handle different surrogate output formats
        if hasattr(surrogate_model, 'predict_proba'):
            surrogate_proba = surrogate_model.predict_proba(X)
        elif hasattr(surrogate_model, 'predict'):
            surrogate_proba = surrogate_model.predict(X)
        else:
            raise ValueError("Surrogate model must have predict or predict_proba method")

        surrogate_pred = np.argmax(surrogate_proba, axis=1)

        metrics = {}

        # Prediction agreement
        metrics['agreement'] = np.mean(original_pred == surrogate_pred)

        # Probability correlation (using mean probability per sample)
        if len(original_proba.shape) == 2 and len(surrogate_proba.shape) == 2:
            # Use max probability as summary
            orig_max_prob = np.max(original_proba, axis=1)
            surr_max_prob = np.max(surrogate_proba, axis=1)

            if np.std(orig_max_prob) > 0 and np.std(surr_max_prob) > 0:
                metrics['correlation'], _ = stats.pearsonr(orig_max_prob, surr_max_prob)
            else:
                metrics['correlation'] = 0.0

            # R-squared on probabilities
            metrics['r2'] = self._compute_r2(original_proba.flatten(), surrogate_proba.flatten())

            # KL divergence (average across samples)
            kl_divs = []
            for i in range(len(X)):
                kl = self._kl_divergence(original_proba[i], surrogate_proba[i])
                if not np.isnan(kl) and not np.isinf(kl):
                    kl_divs.append(kl)

            if kl_divs:
                metrics['kl_divergence'] = np.mean(kl_divs)
            else:
                metrics['kl_divergence'] = np.nan

        else:
            metrics['correlation'] = 0.0
            metrics['r2'] = 0.0
            metrics['kl_divergence'] = np.nan

        # Class-wise agreement
        n_classes = original_proba.shape[1] if len(original_proba.shape) == 2 else 2
        for c in range(n_classes):
            mask = original_pred == c
            if mask.sum() > 0:
                metrics[f'agreement_class_{c}'] = np.mean(surrogate_pred[mask] == c)

        logger.info(f"Fidelity metrics: agreement={metrics['agreement']:.3f}, "
                   f"correlation={metrics['correlation']:.3f}, r2={metrics['r2']:.3f}")

        return metrics

    def _compute_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R-squared."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1 - (ss_res / ss_tot)

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
        """Compute KL divergence between two probability distributions."""
        p = np.clip(p, epsilon, 1 - epsilon)
        q = np.clip(q, epsilon, 1 - epsilon)

        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)

        return np.sum(p * np.log(p / q))

    def is_acceptable(
        self,
        metrics: Dict[str, float],
        min_r2: float = 0.85,
        min_correlation: float = 0.90,
        min_agreement: float = 0.80
    ) -> bool:
        """
        Check if fidelity is acceptable for surrogate explanations.

        Args:
            metrics: Fidelity metrics dictionary
            min_r2: Minimum R-squared threshold
            min_correlation: Minimum correlation threshold
            min_agreement: Minimum prediction agreement threshold

        Returns:
            True if fidelity is acceptable
        """
        r2_ok = metrics.get('r2', 0) >= min_r2
        corr_ok = metrics.get('correlation', 0) >= min_correlation
        agree_ok = metrics.get('agreement', 0) >= min_agreement

        return r2_ok and corr_ok and agree_ok
