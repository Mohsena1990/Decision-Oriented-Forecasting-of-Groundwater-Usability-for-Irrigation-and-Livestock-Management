"""
Metrics Module
==============

Comprehensive metrics for groundwater quality forecasting evaluation,
including ordinal-aware metrics and severity-focused measures.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

logger = logging.getLogger(__name__)


@dataclass
class MetricsCalculator:
    """
    Calculate comprehensive metrics for groundwater classification.

    Includes standard classification metrics plus domain-specific
    ordinal and severity-aware metrics.
    """

    # Ordinal tier index for each tier label (single int per tier after 3-tier merge)
    label_to_ordinal: Dict[str, int] = field(default_factory=dict)

    # High-risk class indices
    high_risk_indices: List[int] = field(default_factory=list)

    # Index to label mapping
    idx_to_label: Dict[int, str] = field(default_factory=dict)

    # Weight for ordinal distance (single tier dimension)
    tier_weight: float = 1.0

    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute all metrics.

        Args:
            y_true: True label indices
            y_pred: Predicted label indices
            y_proba: Predicted probabilities (optional)

        Returns:
            Dictionary of metric name to value
        """
        metrics = {}

        # Standard classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)

        # Ordinal distance metrics
        ordinal_metrics = self._compute_ordinal_metrics(y_true, y_pred)
        metrics.update(ordinal_metrics)

        # Severity-aware metrics
        severity_metrics = self._compute_severity_metrics(y_true, y_pred, y_proba)
        metrics.update(severity_metrics)

        return metrics

    def _compute_ordinal_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute ordinal distance-based metrics."""
        metrics = {}

        if not self.label_to_ordinal or not self.idx_to_label:
            logger.debug("Ordinal mapping not available, skipping ordinal metrics")
            return metrics

        distances = []
        for true_idx, pred_idx in zip(y_true, y_pred):
            true_label = self.idx_to_label.get(int(true_idx))
            pred_label = self.idx_to_label.get(int(pred_idx))

            if true_label and pred_label:
                true_ord = self.label_to_ordinal.get(true_label)
                pred_ord = self.label_to_ordinal.get(pred_label)

                if true_ord is not None and pred_ord is not None:
                    dist = self.tier_weight * abs(true_ord - pred_ord)
                    distances.append(dist)

        if distances:
            distances = np.array(distances)
            metrics['ordinal_distance_mean'] = float(np.mean(distances))
            metrics['ordinal_distance_max'] = float(np.max(distances))
            metrics['ordinal_distance_zero_rate'] = float(np.mean(distances == 0))

        return metrics

    def _compute_severity_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Compute severity-focused metrics for high-risk classes."""
        metrics = {}

        if not self.high_risk_indices:
            logger.debug("High-risk indices not defined, skipping severity metrics")
            return metrics

        # Binary mask for high-risk classes
        true_high_risk = np.isin(y_true, self.high_risk_indices)
        pred_high_risk = np.isin(y_pred, self.high_risk_indices)

        # Count true/false positives/negatives for high-risk
        true_positives = np.sum(true_high_risk & pred_high_risk)
        false_negatives = np.sum(true_high_risk & ~pred_high_risk)
        false_positives = np.sum(~true_high_risk & pred_high_risk)
        true_negatives = np.sum(~true_high_risk & ~pred_high_risk)

        n_true_high_risk = np.sum(true_high_risk)
        n_pred_high_risk = np.sum(pred_high_risk)

        # False Negative Rate for high-risk (CRITICAL METRIC)
        # This is the rate at which we miss high-risk cases
        if n_true_high_risk > 0:
            metrics['severe_fnr'] = false_negatives / n_true_high_risk
            metrics['severe_recall'] = true_positives / n_true_high_risk
        else:
            metrics['severe_fnr'] = 0.0
            metrics['severe_recall'] = 1.0

        # False Positive Rate for high-risk (over-warning)
        n_true_low_risk = len(y_true) - n_true_high_risk
        if n_true_low_risk > 0:
            metrics['severe_fpr'] = false_positives / n_true_low_risk
        else:
            metrics['severe_fpr'] = 0.0

        # Precision for high-risk
        if n_pred_high_risk > 0:
            metrics['severe_precision'] = true_positives / n_pred_high_risk
        else:
            metrics['severe_precision'] = 0.0

        # High-risk F1
        if metrics['severe_precision'] + metrics['severe_recall'] > 0:
            metrics['severe_f1'] = 2 * (metrics['severe_precision'] * metrics['severe_recall']) / \
                                   (metrics['severe_precision'] + metrics['severe_recall'])
        else:
            metrics['severe_f1'] = 0.0

        # Counts for reference
        metrics['n_true_high_risk'] = int(n_true_high_risk)
        metrics['n_pred_high_risk'] = int(n_pred_high_risk)

        return metrics

    def get_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[int]] = None
    ) -> np.ndarray:
        """Get confusion matrix."""
        return confusion_matrix(y_true, y_pred, labels=labels)

    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None
    ) -> str:
        """Get sklearn classification report."""
        return classification_report(
            y_true, y_pred,
            target_names=target_names,
            zero_division=0
        )

    def get_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Get per-class metrics."""
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        per_class = {}

        for cls in unique_classes:
            cls_label = self.idx_to_label.get(int(cls), str(cls))
            mask_true = y_true == cls
            mask_pred = y_pred == cls

            tp = np.sum(mask_true & mask_pred)
            fp = np.sum(~mask_true & mask_pred)
            fn = np.sum(mask_true & ~mask_pred)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            per_class[cls_label] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': int(np.sum(mask_true))
            }

        return per_class


def create_metrics_function(
    label_to_ordinal: Dict[str, int],
    idx_to_label: Dict[int, str],
    high_risk_indices: List[int],
    tier_weight: float = 1.0
) -> callable:
    """
    Create a metrics function for use in CV evaluation.

    Returns a function with signature (y_true, y_pred, y_proba) -> Dict[str, float]
    """
    calculator = MetricsCalculator(
        label_to_ordinal=label_to_ordinal,
        idx_to_label=idx_to_label,
        high_risk_indices=high_risk_indices,
        tier_weight=tier_weight,
    )

    def metrics_fn(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        return calculator.compute_all(y_true, y_pred, y_proba)

    return metrics_fn
