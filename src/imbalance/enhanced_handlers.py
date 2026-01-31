"""
Enhanced Imbalance Handling Module
==================================

Professional strategies for handling severe class imbalance (234:1 ratio).

Includes:
1. Custom high-risk class weighting with validation
2. Focal Loss for deep learning (addresses class imbalance better than CE)
3. Decision threshold optimization with proper fallback
4. Probability calibration before thresholding
5. Cost matrix interface for cost-sensitive learning
6. Proper metrics reporting for imbalanced data
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# High-risk classes that need special attention
HIGH_RISK_CLASSES = ['C4S1', 'C4S2', 'C4S3', 'C4S4', 'C3S3', 'C3S4']


# =============================================================================
# COST MATRIX INTERFACE
# =============================================================================

@dataclass
class CostMatrix:
    """
    Cost matrix for cost-sensitive learning.

    Explicit costs make the decision-making framework defensible.
    """
    severe_fn_cost: float = 10.0   # Missing a high-risk case
    severe_fp_cost: float = 2.0    # False alarm for high-risk
    mild_fn_cost: float = 1.0      # Missing a mild misclassification
    mild_fp_cost: float = 1.0      # False alarm for mild case
    ordinal_distance_cost: float = 0.5  # Per-step ordinal distance

    def get_misclassification_cost(
        self,
        true_class: int,
        pred_class: int,
        high_risk_indices: List[int],
        ordinal_mapping: Optional[Dict[int, Tuple[int, int]]] = None
    ) -> float:
        """
        Compute cost of a specific misclassification.

        Args:
            true_class: True class index
            pred_class: Predicted class index
            high_risk_indices: Indices of high-risk classes
            ordinal_mapping: Optional ordinal encoding (class -> (C, S))

        Returns:
            Cost value
        """
        if true_class == pred_class:
            return 0.0

        true_is_high_risk = true_class in high_risk_indices
        pred_is_high_risk = pred_class in high_risk_indices

        # High-risk false negative (missed detection)
        if true_is_high_risk and not pred_is_high_risk:
            base_cost = self.severe_fn_cost
        # High-risk false positive (false alarm)
        elif not true_is_high_risk and pred_is_high_risk:
            base_cost = self.severe_fp_cost
        # Mild misclassification
        else:
            base_cost = self.mild_fn_cost

        # Add ordinal distance cost if mapping provided
        if ordinal_mapping and true_class in ordinal_mapping and pred_class in ordinal_mapping:
            true_c, true_s = ordinal_mapping[true_class]
            pred_c, pred_s = ordinal_mapping[pred_class]
            distance = abs(true_c - pred_c) + abs(true_s - pred_s)
            base_cost += distance * self.ordinal_distance_cost

        return base_cost


def compute_enhanced_class_weights(
    y: np.ndarray,
    label_encoder: Dict[str, int],
    strategy: str = 'custom_high_risk',
    high_risk_multiplier: float = 3.0,
    effective_number_beta: float = 0.999,
    validate_mapping: bool = True
) -> Dict[int, float]:
    """
    Compute enhanced class weights for severe imbalance.

    Strategies:
    - 'balanced': Standard sklearn balanced (n_samples / (n_classes * n_samples_per_class))
    - 'sqrt': Square root dampening
    - 'custom_high_risk': Balanced + extra multiplier for high-risk classes
    - 'effective_number': Based on "Class-Balanced Loss" paper (Cui et al., 2019)
    - 'inverse_freq': Simple inverse frequency

    Args:
        y: Target array (encoded)
        label_encoder: Label to index mapping
        strategy: Weight strategy
        high_risk_multiplier: Extra multiplier for high-risk classes
        effective_number_beta: Beta for effective number weighting
        validate_mapping: Whether to validate label_encoder consistency

    Returns:
        Dictionary mapping class index to weight
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(unique_classes)

    idx_to_label = {v: k for k, v in label_encoder.items()}

    # VALIDATION: Check for mapping consistency
    if validate_mapping and label_encoder:
        missing_in_encoder = set(unique_classes) - set(label_encoder.values())
        if missing_in_encoder:
            logger.warning(f"Class indices in y not found in label_encoder: {missing_in_encoder}")

        missing_in_y = set(label_encoder.values()) - set(unique_classes)
        if missing_in_y:
            logger.info(f"Classes in label_encoder not present in y: {missing_in_y}")

    if strategy == 'balanced':
        weights = n_samples / (n_classes * counts)

    elif strategy == 'sqrt':
        weights = np.sqrt(n_samples / (n_classes * counts))

    elif strategy == 'inverse_freq':
        weights = 1.0 / counts
        weights = weights / weights.sum() * n_classes  # Normalize

    elif strategy == 'effective_number':
        # From "Class-Balanced Loss Based on Effective Number of Samples"
        # Effective number = (1 - beta^n) / (1 - beta)
        effective_num = (1 - np.power(effective_number_beta, counts)) / (1 - effective_number_beta)
        weights = 1.0 / effective_num
        weights = weights / weights.sum() * n_classes  # Normalize

    elif strategy == 'custom_high_risk':
        # Start with balanced weights
        base_weights = n_samples / (n_classes * counts)
        weights = base_weights.copy()

        # Apply multiplier for high-risk classes
        high_risk_applied = 0
        for i, cls_idx in enumerate(unique_classes):
            label = idx_to_label.get(int(cls_idx), '')

            if label == '':
                logger.warning(f"No label name for class index {cls_idx}; cannot apply high-risk check.")
                continue

            if label in HIGH_RISK_CLASSES:
                weights[i] *= high_risk_multiplier
                high_risk_applied += 1
                logger.info(f"High-risk class {label}: weight {weights[i]:.2f} "
                           f"(base {base_weights[i]:.2f} x {high_risk_multiplier})")

        if high_risk_applied == 0:
            logger.warning("No high-risk classes found in data. "
                          "Check label_encoder or HIGH_RISK_CLASSES definition.")

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


@dataclass
class EnhancedImbalanceHandler:
    """
    Enhanced imbalance handler with multiple strategies.

    Supports:
    - Class weights (standard and custom high-risk)
    - Effective number weighting
    - SMOTE/ADASYN resampling
    - Decision threshold optimization
    """

    strategy: str = "class_weights"  # "class_weights", "smote", "adasyn", "none"
    weight_method: str = "custom_high_risk"  # "balanced", "sqrt", "custom_high_risk", "effective_number"
    high_risk_multiplier: float = 3.0
    effective_number_beta: float = 0.999
    smote_k_neighbors: int = 3  # Reduced from 5 for small minority classes

    _class_weights: Optional[Dict[int, float]] = field(default=None, init=False)
    _label_encoder: Optional[Dict[str, int]] = field(default=None, init=False)

    def fit(
        self,
        y: np.ndarray,
        label_encoder: Optional[Dict[str, int]] = None
    ) -> 'EnhancedImbalanceHandler':
        """
        Fit imbalance handler on target.

        Args:
            y: Target array
            label_encoder: Optional label to index mapping

        Returns:
            self
        """
        self._label_encoder = label_encoder or {}

        if self.strategy == "class_weights":
            self._class_weights = compute_enhanced_class_weights(
                y,
                self._label_encoder,
                strategy=self.weight_method,
                high_risk_multiplier=self.high_risk_multiplier,
                effective_number_beta=self.effective_number_beta
            )

            # Log class distribution and weights
            unique, counts = np.unique(y, return_counts=True)
            logger.info("Class distribution and weights:")
            for cls, cnt in zip(unique, counts):
                weight = self._class_weights.get(int(cls), 1.0)
                pct = 100 * cnt / len(y)
                logger.info(f"  Class {cls}: {cnt} samples ({pct:.1f}%), weight={weight:.2f}")

        return self

    def get_class_weights(self) -> Optional[Dict[int, float]]:
        """Get computed class weights."""
        return self._class_weights.copy() if self._class_weights else None

    def get_class_weights_list(self, n_classes: int) -> List[float]:
        """Get class weights as ordered list for PyTorch/TensorFlow."""
        if self._class_weights is None:
            return [1.0] * n_classes
        return [self._class_weights.get(i, 1.0) for i in range(n_classes)]

    def get_sample_weights(self, y: np.ndarray) -> Optional[np.ndarray]:
        """Get per-sample weights for training."""
        if self._class_weights is None:
            return None
        return compute_sample_weights(y, self._class_weights)

    def resample(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample training data if using SMOTE/ADASYN.

        Args:
            X: Features
            y: Targets

        Returns:
            Resampled (X, y)
        """
        if self.strategy not in ["smote", "adasyn"]:
            return X, y

        try:
            if self.strategy == "smote":
                from imblearn.over_sampling import SMOTE
                sampler = SMOTE(
                    k_neighbors=self.smote_k_neighbors,
                    random_state=42
                )
            else:  # adasyn
                from imblearn.over_sampling import ADASYN
                sampler = ADASYN(
                    n_neighbors=self.smote_k_neighbors,
                    random_state=42
                )

            X_resampled, y_resampled = sampler.fit_resample(X, y)

            logger.info(
                f"{self.strategy.upper()} resampling: {len(y)} -> {len(y_resampled)} samples"
            )

            # Log new distribution
            unique, counts = np.unique(y_resampled, return_counts=True)
            for cls, cnt in zip(unique, counts):
                logger.info(f"  Class {cls}: {cnt} samples")

            return X_resampled, y_resampled

        except ImportError:
            logger.warning("imbalanced-learn not installed. Skipping resampling.")
            return X, y
        except Exception as e:
            logger.warning(f"Resampling failed: {e}. Using original data.")
            return X, y


# =============================================================================
# FOCAL LOSS IMPLEMENTATION FOR PYTORCH
# =============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class FocalLoss(nn.Module):
        """
        Focal Loss for addressing class imbalance in classification.

        From "Focal Loss for Dense Object Detection" (Lin et al., 2017)

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

        where p_t is the probability of the correct class.

        Args:
            alpha: Class weights (tensor or None)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """

        def __init__(
            self,
            alpha: Optional[torch.Tensor] = None,
            gamma: float = 2.0,
            reduction: str = 'mean'
        ):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            """
            Compute focal loss.

            Args:
                inputs: Model outputs (logits), shape (N, C)
                targets: Ground truth labels, shape (N,)

            Returns:
                Focal loss value
            """
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            p_t = torch.exp(-ce_loss)

            # Focal term
            focal_term = (1 - p_t) ** self.gamma

            # Apply class weights if provided
            if self.alpha is not None:
                alpha_t = self.alpha[targets]
                focal_loss = alpha_t * focal_term * ce_loss
            else:
                focal_loss = focal_term * ce_loss

            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss


    class ClassBalancedLoss(nn.Module):
        """
        Class-Balanced Loss based on effective number of samples.

        From "Class-Balanced Loss Based on Effective Number of Samples" (Cui et al., 2019)

        Args:
            samples_per_class: Number of samples for each class
            n_classes: Total number of classes
            loss_type: 'focal', 'cross_entropy', or 'sigmoid'
            beta: Hyperparameter for effective number calculation
            gamma: Focal loss gamma (only used if loss_type='focal')
        """

        def __init__(
            self,
            samples_per_class: List[int],
            n_classes: int,
            loss_type: str = 'focal',
            beta: float = 0.999,
            gamma: float = 2.0
        ):
            super().__init__()

            # Compute effective number of samples
            effective_num = 1.0 - np.power(beta, samples_per_class)
            weights = (1.0 - beta) / (effective_num + 1e-8)
            weights = weights / np.sum(weights) * n_classes

            self.weights = torch.tensor(weights, dtype=torch.float32)
            self.n_classes = n_classes
            self.loss_type = loss_type
            self.gamma = gamma

        def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            """Compute class-balanced loss."""
            device = inputs.device
            weights = self.weights.to(device)

            if self.loss_type == 'focal':
                # Focal loss with class-balanced weights
                ce_loss = F.cross_entropy(inputs, targets, reduction='none')
                p_t = torch.exp(-ce_loss)
                focal_term = (1 - p_t) ** self.gamma
                cb_loss = weights[targets] * focal_term * ce_loss
                return cb_loss.mean()

            elif self.loss_type == 'cross_entropy':
                return F.cross_entropy(inputs, targets, weight=weights)

            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")


    def create_loss_function(
        class_weights: Optional[Dict[int, float]] = None,
        loss_type: str = 'focal',
        gamma: float = 2.0,
        n_classes: int = 9,
        samples_per_class: Optional[List[int]] = None,
        device: str = 'cpu'
    ) -> nn.Module:
        """
        Create appropriate loss function for imbalanced classification.

        Args:
            class_weights: Optional class weight dictionary
            loss_type: 'focal', 'cross_entropy', 'class_balanced'
            gamma: Focal loss gamma
            n_classes: Number of classes
            samples_per_class: Samples per class (for class-balanced loss)
            device: Device to use

        Returns:
            Loss function module
        """
        if loss_type == 'focal':
            if class_weights is not None:
                weights = torch.tensor(
                    [class_weights.get(i, 1.0) for i in range(n_classes)],
                    dtype=torch.float32
                ).to(device)
            else:
                weights = None
            return FocalLoss(alpha=weights, gamma=gamma)

        elif loss_type == 'class_balanced' and samples_per_class is not None:
            return ClassBalancedLoss(
                samples_per_class=samples_per_class,
                n_classes=n_classes,
                loss_type='focal',
                gamma=gamma
            ).to(device)

        elif loss_type == 'cross_entropy':
            if class_weights is not None:
                weights = torch.tensor(
                    [class_weights.get(i, 1.0) for i in range(n_classes)],
                    dtype=torch.float32
                ).to(device)
                return nn.CrossEntropyLoss(weight=weights)
            return nn.CrossEntropyLoss()

        else:
            return nn.CrossEntropyLoss()


# =============================================================================
# DECISION THRESHOLD OPTIMIZATION
# =============================================================================

def optimize_threshold_for_recall(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    high_risk_indices: List[int],
    target_recall: float = 0.8,
    n_thresholds: int = 100,
    calibrated_proba: Optional[np.ndarray] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Optimize decision threshold to achieve target recall for high-risk classes.

    IMPROVED: Better fallback when target recall not achievable.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities (N, n_classes)
        high_risk_indices: Indices of high-risk classes
        target_recall: Minimum target recall for high-risk classes
        n_thresholds: Number of thresholds to try
        calibrated_proba: Optional calibrated probabilities (recommended)

    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold)
    """
    # Use calibrated probabilities if provided
    proba = calibrated_proba if calibrated_proba is not None else y_proba

    # Compute high-risk probability (sum of high-risk class probabilities)
    high_risk_proba = proba[:, high_risk_indices].sum(axis=1)

    # True high-risk labels
    y_high_risk = np.isin(y_true, high_risk_indices)
    n_true_high_risk = y_high_risk.sum()

    if n_true_high_risk == 0:
        logger.warning("No high-risk samples in y_true. Cannot optimize threshold.")
        return 0.5, {'threshold': 0.5, 'recall': 0, 'precision': 0, 'f1': 0, 'warning': 'no_high_risk_samples'}

    # Track all threshold results for fallback
    all_results = []
    best_threshold_target = None
    best_f1_target = 0
    best_metrics_target = {}

    max_recall_threshold = 0.5
    max_recall_value = 0
    max_recall_metrics = {}

    for threshold in np.linspace(0.01, 0.99, n_thresholds):
        y_pred_high_risk = high_risk_proba >= threshold

        # Compute metrics
        tp = ((y_pred_high_risk) & (y_high_risk)).sum()
        fp = ((y_pred_high_risk) & (~y_high_risk)).sum()
        fn = ((~y_pred_high_risk) & (y_high_risk)).sum()
        tn = ((~y_pred_high_risk) & (~y_high_risk)).sum()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics = {
            'threshold': float(threshold),
            'recall': float(recall),
            'precision': float(precision),
            'f1': float(f1),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        }
        all_results.append(metrics)

        # Track best threshold meeting target recall
        if recall >= target_recall and f1 > best_f1_target:
            best_f1_target = f1
            best_threshold_target = threshold
            best_metrics_target = metrics.copy()

        # Track maximum recall achieved (for fallback)
        if recall > max_recall_value or (recall == max_recall_value and f1 > max_recall_metrics.get('f1', 0)):
            max_recall_value = recall
            max_recall_threshold = threshold
            max_recall_metrics = metrics.copy()

    # Determine final threshold
    if best_threshold_target is not None:
        best_threshold = best_threshold_target
        best_metrics = best_metrics_target
        best_metrics['target_achieved'] = True
        logger.info(f"Target recall {target_recall:.2f} achieved at threshold {best_threshold:.3f}")
    else:
        # IMPROVED FALLBACK: Use threshold that maximizes recall (not arbitrary 0.1)
        best_threshold = max_recall_threshold
        best_metrics = max_recall_metrics
        best_metrics['target_achieved'] = False
        best_metrics['max_possible_recall'] = max_recall_value
        logger.warning(f"Could not achieve target recall {target_recall:.2f}. "
                      f"Best achievable recall: {max_recall_value:.3f} at threshold {best_threshold:.3f}")

    logger.info(f"Optimized threshold: {best_threshold:.3f} "
               f"(recall={best_metrics.get('recall', 0):.3f}, "
               f"precision={best_metrics.get('precision', 0):.3f}, "
               f"f1={best_metrics.get('f1', 0):.3f})")

    return best_threshold, best_metrics


def apply_threshold_adjustment(
    y_proba: np.ndarray,
    high_risk_indices: List[int],
    threshold: float = 0.3,
    default_prediction_fn: Optional[callable] = None
) -> np.ndarray:
    """
    Apply adjusted threshold for high-risk class prediction.

    If sum of high-risk class probabilities exceeds threshold,
    predict the most likely high-risk class.

    This implements a "risk escalation rule" - a post-hoc policy that
    overrides standard argmax predictions when high-risk probability
    exceeds threshold.

    Args:
        y_proba: Predicted probabilities (N, n_classes)
        high_risk_indices: Indices of high-risk classes
        threshold: Threshold for high-risk prediction
        default_prediction_fn: Optional function to get default predictions
                               If None, uses argmax

    Returns:
        Adjusted predictions
    """
    n_samples = y_proba.shape[0]

    # FIXED: Use default_prediction_fn if provided
    if default_prediction_fn is not None:
        y_pred = default_prediction_fn(y_proba)
    else:
        y_pred = np.argmax(y_proba, axis=1)

    # Compute high-risk probability
    high_risk_proba = y_proba[:, high_risk_indices].sum(axis=1)

    # Count adjustments for logging
    n_adjusted = 0

    # Adjust predictions
    for i in range(n_samples):
        if high_risk_proba[i] >= threshold:
            # Predict most likely high-risk class
            high_risk_class_proba = y_proba[i, high_risk_indices]
            best_high_risk_idx = np.argmax(high_risk_class_proba)
            new_pred = high_risk_indices[best_high_risk_idx]

            if y_pred[i] != new_pred:
                n_adjusted += 1
                y_pred[i] = new_pred

    if n_adjusted > 0:
        logger.info(f"Threshold adjustment: {n_adjusted}/{n_samples} predictions "
                   f"escalated to high-risk (threshold={threshold:.3f})")

    return y_pred


# =============================================================================
# PROBABILITY CALIBRATION
# =============================================================================

def calibrate_probabilities(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    method: str = 'isotonic',
    cv: int = 3
) -> Tuple[np.ndarray, Any]:
    """
    Calibrate predicted probabilities for better threshold optimization.

    With imbalanced data, raw probabilities are often miscalibrated.
    Calibration improves the reliability of threshold-based decisions.

    Args:
        y_true: True labels (for fitting calibrator)
        y_proba: Predicted probabilities (N, n_classes)
        method: 'isotonic' or 'platt' (sigmoid)
        cv: Cross-validation folds for calibration

    Returns:
        Tuple of (calibrated_proba, calibrator)
    """
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.base import BaseEstimator, ClassifierMixin

        # Create a dummy classifier that just returns pre-computed probabilities
        class PrecomputedClassifier(BaseEstimator, ClassifierMixin):
            def __init__(self, proba):
                self.proba = proba
                self._idx = 0

            def fit(self, X, y):
                self.classes_ = np.unique(y)
                return self

            def predict_proba(self, X):
                # Return stored probabilities
                return self.proba

            def predict(self, X):
                return np.argmax(self.proba, axis=1)

        # For each class, calibrate using one-vs-rest
        n_classes = y_proba.shape[1]
        calibrated_proba = np.zeros_like(y_proba)

        for cls in range(n_classes):
            # Binary problem: class vs rest
            y_binary = (y_true == cls).astype(int)
            proba_cls = y_proba[:, cls]

            if y_binary.sum() < cv:
                # Not enough positive samples, use raw probabilities
                calibrated_proba[:, cls] = proba_cls
                continue

            try:
                from sklearn.isotonic import IsotonicRegression
                from sklearn.linear_model import LogisticRegression

                if method == 'isotonic':
                    calibrator = IsotonicRegression(out_of_bounds='clip')
                    calibrator.fit(proba_cls, y_binary)
                    calibrated_proba[:, cls] = calibrator.predict(proba_cls)
                else:  # platt/sigmoid
                    # Platt scaling
                    lr = LogisticRegression()
                    lr.fit(proba_cls.reshape(-1, 1), y_binary)
                    calibrated_proba[:, cls] = lr.predict_proba(proba_cls.reshape(-1, 1))[:, 1]

            except Exception as e:
                logger.warning(f"Calibration failed for class {cls}: {e}")
                calibrated_proba[:, cls] = proba_cls

        # Normalize to sum to 1
        row_sums = calibrated_proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
        calibrated_proba = calibrated_proba / row_sums

        logger.info(f"Probabilities calibrated using {method} method")
        return calibrated_proba, None

    except Exception as e:
        logger.warning(f"Probability calibration failed: {e}. Using raw probabilities.")
        return y_proba, None


# =============================================================================
# COMPREHENSIVE METRICS FOR IMBALANCED DATA
# =============================================================================

def compute_imbalance_aware_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    high_risk_indices: List[int],
    idx_to_label: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics suitable for imbalanced classification.

    Reports both multiclass metrics and binary high-risk detection metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        high_risk_indices: Indices of high-risk classes
        idx_to_label: Optional index to label mapping

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix, classification_report
    )

    metrics = {}

    # Standard multiclass metrics
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['macro_f1'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['weighted_f1'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    metrics['macro_precision'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['macro_recall'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))

    # Binary high-risk detection metrics
    y_true_hr = np.isin(y_true, high_risk_indices)
    y_pred_hr = np.isin(y_pred, high_risk_indices)

    if y_true_hr.sum() > 0:
        metrics['high_risk_recall'] = float((y_true_hr & y_pred_hr).sum() / y_true_hr.sum())
        metrics['high_risk_fnr'] = 1.0 - metrics['high_risk_recall']
    else:
        metrics['high_risk_recall'] = 0.0
        metrics['high_risk_fnr'] = 0.0

    if y_pred_hr.sum() > 0:
        metrics['high_risk_precision'] = float((y_true_hr & y_pred_hr).sum() / y_pred_hr.sum())
    else:
        metrics['high_risk_precision'] = 0.0

    if metrics['high_risk_precision'] + metrics['high_risk_recall'] > 0:
        metrics['high_risk_f1'] = 2 * metrics['high_risk_precision'] * metrics['high_risk_recall'] / \
                                  (metrics['high_risk_precision'] + metrics['high_risk_recall'])
    else:
        metrics['high_risk_f1'] = 0.0

    # Per-class recall for high-risk classes
    metrics['per_class_recall'] = {}
    for hr_idx in high_risk_indices:
        mask = y_true == hr_idx
        if mask.sum() > 0:
            recall = (y_pred[mask] == hr_idx).mean()
            label = idx_to_label.get(hr_idx, str(hr_idx)) if idx_to_label else str(hr_idx)
            metrics['per_class_recall'][label] = float(recall)

    # PR-AUC for high-risk detection (binary)
    try:
        from sklearn.metrics import average_precision_score
        high_risk_proba = y_proba[:, high_risk_indices].sum(axis=1)
        metrics['high_risk_pr_auc'] = float(average_precision_score(y_true_hr, high_risk_proba))
    except Exception:
        metrics['high_risk_pr_auc'] = 0.0

    # Counts
    metrics['n_true_high_risk'] = int(y_true_hr.sum())
    metrics['n_pred_high_risk'] = int(y_pred_hr.sum())
    metrics['n_samples'] = len(y_true)

    return metrics
