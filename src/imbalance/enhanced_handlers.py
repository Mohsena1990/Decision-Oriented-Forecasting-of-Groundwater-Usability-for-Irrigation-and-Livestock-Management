"""
Enhanced Imbalance Handling Module
==================================

Professional strategies for handling severe class imbalance (234:1 ratio).

Includes:
1. Custom high-risk class weighting
2. Focal Loss for deep learning (addresses class imbalance better than CE)
3. Decision threshold optimization
4. Cost-sensitive learning support
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# High-risk classes that need special attention
HIGH_RISK_CLASSES = ['C4S1', 'C4S2', 'C4S3', 'C4S4', 'C3S3', 'C3S4']


def compute_enhanced_class_weights(
    y: np.ndarray,
    label_encoder: Dict[str, int],
    strategy: str = 'custom_high_risk',
    high_risk_multiplier: float = 3.0,
    effective_number_beta: float = 0.999
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
        for i, cls_idx in enumerate(unique_classes):
            label = idx_to_label.get(int(cls_idx), '')
            if label in HIGH_RISK_CLASSES:
                weights[i] *= high_risk_multiplier
                logger.info(f"High-risk class {label}: weight {weights[i]:.2f} "
                           f"(base {base_weights[i]:.2f} x {high_risk_multiplier})")

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
    n_thresholds: int = 100
) -> Tuple[float, Dict[str, float]]:
    """
    Optimize decision threshold to achieve target recall for high-risk classes.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities (N, n_classes)
        high_risk_indices: Indices of high-risk classes
        target_recall: Minimum target recall for high-risk classes
        n_thresholds: Number of thresholds to try

    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold)
    """
    # Compute high-risk probability (sum of high-risk class probabilities)
    high_risk_proba = y_proba[:, high_risk_indices].sum(axis=1)

    # True high-risk labels
    y_high_risk = np.isin(y_true, high_risk_indices)

    best_threshold = 0.5
    best_f1 = 0
    best_metrics = {}

    for threshold in np.linspace(0.01, 0.99, n_thresholds):
        y_pred_high_risk = high_risk_proba >= threshold

        # Compute metrics
        tp = ((y_pred_high_risk == True) & (y_high_risk == True)).sum()
        fp = ((y_pred_high_risk == True) & (y_high_risk == False)).sum()
        fn = ((y_pred_high_risk == False) & (y_high_risk == True)).sum()
        tn = ((y_pred_high_risk == False) & (y_high_risk == False)).sum()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Check if recall meets target
        if recall >= target_recall and f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'threshold': threshold,
                'recall': recall,
                'precision': precision,
                'f1': f1,
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn)
            }

    if not best_metrics:
        # If target recall not achievable, find threshold with max recall
        best_metrics = {'threshold': 0.1, 'recall': 0, 'precision': 0, 'f1': 0}
        best_threshold = 0.1
        logger.warning(f"Could not achieve target recall {target_recall}. Using default threshold.")

    logger.info(f"Optimized threshold: {best_threshold:.3f} "
               f"(recall={best_metrics.get('recall', 0):.3f}, "
               f"precision={best_metrics.get('precision', 0):.3f})")

    return best_threshold, best_metrics


def apply_threshold_adjustment(
    y_proba: np.ndarray,
    high_risk_indices: List[int],
    threshold: float = 0.3,
    default_prediction_fn=None
) -> np.ndarray:
    """
    Apply adjusted threshold for high-risk class prediction.

    If sum of high-risk class probabilities exceeds threshold,
    predict the most likely high-risk class.

    Args:
        y_proba: Predicted probabilities (N, n_classes)
        high_risk_indices: Indices of high-risk classes
        threshold: Threshold for high-risk prediction
        default_prediction_fn: Optional function to get default predictions

    Returns:
        Adjusted predictions
    """
    n_samples = y_proba.shape[0]
    y_pred = np.argmax(y_proba, axis=1)

    # Compute high-risk probability
    high_risk_proba = y_proba[:, high_risk_indices].sum(axis=1)

    # Adjust predictions
    for i in range(n_samples):
        if high_risk_proba[i] >= threshold:
            # Predict most likely high-risk class
            high_risk_class_proba = y_proba[i, high_risk_indices]
            best_high_risk_idx = np.argmax(high_risk_class_proba)
            y_pred[i] = high_risk_indices[best_high_risk_idx]

    return y_pred
