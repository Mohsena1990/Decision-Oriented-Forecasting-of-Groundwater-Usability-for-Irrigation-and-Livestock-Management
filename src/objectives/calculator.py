"""
Objective Calculator Module
===========================

Computes multi-objective values for model evaluation and optimization.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


@dataclass
class ObjectiveCalculator:
    """
    Calculate multi-objective values for groundwater quality forecasting.

    Objectives (all to be minimized):
    1. Ordinal distance error
    2. Severe-risk false negative rate
    3. 1 - MacroF1
    4. Complexity (number of features)
    """

    # Label information
    label_to_ordinal: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    idx_to_label: Dict[int, str] = field(default_factory=dict)
    high_risk_indices: List[int] = field(default_factory=list)

    # Ordinal weights
    c_weight: float = 1.0
    s_weight: float = 1.0

    # Objective weights (for weighted sum if needed)
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        'ordinal_distance': 1.0,
        'severe_fnr': 1.5,
        'macro_f1_complement': 1.0,
        'complexity': 0.5
    })

    def compute_objectives(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        n_features: int = 0,
        model_complexity: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Compute all objective values.

        Args:
            y_true: True label indices
            y_pred: Predicted label indices
            y_proba: Predicted probabilities (optional)
            n_features: Number of selected features
            model_complexity: Optional model complexity metrics

        Returns:
            Array of objective values (all to minimize)
        """
        objectives = []

        # O1: Ordinal distance (minimize)
        ord_dist = self._compute_ordinal_distance(y_true, y_pred)
        objectives.append(ord_dist)

        # O2: Severe-risk FNR (minimize)
        severe_fnr = self._compute_severe_fnr(y_true, y_pred)
        objectives.append(severe_fnr)

        # O3: 1 - MacroF1 (minimize)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        objectives.append(1.0 - macro_f1)

        # O4: Complexity (minimize)
        complexity = self._compute_complexity(n_features, model_complexity)
        objectives.append(complexity)

        return np.array(objectives)

    def _compute_ordinal_distance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """Compute mean ordinal distance."""
        if not self.label_to_ordinal or not self.idx_to_label:
            # Fallback to simple error rate
            return 1.0 - np.mean(y_true == y_pred)

        distances = []
        for true_idx, pred_idx in zip(y_true, y_pred):
            true_label = self.idx_to_label.get(int(true_idx))
            pred_label = self.idx_to_label.get(int(pred_idx))

            if true_label and pred_label:
                true_ord = self.label_to_ordinal.get(true_label)
                pred_ord = self.label_to_ordinal.get(pred_label)

                if true_ord and pred_ord:
                    dist = self.c_weight * abs(true_ord[0] - pred_ord[0]) + \
                           self.s_weight * abs(true_ord[1] - pred_ord[1])
                    distances.append(dist)
                else:
                    # Different labels, max penalty
                    distances.append(self.c_weight * 3 + self.s_weight * 3)
            else:
                distances.append(self.c_weight * 3 + self.s_weight * 3)

        return float(np.mean(distances)) if distances else 0.0

    def _compute_severe_fnr(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """Compute false negative rate for high-risk classes."""
        if not self.high_risk_indices:
            return 0.0

        true_high_risk = np.isin(y_true, self.high_risk_indices)
        n_true_high_risk = np.sum(true_high_risk)

        if n_true_high_risk == 0:
            return 0.0

        pred_high_risk = np.isin(y_pred, self.high_risk_indices)
        false_negatives = np.sum(true_high_risk & ~pred_high_risk)

        return false_negatives / n_true_high_risk

    def _compute_complexity(
        self,
        n_features: int,
        model_complexity: Optional[Dict[str, Any]]
    ) -> float:
        """Compute normalized complexity score."""
        # Primary: number of features (normalized by max expected)
        max_features = 20  # Adjust based on your feature set
        feature_complexity = n_features / max_features

        # Optional: add model-specific complexity
        if model_complexity:
            # For trees: consider depth or number of trees
            if 'n_trees' in model_complexity:
                tree_complexity = model_complexity['n_trees'] / 1000
                feature_complexity = 0.7 * feature_complexity + 0.3 * tree_complexity

            # For neural networks: consider number of parameters
            if 'n_params' in model_complexity:
                param_complexity = np.log10(model_complexity['n_params'] + 1) / 6  # Normalize
                feature_complexity = 0.7 * feature_complexity + 0.3 * param_complexity

        return feature_complexity

    def compute_weighted_sum(self, objectives: np.ndarray) -> float:
        """Compute weighted sum of objectives."""
        weights = np.array([
            self.objective_weights.get('ordinal_distance', 1.0),
            self.objective_weights.get('severe_fnr', 1.5),
            self.objective_weights.get('macro_f1_complement', 1.0),
            self.objective_weights.get('complexity', 0.5)
        ])

        # Ensure proper length
        weights = weights[:len(objectives)]

        return float(np.sum(weights * objectives))

    def get_objective_names(self) -> List[str]:
        """Get names of objectives."""
        return ['ordinal_distance', 'severe_fnr', 'macro_f1_complement', 'complexity']


def create_objective_function(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_factory,
    calculator: ObjectiveCalculator,
    cv_splitter,
    param_decoder
):
    """
    Create objective function for optimization.

    Returns a function that takes a position vector and returns objective values.
    """
    def objective_fn(position: np.ndarray) -> np.ndarray:
        # Decode position into parameters and feature mask
        params, feature_mask = param_decoder(position)

        # Apply feature mask
        X_masked = X_train[:, feature_mask.astype(bool)] if feature_mask.sum() > 0 else X_train
        n_features = int(feature_mask.sum()) if feature_mask.sum() > 0 else X_train.shape[1]

        # Cross-validation evaluation
        all_objectives = []

        for train_idx, val_idx in cv_splitter:
            X_tr, X_val = X_masked[train_idx], X_masked[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            # Create and train model
            model = model_factory(params)
            model.fit(X_tr, y_tr)

            # Predict
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)

            # Compute objectives
            objectives = calculator.compute_objectives(
                y_val, y_pred, y_proba,
                n_features=n_features,
                model_complexity=model.get_model_complexity()
            )
            all_objectives.append(objectives)

        # Average across folds
        return np.mean(all_objectives, axis=0)

    return objective_fn
