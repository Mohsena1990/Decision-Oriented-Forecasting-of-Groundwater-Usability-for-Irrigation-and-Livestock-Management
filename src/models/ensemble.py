"""
Weighted Ensemble Module
========================

Combines predictions from multiple fitted forecasters using
optimizable weights.  Weights are tuned on CV out-of-fold probabilities
to minimise Severe FNR while maintaining macro-F1 ≥ a floor threshold.

Usage
-----
    ensemble = WeightedEnsemble(models, high_risk_indices=high_risk_idx)
    ensemble.fit_weights(X_val, y_val)   # optional weight optimisation
    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


class WeightedEnsemble:
    """
    Soft-voting ensemble with optimisable per-model weights.

    Parameters
    ----------
    models : list
        Fitted model objects (must expose ``predict_proba(X)``).
    model_names : list of str, optional
        Display names aligned with ``models``.
    high_risk_indices : list of int
        Class indices considered high-risk (T3_Restricted).
    weights : array-like, optional
        Initial weights (will be normalised to sum to 1).
        Defaults to uniform weights.
    optimize_weights : bool
        If True, ``fit_weights()`` searches for weights that minimise
        a safety-aware loss (FNR-penalised, F1-floored).
    f1_floor : float
        Minimum acceptable macro-F1 during weight search.
    fnr_penalty : float
        Multiplier applied to Severe FNR in the composite loss.
    """

    def __init__(
        self,
        models: List[Any],
        model_names: Optional[List[str]] = None,
        high_risk_indices: Optional[List[int]] = None,
        weights: Optional[np.ndarray] = None,
        optimize_weights: bool = True,
        f1_floor: float = 0.40,
        fnr_penalty: float = 3.0,
    ):
        self.models = models
        self.model_names = model_names or [f"Model_{i}" for i in range(len(models))]
        self.high_risk_indices = high_risk_indices or []
        self.optimize_weights = optimize_weights
        self.f1_floor = f1_floor
        self.fnr_penalty = fnr_penalty

        n = len(models)
        if weights is not None:
            w = np.array(weights, dtype=float)
        else:
            w = np.ones(n, dtype=float)
        self.weights = w / w.sum()

    # ------------------------------------------------------------------
    # Weight optimisation
    # ------------------------------------------------------------------

    def fit_weights(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 500,
        random_state: int = 42,
        anchor_weights: Optional[np.ndarray] = None,
        anchor_strength: float = 0.7,
    ) -> "WeightedEnsemble":
        """
        Search for weights that minimise a safety-aware composite loss on
        the provided validation set.

        Loss = FNR * fnr_penalty + (1 − macro_F1)
        subject to macro_F1 ≥ f1_floor.

        anchor_weights : optional prior over the simplex (e.g. from VIKOR Q-scores).
            When supplied, 70% of trials sample near the anchor using a
            concentrated Dirichlet, and 30% explore uniformly.  This prevents
            the naive LightGBM-dominant solution that arises from purely random
            search when the model with best overall F1 happens to dominate on
            the small validation split.
        anchor_strength : Dirichlet concentration parameter for biased trials
            (higher = tighter around anchor).
        """
        if not self.optimize_weights:
            return self

        rng = np.random.default_rng(random_state)
        n = len(self.models)

        # Collect per-model probabilities once
        probas = [m.predict_proba(X_val) for m in self.models]

        best_loss = float("inf")
        best_weights = self.weights.copy()

        n_anchored = int(n_trials * anchor_strength) if anchor_weights is not None else 0
        n_uniform  = n_trials - n_anchored

        def _try_weights(w: np.ndarray) -> None:
            nonlocal best_loss, best_weights
            ensemble_proba = sum(w[i] * probas[i] for i in range(n))
            y_pred = np.argmax(ensemble_proba, axis=1)

            macro_f1 = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
            if macro_f1 < self.f1_floor:
                return

            fnr = self._compute_fnr(y_val, y_pred)
            loss = fnr * self.fnr_penalty + (1.0 - macro_f1)

            if loss < best_loss:
                best_loss = loss
                best_weights = w.copy()

        # Anchored trials: sample near VIKOR-derived weights
        if anchor_weights is not None and n_anchored > 0:
            alpha = np.maximum(anchor_weights, 1e-3) * anchor_strength * n
            for _ in range(n_anchored):
                w = rng.dirichlet(alpha)
                _try_weights(w)

        # Uniform exploration trials
        for _ in range(n_uniform):
            w = rng.dirichlet(np.ones(n))
            _try_weights(w)

        self.weights = best_weights
        logger.info(
            "Ensemble weights optimised: %s → loss=%.4f",
            dict(zip(self.model_names, self.weights.round(3))),
            best_loss,
        )
        return self

    # ------------------------------------------------------------------
    # Prediction interface
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return weighted average of model probability outputs."""
        combined = None
        for i, model in enumerate(self.models):
            proba = model.predict_proba(X)
            if combined is None:
                combined = self.weights[i] * proba
            else:
                combined += self.weights[i] * proba
        return combined

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return argmax of weighted ensemble probabilities."""
        return np.argmax(self.predict_proba(X), axis=1)

    def get_model_complexity(self) -> Dict[str, Any]:
        """Complexity is the sum of member complexities."""
        total_params = 0
        for m in self.models:
            c = m.get_model_complexity()
            total_params += c.get("n_params", 0)
        return {"n_params": total_params, "n_members": len(self.models)}

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def weight_summary(self) -> Dict[str, float]:
        return dict(zip(self.model_names, self.weights.tolist()))

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _compute_fnr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if not self.high_risk_indices:
            return 0.0
        true_hr = np.isin(y_true, self.high_risk_indices)
        n_true = true_hr.sum()
        if n_true == 0:
            return 0.0
        fn = np.sum(true_hr & ~np.isin(y_pred, self.high_risk_indices))
        return float(fn / n_true)
