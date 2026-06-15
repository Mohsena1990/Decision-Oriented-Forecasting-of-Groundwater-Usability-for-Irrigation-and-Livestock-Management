"""
VIKOR Decision Making Module
============================

Implements VIKOR (VlseKriterijumska Optimizacija I Kompromisno Resenje)
method for multi-criteria decision making.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class VIKORResult:
    """Container for VIKOR results."""
    rankings: np.ndarray  # Indices sorted by Q (best to worst)
    Q: np.ndarray  # VIKOR Q values
    S: np.ndarray  # Group utility values
    R: np.ndarray  # Individual regret values
    best_idx: int  # Index of best alternative
    compromise_set: List[int]  # Indices of compromise solutions


@dataclass
class VIKOR:
    """
    VIKOR method for multi-criteria decision making.

    Selects the best compromise solution from a set of alternatives
    evaluated on multiple criteria.

    Attributes:
        v: Weight of group utility vs individual regret (default 0.5)
        weights: Weights for each criterion (will be normalized)
    """

    v: float = 0.5  # Balance between group utility and individual regret
    weights: Optional[np.ndarray] = None

    def rank(
        self,
        objectives: np.ndarray,
        weights: Optional[np.ndarray] = None,
        criteria_directions: Optional[List[str]] = None
    ) -> VIKORResult:
        """
        Rank alternatives using VIKOR method.

        Args:
            objectives: Matrix of objective values (n_alternatives, n_criteria)
            weights: Weights for each criterion (optional, uniform if not provided)
            criteria_directions: List of 'min' or 'max' for each criterion
                               (default: all 'min' for minimization)

        Returns:
            VIKORResult with rankings and scores
        """
        if objectives.ndim == 1:
            objectives = objectives.reshape(1, -1)
        if objectives.shape[0] == 0:
            raise ValueError("No alternatives to rank — objectives matrix is empty")
        n_alternatives, n_criteria = objectives.shape

        # Set weights
        if weights is not None:
            w = np.array(weights)
        elif self.weights is not None:
            w = self.weights
        else:
            w = np.ones(n_criteria)
        w = w / np.sum(w)  # Normalize

        # Set criteria directions (default: minimize all)
        if criteria_directions is None:
            criteria_directions = ['min'] * n_criteria

        # Normalize objectives matrix
        # Convert maximization criteria to minimization
        obj_normalized = objectives.copy()
        for j in range(n_criteria):
            if criteria_directions[j] == 'max':
                obj_normalized[:, j] = -obj_normalized[:, j]

        # Find best (f*) and worst (f-) values for each criterion
        f_best = np.min(obj_normalized, axis=0)
        f_worst = np.max(obj_normalized, axis=0)

        # Avoid division by zero
        ranges = f_worst - f_best
        ranges[ranges == 0] = 1e-10

        # Compute S (group utility) and R (individual regret)
        S = np.zeros(n_alternatives)
        R = np.zeros(n_alternatives)

        for i in range(n_alternatives):
            for j in range(n_criteria):
                normalized_gap = (f_best[j] - obj_normalized[i, j]) / ranges[j]
                # Note: for minimization, we want (f_worst - f_i) / range
                # Since we flipped max criteria, we need: (f_i - f_best) / range
                normalized_gap = (obj_normalized[i, j] - f_best[j]) / ranges[j]
                weighted_gap = w[j] * normalized_gap
                S[i] += weighted_gap
                R[i] = max(R[i], weighted_gap)

        # Compute Q (VIKOR index)
        S_best, S_worst = np.min(S), np.max(S)
        R_best, R_worst = np.min(R), np.max(R)

        # Avoid division by zero
        S_range = S_worst - S_best if S_worst != S_best else 1e-10
        R_range = R_worst - R_best if R_worst != R_best else 1e-10

        Q = self.v * (S - S_best) / S_range + (1 - self.v) * (R - R_best) / R_range

        # Rankings (lower Q is better)
        rankings = np.argsort(Q)
        best_idx = rankings[0]

        # Determine compromise set (alternatives within acceptable advantage)
        compromise_set = self._find_compromise_set(Q, rankings, n_alternatives)

        return VIKORResult(
            rankings=rankings,
            Q=Q,
            S=S,
            R=R,
            best_idx=best_idx,
            compromise_set=compromise_set
        )

    def _find_compromise_set(
        self,
        Q: np.ndarray,
        rankings: np.ndarray,
        n_alternatives: int
    ) -> List[int]:
        """Find the set of compromise solutions."""
        compromise_set = [rankings[0]]

        if n_alternatives < 2:
            return compromise_set

        # Condition 1: Acceptable advantage
        DQ = 1.0 / (n_alternatives - 1)
        Q_sorted = Q[rankings]

        for i in range(1, len(rankings)):
            if Q_sorted[i] - Q_sorted[0] < DQ:
                compromise_set.append(rankings[i])
            else:
                break

        return compromise_set

    def get_ranking_report(
        self,
        result: VIKORResult,
        alternative_names: Optional[List[str]] = None,
        criteria_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate a ranking report.

        Args:
            result: VIKORResult from rank()
            alternative_names: Optional names for alternatives
            criteria_names: Optional names for criteria

        Returns:
            DataFrame with rankings
        """
        n = len(result.Q)

        if alternative_names is None:
            alternative_names = [f'Alt_{i}' for i in range(n)]

        data = {
            'Alternative': alternative_names,
            'Rank': np.argsort(result.rankings) + 1,
            'Q': result.Q,
            'S': result.S,
            'R': result.R,
            'In_Compromise_Set': [i in result.compromise_set for i in range(n)]
        }

        df = pd.DataFrame(data)
        df = df.sort_values('Rank')

        return df


def normalize_objectives(
    objectives: np.ndarray,
    method: str = 'minmax'
) -> np.ndarray:
    """
    Normalize objective matrix.

    Args:
        objectives: Raw objective values
        method: 'minmax' or 'vector'

    Returns:
        Normalized objectives
    """
    if method == 'minmax':
        min_vals = np.min(objectives, axis=0)
        max_vals = np.max(objectives, axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1e-10
        return (objectives - min_vals) / ranges

    elif method == 'vector':
        norms = np.sqrt(np.sum(objectives ** 2, axis=0))
        norms[norms == 0] = 1e-10
        return objectives / norms

    else:
        raise ValueError(f"Unknown normalization method: {method}")
