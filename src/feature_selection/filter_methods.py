"""
Filter-based Feature Selection
==============================

Pre-selection methods for warm-starting optimization.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FilterSelector:
    """
    Filter-based feature selection for warm start.

    Ranks features using statistical methods to initialize
    optimization population near good solutions.
    """

    method: str = "mutual_info"  # "mutual_info", "anova", "tree_importance"
    top_k: int = 15

    _rankings: Optional[pd.DataFrame] = field(default=None, init=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> 'FilterSelector':
        """
        Rank features using filter method.

        Args:
            X: Features
            y: Targets
            feature_names: Feature names

        Returns:
            self
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        if self.method == "mutual_info":
            scores = self._mutual_info_scores(X, y)
        elif self.method == "anova":
            scores = self._anova_scores(X, y)
        elif self.method == "tree_importance":
            scores = self._tree_importance_scores(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self._rankings = pd.DataFrame({
            'feature': feature_names,
            'score': scores,
            'rank': np.argsort(np.argsort(-scores)) + 1
        }).sort_values('rank')

        logger.info(f"Feature rankings computed using {self.method}")
        return self

    def _mutual_info_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute mutual information scores."""
        from sklearn.feature_selection import mutual_info_classif

        # Handle NaN
        X_filled = np.nan_to_num(X, nan=0)

        scores = mutual_info_classif(X_filled, y, random_state=42)
        return scores

    def _anova_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute ANOVA F-scores."""
        from sklearn.feature_selection import f_classif

        X_filled = np.nan_to_num(X, nan=0)
        scores, _ = f_classif(X_filled, y)

        # Replace NaN/inf with 0
        scores = np.nan_to_num(scores, nan=0, posinf=0, neginf=0)
        return scores

    def _tree_importance_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute tree-based importance scores."""
        try:
            from sklearn.ensemble import RandomForestClassifier

            X_filled = np.nan_to_num(X, nan=0)

            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_filled, y)

            return rf.feature_importances_

        except Exception as e:
            logger.warning(f"Tree importance failed: {e}")
            return np.ones(X.shape[1])

    def get_top_features(self, k: Optional[int] = None) -> List[str]:
        """Get top k feature names."""
        if self._rankings is None:
            raise ValueError("Not fitted. Call fit() first.")

        k = k or self.top_k
        return self._rankings.head(k)['feature'].tolist()

    def get_warm_start_mask(self, n_features: int, k: Optional[int] = None) -> np.ndarray:
        """
        Get binary mask for warm starting optimization.

        Args:
            n_features: Total number of features
            k: Number of top features to select

        Returns:
            Binary mask
        """
        if self._rankings is None:
            raise ValueError("Not fitted. Call fit() first.")

        k = k or self.top_k
        mask = np.zeros(n_features)

        top_indices = self._rankings.head(k).index
        for idx in top_indices:
            if idx < n_features:
                mask[idx] = 1

        return mask

    def get_warm_start_population(
        self,
        n_features: int,
        n_individuals: int,
        k: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate warm start population for optimization.

        Args:
            n_features: Total number of features
            n_individuals: Number of individuals in population
            k: Number of top features as base

        Returns:
            Population array (n_individuals, n_features)
        """
        k = k or self.top_k
        population = np.zeros((n_individuals, n_features))

        top_indices = self._rankings.head(k).index.tolist()

        for i in range(n_individuals):
            # Add some variation to each individual
            selected = set(top_indices[:max(3, k - i)])

            # Add some random features
            remaining = set(range(n_features)) - selected
            n_random = min(i, len(remaining))
            if n_random > 0 and remaining:
                random_additions = np.random.choice(
                    list(remaining), n_random, replace=False
                )
                selected.update(random_additions)

            for idx in selected:
                if idx < n_features:
                    population[i, idx] = 1

        return population

    def get_rankings(self) -> pd.DataFrame:
        """Get full feature rankings."""
        return self._rankings.copy() if self._rankings is not None else pd.DataFrame()
