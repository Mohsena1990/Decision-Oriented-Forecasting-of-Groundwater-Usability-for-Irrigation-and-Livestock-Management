"""
Enhanced Filter-based Feature Selection
=======================================

Professional filter selection with:
1. Correct indexing (feature_idx column)
2. Fold-aware fitting to prevent data leakage
3. Proper NaN handling (median imputation, not zeros)
4. Stability via rank aggregation
5. Permutation importance option
6. Reproducibility via random_state
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


@dataclass
class EnhancedFilterSelector:
    """
    Enhanced filter-based feature selection with professional practices.

    Key improvements over basic FilterSelector:
    - Stores feature_idx explicitly (fixes indexing bug)
    - Fold-aware fitting to prevent data leakage
    - Median imputation instead of nan→0
    - Stability via multiple runs / rank aggregation
    - Permutation importance option for unbiased ranking
    - Reproducibility via random_state
    """

    method: str = "mutual_info"  # "mutual_info", "anova", "tree_importance", "permutation"
    top_k: int = 15
    n_runs: int = 5  # For stability (average across runs)
    use_cv_aggregation: bool = True  # Fit within CV folds and aggregate
    n_cv_folds: int = 3
    random_state: int = 42

    # Internal state
    _rankings: Optional[pd.DataFrame] = field(default=None, init=False)
    _imputer: Optional[SimpleImputer] = field(default=None, init=False)
    _rng: Optional[np.random.Generator] = field(default=None, init=False)
    _fold_rankings: Optional[List[pd.DataFrame]] = field(default=None, init=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.random_state)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> 'EnhancedFilterSelector':
        """
        Rank features using filter method with proper practices.

        Args:
            X: Features (n_samples, n_features)
            y: Targets (n_samples,)
            feature_names: Optional feature names

        Returns:
            self
        """
        n_samples, n_features = X.shape

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(n_features)]

        # Fit imputer on full data (for handling NaN during scoring)
        self._imputer = SimpleImputer(strategy='median')
        self._imputer.fit(X)
        X_imputed = self._imputer.transform(X)

        if self.use_cv_aggregation:
            # Fold-aware ranking to prevent data leakage
            scores = self._fit_with_cv_aggregation(X_imputed, y)
        else:
            # Simple fit with stability (multiple runs)
            scores = self._fit_with_stability(X_imputed, y)

        # Build rankings DataFrame with explicit feature_idx
        self._rankings = pd.DataFrame({
            'feature_idx': np.arange(n_features),  # CRITICAL FIX
            'feature': feature_names,
            'score': scores
        }).sort_values('score', ascending=False).reset_index(drop=True)

        # Add rank column after sorting
        self._rankings['rank'] = np.arange(1, len(self._rankings) + 1)

        logger.info(f"Feature rankings computed using {self.method}")
        logger.info(f"Top 5 features: {self._rankings.head(5)['feature'].tolist()}")

        return self

    def _fit_with_cv_aggregation(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """
        Fit filter method within CV folds and aggregate ranks.

        This prevents data leakage by not using test labels during selection.
        """
        n_features = X.shape[1]
        fold_scores = []
        self._fold_rankings = []

        cv = StratifiedKFold(
            n_splits=self.n_cv_folds,
            shuffle=True,
            random_state=self.random_state
        )

        for fold_idx, (train_idx, _) in enumerate(cv.split(X, y)):
            X_fold = X[train_idx]
            y_fold = y[train_idx]

            # Compute scores on training fold only
            if self.method == "mutual_info":
                scores = self._mutual_info_scores(X_fold, y_fold)
            elif self.method == "anova":
                scores = self._anova_scores(X_fold, y_fold)
            elif self.method == "tree_importance":
                scores = self._tree_importance_scores(X_fold, y_fold)
            elif self.method == "permutation":
                scores = self._permutation_importance_scores(X_fold, y_fold)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            fold_scores.append(scores)

            # Store fold rankings for stability analysis
            fold_df = pd.DataFrame({
                'feature_idx': np.arange(n_features),
                'score': scores,
                'rank': np.argsort(np.argsort(-scores)) + 1
            })
            self._fold_rankings.append(fold_df)

        # Aggregate across folds (mean rank aggregation)
        all_ranks = np.array([
            np.argsort(np.argsort(-s)) + 1 for s in fold_scores
        ])
        mean_ranks = np.mean(all_ranks, axis=0)

        # Convert mean rank back to score (inverse so lower rank = higher score)
        aggregated_scores = 1.0 / mean_ranks

        return aggregated_scores

    def _fit_with_stability(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """
        Fit filter method multiple times and average for stability.
        """
        all_scores = []

        for run in range(self.n_runs):
            seed = self.random_state + run

            if self.method == "mutual_info":
                scores = self._mutual_info_scores(X, y, random_state=seed)
            elif self.method == "anova":
                scores = self._anova_scores(X, y)
            elif self.method == "tree_importance":
                scores = self._tree_importance_scores(X, y, random_state=seed)
            elif self.method == "permutation":
                scores = self._permutation_importance_scores(X, y, random_state=seed)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            all_scores.append(scores)

        # Average scores across runs
        return np.mean(all_scores, axis=0)

    def _mutual_info_scores(
        self,
        X: np.ndarray,
        y: np.ndarray,
        random_state: Optional[int] = None
    ) -> np.ndarray:
        """Compute mutual information scores."""
        from sklearn.feature_selection import mutual_info_classif

        seed = random_state if random_state is not None else self.random_state
        scores = mutual_info_classif(X, y, random_state=seed)
        return scores

    def _anova_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute ANOVA F-scores."""
        from sklearn.feature_selection import f_classif

        scores, _ = f_classif(X, y)
        # Replace NaN/inf with 0 (can happen if feature has zero variance)
        scores = np.nan_to_num(scores, nan=0, posinf=0, neginf=0)
        return scores

    def _tree_importance_scores(
        self,
        X: np.ndarray,
        y: np.ndarray,
        random_state: Optional[int] = None
    ) -> np.ndarray:
        """Compute tree-based importance scores."""
        from sklearn.ensemble import RandomForestClassifier

        seed = random_state if random_state is not None else self.random_state

        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=seed,
            n_jobs=-1
        )
        rf.fit(X, y)

        return rf.feature_importances_

    def _permutation_importance_scores(
        self,
        X: np.ndarray,
        y: np.ndarray,
        random_state: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute permutation importance (unbiased, model-agnostic).

        Better than RF impurity importance for:
        - Handling correlated features
        - Avoiding bias toward high-cardinality features
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.inspection import permutation_importance

        seed = random_state if random_state is not None else self.random_state

        # Use a simple RF as the base model
        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            random_state=seed,
            n_jobs=-1
        )
        rf.fit(X, y)

        # Compute permutation importance
        result = permutation_importance(
            rf, X, y,
            n_repeats=10,
            random_state=seed,
            n_jobs=-1
        )

        return result.importances_mean

    def get_top_features(self, k: Optional[int] = None) -> List[str]:
        """Get top k feature names."""
        if self._rankings is None:
            raise ValueError("Not fitted. Call fit() first.")

        k = k or self.top_k
        return self._rankings.head(k)['feature'].tolist()

    def get_top_indices(self, k: Optional[int] = None) -> np.ndarray:
        """Get top k feature indices (original column indices)."""
        if self._rankings is None:
            raise ValueError("Not fitted. Call fit() first.")

        k = k or self.top_k
        return self._rankings.head(k)['feature_idx'].to_numpy()

    def get_warm_start_mask(self, n_features: int, k: Optional[int] = None) -> np.ndarray:
        """
        Get binary mask for warm starting optimization.

        FIXED: Uses feature_idx column instead of DataFrame index.
        """
        if self._rankings is None:
            raise ValueError("Not fitted. Call fit() first.")

        k = k or self.top_k
        mask = np.zeros(n_features)

        # CRITICAL FIX: Use feature_idx column, not DataFrame index
        top_indices = self._rankings.head(k)['feature_idx'].to_numpy()
        mask[top_indices] = 1

        return mask

    def get_warm_start_population(
        self,
        n_features: int,
        n_individuals: int,
        k: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate warm start population for optimization.

        FIXED: Uses feature_idx column and seeded random generation.
        """
        if self._rankings is None:
            raise ValueError("Not fitted. Call fit() first.")

        k = k or self.top_k
        population = np.zeros((n_individuals, n_features))

        # CRITICAL FIX: Use feature_idx column
        top_indices = self._rankings.head(k)['feature_idx'].to_numpy().tolist()

        for i in range(n_individuals):
            # Add some variation to each individual
            n_base = max(3, k - i)
            selected = set(top_indices[:n_base])

            # Add some random features (with seeded RNG for reproducibility)
            remaining = set(range(n_features)) - selected
            n_random = min(i, len(remaining))

            if n_random > 0 and remaining:
                # FIXED: Use seeded RNG instead of np.random.choice
                random_additions = self._rng.choice(
                    list(remaining), size=n_random, replace=False
                )
                selected.update(random_additions)

            for idx in selected:
                population[i, idx] = 1

        return population

    def get_rankings(self) -> pd.DataFrame:
        """Get full feature rankings."""
        return self._rankings.copy() if self._rankings is not None else pd.DataFrame()

    def get_selection_stability(self, k: Optional[int] = None) -> Dict[str, float]:
        """
        Compute selection stability across CV folds.

        Returns metrics showing how consistently features are selected.
        """
        if self._fold_rankings is None:
            return {'stability': 1.0, 'warning': 'No CV aggregation performed'}

        k = k or self.top_k

        # Count how many times each feature appears in top-k across folds
        top_k_counts = np.zeros(len(self._fold_rankings[0]))

        for fold_df in self._fold_rankings:
            top_k_indices = fold_df.nsmallest(k, 'rank')['feature_idx'].values
            top_k_counts[top_k_indices] += 1

        # Stability = average consistency
        max_count = len(self._fold_rankings)
        stability = np.mean(top_k_counts[top_k_counts > 0]) / max_count

        # Identify stable features (appear in top-k in all folds)
        stable_features = np.where(top_k_counts == max_count)[0]

        return {
            'stability': float(stability),
            'n_stable_features': len(stable_features),
            'stable_feature_indices': stable_features.tolist(),
            'top_k_counts': top_k_counts.tolist()
        }


@dataclass
class EnhancedFeatureSelector:
    """
    Enhanced wrapper-based feature selection with professional practices.

    Improvements:
    - Stronger parsimony penalty (convex)
    - Adaptive penalty weight
    - Maximum feature constraint
    - Decision-aware objective support
    """

    n_features: int = 0
    mode: str = "wrapper"
    min_features: int = 3
    max_features: Optional[int] = None  # Hard constraint
    parsimony_weight: float = 0.01
    parsimony_power: float = 2.0  # Convex penalty exponent
    adaptive_penalty: bool = True  # Increase penalty over iterations

    _selected_mask: Optional[np.ndarray] = field(default=None, init=False)
    _iteration: int = field(default=0, init=False)
    _max_iterations: int = field(default=50, init=False)

    def initialize_mask(self, n_features: int) -> np.ndarray:
        """Initialize feature mask for optimization."""
        self.n_features = n_features
        if self.max_features is None:
            self.max_features = n_features  # No constraint by default
        return np.ones(n_features)

    def set_iteration(self, iteration: int, max_iterations: int):
        """Update iteration for adaptive penalty."""
        self._iteration = iteration
        self._max_iterations = max_iterations

    def decode_mask(
        self,
        position: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Decode continuous position to binary mask.

        Enforces min/max feature constraints.
        """
        mask = (position > threshold).astype(int)
        n_selected = np.sum(mask)

        # Ensure minimum features
        if n_selected < self.min_features:
            top_indices = np.argsort(position)[-self.min_features:]
            mask = np.zeros_like(mask)
            mask[top_indices] = 1

        # Enforce maximum features
        elif self.max_features and n_selected > self.max_features:
            top_indices = np.argsort(position)[-self.max_features:]
            mask = np.zeros_like(mask)
            mask[top_indices] = 1

        return mask

    def compute_parsimony_penalty(self, mask: np.ndarray) -> float:
        """
        Compute parsimony penalty with convex scaling.

        Convex penalty (power > 1) punishes "select almost all" more heavily.
        Adaptive penalty increases weight over optimization iterations.
        """
        n_selected = np.sum(mask)
        ratio = n_selected / self.n_features

        # Convex penalty
        base_penalty = self.parsimony_weight * (ratio ** self.parsimony_power)

        # Adaptive scaling (increases as optimization progresses)
        if self.adaptive_penalty and self._max_iterations > 0:
            progress = self._iteration / self._max_iterations
            adaptive_factor = 1.0 + progress  # Doubles by end
            return base_penalty * adaptive_factor

        return base_penalty

    def apply_mask(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply feature mask to data."""
        return X[:, mask.astype(bool)]

    def get_selected_indices(self, mask: np.ndarray) -> np.ndarray:
        """Get indices of selected features."""
        return np.where(mask.astype(bool))[0]

    def get_selected_names(
        self,
        mask: np.ndarray,
        feature_names: List[str]
    ) -> List[str]:
        """Get names of selected features."""
        indices = self.get_selected_indices(mask)
        return [feature_names[i] for i in indices if i < len(feature_names)]
