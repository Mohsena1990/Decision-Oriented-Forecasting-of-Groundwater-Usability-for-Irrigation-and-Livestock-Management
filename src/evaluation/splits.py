"""
Temporal Splitting Module
=========================

Implements temporal forward splitting for time-series forecasting
evaluation, ensuring no data leakage across time.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TemporalSplitter:
    """
    Temporal forward splitter for groundwater forecasting.

    Implements strict temporal splits where:
    - Training: earlier transitions (e.g., 2018->2019)
    - Testing: later transitions (e.g., 2019->2020)

    This ensures no information from the future leaks into training.
    """

    train_transitions: List[str] = field(default_factory=lambda: ["2018_2019"])
    test_transitions: List[str] = field(default_factory=lambda: ["2019_2020"])

    def split(
        self,
        transitions: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split transitions into train and test sets.

        Args:
            transitions: Dictionary mapping transition name to DataFrame

        Returns:
            Tuple of (train_df, test_df)
        """
        # Combine training transitions
        train_dfs = []
        for name in self.train_transitions:
            if name in transitions:
                df = transitions[name].copy()
                df['transition'] = name
                train_dfs.append(df)
            else:
                logger.warning(f"Training transition not found: {name}")

        if not train_dfs:
            raise ValueError("No training transitions found")

        train_df = pd.concat(train_dfs, ignore_index=True)

        # Combine test transitions
        test_dfs = []
        for name in self.test_transitions:
            if name in transitions:
                df = transitions[name].copy()
                df['transition'] = name
                test_dfs.append(df)
            else:
                logger.warning(f"Test transition not found: {name}")

        if not test_dfs:
            raise ValueError("No test transitions found")

        test_df = pd.concat(test_dfs, ignore_index=True)

        logger.info(f"Temporal split: {len(train_df)} train, {len(test_df)} test")

        return train_df, test_df

    def get_split_info(self, transitions: Dict[str, pd.DataFrame]) -> Dict:
        """Get information about the splits."""
        info = {
            'train_transitions': self.train_transitions,
            'test_transitions': self.test_transitions,
            'train_samples': 0,
            'test_samples': 0
        }

        for name in self.train_transitions:
            if name in transitions:
                info['train_samples'] += len(transitions[name])

        for name in self.test_transitions:
            if name in transitions:
                info['test_samples'] += len(transitions[name])

        return info


@dataclass
class GroupedTemporalSplitter(TemporalSplitter):
    """
    Temporal splitter with optional grouping (e.g., by district).

    Useful for testing model generalization across different regions.
    """

    group_column: Optional[str] = None

    def get_group_splits(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Get train/test splits by group.

        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame

        Returns:
            Dictionary mapping group name to (train, test) tuple
        """
        if self.group_column is None or self.group_column not in train_df.columns:
            return {'all': (train_df, test_df)}

        groups = {}
        all_groups = set(train_df[self.group_column].unique()) | \
                    set(test_df[self.group_column].unique())

        for group in all_groups:
            group_train = train_df[train_df[self.group_column] == group]
            group_test = test_df[test_df[self.group_column] == group]

            if len(group_train) > 0 and len(group_test) > 0:
                groups[str(group)] = (group_train, group_test)
            else:
                logger.debug(f"Skipping group {group}: insufficient data")

        return groups


def create_stratified_indices(
    y: np.ndarray,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified cross-validation indices.

    Args:
        y: Target array
        n_splits: Number of CV folds
        shuffle: Whether to shuffle before splitting
        random_state: Random seed

    Returns:
        List of (train_indices, val_indices) tuples
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    indices = list(skf.split(np.zeros(len(y)), y))

    return indices


def create_grouped_stratified_indices(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified cross-validation indices with group constraints.

    Groups are kept together (e.g., all samples from same district
    in same fold).

    Args:
        y: Target array
        groups: Group labels array
        n_splits: Number of CV folds
        random_state: Random seed

    Returns:
        List of (train_indices, val_indices) tuples
    """
    from sklearn.model_selection import StratifiedGroupKFold

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    indices = list(sgkf.split(np.zeros(len(y)), y, groups))

    return indices
