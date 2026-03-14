"""
Label Encoding Module
=====================

Specialized encoders for groundwater classification labels
with ordinal parsing for C#S# format.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LabelParser:
    """
    Parse groundwater quality classification labels.

    Extracts C (salinity) and S (sodium) components from labels
    like "C2S1", "C4S3", etc.
    """

    # Valid C values (1-4)
    valid_c: List[int] = field(default_factory=lambda: [1, 2, 3, 4])

    # Valid S values (1-4)
    valid_s: List[int] = field(default_factory=lambda: [1, 2, 3, 4])

    # Pattern for extraction
    pattern: str = r'C(\d)S(\d)'

    def parse(self, label: str) -> Optional[Tuple[int, int]]:
        """
        Parse a single label into (C, S) components.

        Args:
            label: Classification label (e.g., "C2S1")

        Returns:
            Tuple of (C, S) integers, or None if parsing fails
        """
        if pd.isna(label):
            return None

        label = str(label).strip().upper()
        match = re.match(self.pattern, label)

        if match:
            c = int(match.group(1))
            s = int(match.group(2))

            if c in self.valid_c and s in self.valid_s:
                return (c, s)

        logger.debug(f"Could not parse label: {label}")
        return None

    def parse_series(self, labels: pd.Series) -> pd.DataFrame:
        """
        Parse a series of labels.

        Args:
            labels: Series of classification labels

        Returns:
            DataFrame with 'C' and 'S' columns
        """
        parsed = labels.apply(self.parse)
        c_values = parsed.apply(lambda x: x[0] if x else np.nan)
        s_values = parsed.apply(lambda x: x[1] if x else np.nan)

        return pd.DataFrame({
            'C': c_values,
            'S': s_values
        })

    def compute_ordinal_distance(
        self,
        label1: str,
        label2: str,
        c_weight: float = 1.0,
        s_weight: float = 1.0
    ) -> Optional[float]:
        """
        Compute ordinal distance between two labels.

        Args:
            label1: First label
            label2: Second label
            c_weight: Weight for C component distance
            s_weight: Weight for S component distance

        Returns:
            Weighted ordinal distance, or None if parsing fails
        """
        parsed1 = self.parse(label1)
        parsed2 = self.parse(label2)

        if parsed1 is None or parsed2 is None:
            return None

        c_dist = abs(parsed1[0] - parsed2[0])
        s_dist = abs(parsed1[1] - parsed2[1])

        return c_weight * c_dist + s_weight * s_dist

    def is_high_risk(
        self,
        label: str,
        high_risk_classes: Optional[List[str]] = None
    ) -> bool:
        """
        Check if a label represents high risk.

        Args:
            label: Classification label
            high_risk_classes: List of high-risk class labels

        Returns:
            True if high risk
        """
        if high_risk_classes is None:
            # Default: T3_Restricted and T4_Unsafe tiers
            from src.objectives.definitions import HIGH_RISK_CLASSES
            high_risk_classes = HIGH_RISK_CLASSES

        label = str(label).strip().upper()
        return label in high_risk_classes


@dataclass
class OrdinalEncoder:
    """
    Ordinal encoder for classification labels.

    Provides both standard label encoding and ordinal
    encoding based on C and S components.
    """

    label_parser: LabelParser = field(default_factory=LabelParser)
    label_to_idx: Dict[str, int] = field(default_factory=dict, init=False)
    idx_to_label: Dict[int, str] = field(default_factory=dict, init=False)
    label_to_ordinal: Dict[str, Tuple[int, int]] = field(default_factory=dict, init=False)

    def fit(self, labels: Union[pd.Series, List[str]]) -> 'OrdinalEncoder':
        """
        Fit the encoder on observed labels.

        Args:
            labels: Series or list of labels

        Returns:
            self
        """
        if isinstance(labels, list):
            labels = pd.Series(labels)

        unique_labels = labels.dropna().unique()

        # Sort by C then S for consistent ordering
        def sort_key(label):
            parsed = self.label_parser.parse(label)
            if parsed:
                return parsed
            return (99, 99)

        sorted_labels = sorted(unique_labels, key=sort_key)

        self.label_to_idx = {str(label): idx for idx, label in enumerate(sorted_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        # Store ordinal components
        for label in sorted_labels:
            parsed = self.label_parser.parse(label)
            if parsed:
                self.label_to_ordinal[str(label)] = parsed

        logger.info(f"Fitted ordinal encoder with {len(self.label_to_idx)} classes")
        return self

    def transform(self, labels: Union[pd.Series, List[str]]) -> np.ndarray:
        """
        Transform labels to integer indices.

        Args:
            labels: Series or list of labels

        Returns:
            Array of integer indices
        """
        if isinstance(labels, list):
            labels = pd.Series(labels)

        return labels.map(lambda x: self.label_to_idx.get(str(x).strip(), -1)).values

    def inverse_transform(self, indices: np.ndarray) -> List[str]:
        """
        Transform indices back to labels.

        Args:
            indices: Array of integer indices

        Returns:
            List of labels
        """
        return [self.idx_to_label.get(idx, 'Unknown') for idx in indices]

    def get_ordinal_components(
        self,
        labels: Union[pd.Series, List[str]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get C and S components for labels.

        Args:
            labels: Series or list of labels

        Returns:
            Tuple of (C array, S array)
        """
        if isinstance(labels, list):
            labels = pd.Series(labels)

        c_values = []
        s_values = []

        for label in labels:
            parsed = self.label_parser.parse(label)
            if parsed:
                c_values.append(parsed[0])
                s_values.append(parsed[1])
            else:
                c_values.append(np.nan)
                s_values.append(np.nan)

        return np.array(c_values), np.array(s_values)

    @property
    def n_classes(self) -> int:
        """Number of classes."""
        return len(self.label_to_idx)

    @property
    def classes(self) -> List[str]:
        """List of class labels in order."""
        return [self.idx_to_label[i] for i in range(len(self.idx_to_label))]


def compute_ordinal_distance_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    encoder: OrdinalEncoder,
    c_weight: float = 1.0,
    s_weight: float = 1.0
) -> np.ndarray:
    """
    Compute ordinal distances between true and predicted labels.

    Args:
        y_true: True label indices
        y_pred: Predicted label indices
        encoder: Fitted OrdinalEncoder
        c_weight: Weight for C component
        s_weight: Weight for S component

    Returns:
        Array of ordinal distances
    """
    distances = []

    for true_idx, pred_idx in zip(y_true, y_pred):
        true_label = encoder.idx_to_label.get(true_idx)
        pred_label = encoder.idx_to_label.get(pred_idx)

        if true_label and pred_label:
            true_ord = encoder.label_to_ordinal.get(true_label)
            pred_ord = encoder.label_to_ordinal.get(pred_label)

            if true_ord and pred_ord:
                dist = c_weight * abs(true_ord[0] - pred_ord[0]) + \
                       s_weight * abs(true_ord[1] - pred_ord[1])
                distances.append(dist)
            else:
                distances.append(np.nan)
        else:
            distances.append(np.nan)

    return np.array(distances)
