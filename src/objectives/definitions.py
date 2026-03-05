"""
Objective Definitions Module
============================

Defines the multi-objective optimization framework objectives.
"""

from typing import Dict, Any


OBJECTIVE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    'ordinal_distance': {
        'name': 'Ordinal Distance Error',
        'description': 'Mean ordinal distance between predicted and true classes',
        'direction': 'minimize',
        'range': (0, 6),  # Max C distance (3) + max S distance (3)
        'weight_default': 1.0,
        'interpretation': 'Measures severity-aware prediction accuracy. '
                         'Lower values indicate predictions closer to true risk level.'
    },

    'severe_fnr': {
        'name': 'Severe-Risk False Negative Rate',
        'description': 'Rate of missed high-risk groundwater classifications',
        'direction': 'minimize',
        'range': (0, 1),
        'weight_default': 1.5,
        'interpretation': 'Critical safety metric. Higher weight because missing '
                         'high-risk cases can lead to adverse health outcomes.'
    },

    'macro_f1_complement': {
        'name': '1 - Macro F1',
        'description': 'Complement of macro-averaged F1 score',
        'direction': 'minimize',
        'range': (0, 1),
        'weight_default': 1.0,
        'interpretation': 'Overall classification performance across all classes. '
                         'Macro averaging ensures minority classes are weighted equally.'
    },

    'complexity': {
        'name': 'Model Complexity',
        'description': 'Normalized complexity based on features and model size',
        'direction': 'minimize',
        'range': (0, 1),
        'weight_default': 0.5,
        'interpretation': 'Parsimony metric encouraging simpler models. '
                         'Simpler models are more interpretable and generalizable.'
    }
}


HIGH_RISK_CLASSES = [
    'C4S1', 'C4S2', 'C4S3', 'C4S4',  # Very high salinity
    'C3S3', 'C3S4',                    # High salinity + high sodium
    'Other',                           # Merged rare classes (includes C3S3/C3S4/C4S3/C4S4)
]


ORDINAL_ENCODING = {
    'C1S1': (1, 1), 'C1S2': (1, 2), 'C1S3': (1, 3), 'C1S4': (1, 4),
    'C2S1': (2, 1), 'C2S2': (2, 2), 'C2S3': (2, 3), 'C2S4': (2, 4),
    'C3S1': (3, 1), 'C3S2': (3, 2), 'C3S3': (3, 3), 'C3S4': (3, 4),
    'C4S1': (4, 1), 'C4S2': (4, 2), 'C4S3': (4, 3), 'C4S4': (4, 4),
    # "Other" contains merged rare classes (mostly C3S3/C3S4/C4S3/C4S4 + OG/C1S1/C2S2)
    # Use (3, 3) as a conservative mid-high risk ordinal proxy for distance computation
    'Other': (3, 3),
}


def get_high_risk_indices(label_encoder: Dict[str, int]) -> list:
    """
    Get indices of high-risk classes from label encoder.

    Args:
        label_encoder: Mapping of label string to index

    Returns:
        List of high-risk class indices
    """
    indices = []
    for cls in HIGH_RISK_CLASSES:
        if cls in label_encoder:
            indices.append(label_encoder[cls])
    return indices


def get_ordinal_mapping(label_encoder: Dict[str, int]) -> Dict[str, tuple]:
    """
    Get ordinal (C, S) mapping for labels in encoder.

    Args:
        label_encoder: Mapping of label string to index

    Returns:
        Dictionary mapping label to (C, S) tuple
    """
    mapping = {}
    for label in label_encoder.keys():
        if label in ORDINAL_ENCODING:
            mapping[label] = ORDINAL_ENCODING[label]
    return mapping
