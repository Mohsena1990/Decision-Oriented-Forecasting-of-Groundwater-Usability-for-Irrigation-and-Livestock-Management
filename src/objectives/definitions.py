"""
Objective Definitions Module
============================

Defines the multi-objective optimization framework objectives.

Classification uses a 3-tier semantic risk system based on the USDA
salinity-sodium hazard chart, replacing the raw C#S# collapse.
T4_Unsafe (C4S3, C4S4) is merged into T3_Restricted because transition
data contains almost no T4 samples — keeping them separate starves the
model of meaningful signal:

    T1_Safe        — unrestricted use   (C1S1, C1S2, C1S3, C2S1, OG)
    T2_Marginal    — use with caution   (C1S4, C2S2, C2S3, C3S1, C3S2)
    T3_Restricted  — restricted/unsafe  (C2S4, C3S3, C3S4, C4S1, C4S2,
                                         C4S3, C4S4)

High-risk = T3_Restricted (the single high-risk tier).
"""

from typing import Dict, Any


OBJECTIVE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    'ordinal_distance': {
        'name': 'Ordinal Distance Error',
        'description': 'Mean ordinal distance between predicted and true tier',
        'direction': 'minimize',
        'range': (0, 2),  # Max tier distance: T1 to T3 = 2 steps (3-tier system)
        'weight_default': 1.0,
        'interpretation': 'Measures severity-aware prediction accuracy. '
                         'Lower values indicate predictions closer to true risk tier.'
    },

    'severe_fnr': {
        'name': 'Severe-Risk False Negative Rate',
        'description': 'Rate of missed high-risk groundwater classifications (T3+T4)',
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
        'interpretation': 'Overall classification performance across all tiers. '
                         'Macro averaging ensures minority tiers are weighted equally.'
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


# ---------------------------------------------------------------------------
# 3-Tier Semantic Risk Classification (USDA salinity-sodium hazard chart)
# ---------------------------------------------------------------------------
# T4_Unsafe (C4S3, C4S4) is merged into T3_Restricted because transition
# data contains almost no T4-labelled samples, making a 4-class model
# unable to learn the T4 boundary.  The merged tier still captures the
# full "do not use" risk signal under a single high-risk label.

RISK_TIER_MAPPING: Dict[str, str] = {
    # Tier 1 — Safe: suitable for all irrigation/livestock uses
    'C1S1': 'T1_Safe', 'C1S2': 'T1_Safe', 'C1S3': 'T1_Safe', 'C2S1': 'T1_Safe',
    'OG':   'T1_Safe',   # "Ordinary Groundwater" — generally safe quality

    # Tier 2 — Marginal: use with caution, some crop/livestock restrictions
    'C1S4': 'T2_Marginal', 'C2S2': 'T2_Marginal', 'C2S3': 'T2_Marginal',
    'C3S1': 'T2_Marginal', 'C3S2': 'T2_Marginal',

    # Tier 3 — Restricted/Unsafe: not suitable; includes former T4 (C4S3, C4S4)
    # merged because transition data has <3 T4 samples — insufficient to learn
    'C2S4': 'T3_Restricted', 'C3S3': 'T3_Restricted', 'C3S4': 'T3_Restricted',
    'C4S1': 'T3_Restricted', 'C4S2': 'T3_Restricted',
    'C4S3': 'T3_Restricted', 'C4S4': 'T3_Restricted',  # formerly T4_Unsafe
}

# Tier 3 is the single high-risk tier for the safety FNR objective
HIGH_RISK_CLASSES = ['T3_Restricted']

# Ordinal encoding for tier-distance computations (tier index on a 1-3 scale)
ORDINAL_ENCODING: Dict[str, tuple] = {
    'T1_Safe':       (1, 1),
    'T2_Marginal':   (2, 2),
    'T3_Restricted': (3, 3),
}


def get_high_risk_indices(label_encoder: Dict[str, int]) -> list:
    """
    Get indices of high-risk tiers (T3_Restricted, T4_Unsafe) from label encoder.

    Args:
        label_encoder: Mapping of tier label string to index

    Returns:
        List of high-risk tier indices
    """
    indices = []
    for cls in HIGH_RISK_CLASSES:
        if cls in label_encoder:
            indices.append(label_encoder[cls])
    return indices


def get_ordinal_mapping(label_encoder: Dict[str, int]) -> Dict[str, tuple]:
    """
    Get ordinal (tier, tier) mapping for labels in encoder.

    Args:
        label_encoder: Mapping of tier label to index

    Returns:
        Dictionary mapping tier label to (tier, tier) tuple
    """
    mapping = {}
    for label in label_encoder.keys():
        if label in ORDINAL_ENCODING:
            mapping[label] = ORDINAL_ENCODING[label]
    return mapping


def map_raw_label_to_tier(raw_label: str) -> str:
    """
    Map a raw C#S# classification label to its semantic risk tier (3-tier).

    Unrecognised labels default to T2_Marginal (conservative middle tier).

    Args:
        raw_label: Original classification string (e.g. 'C3S2', 'OG')

    Returns:
        Tier label string (e.g. 'T2_Marginal')
    """
    label = str(raw_label).strip()
    return RISK_TIER_MAPPING.get(label, 'T2_Marginal')
