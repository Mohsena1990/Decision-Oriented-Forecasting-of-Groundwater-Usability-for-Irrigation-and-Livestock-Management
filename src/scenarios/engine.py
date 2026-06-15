"""
Scenario Simulation Engine
==========================

Simulates "what-if" scenarios by perturbing key hydrochemical
features and measuring impacts on groundwater quality predictions.

Scaling note
------------
Tree-based models (CatBoost, LightGBM) receive raw, unscaled features.
Deep models (GRU, LSTM) receive StandardScaler-normalised features.

When an sklearn-compatible ``scaler`` is provided to ScenarioEngine,
percentage perturbations are applied correctly in *raw-feature space*:
the column is inverse-transformed, perturbed, then re-transformed.
This ensures that "+10% TDS" always means a true 10% increase in the
original mg/L values, regardless of whether the model sees scaled data.

Without a scaler (tree models), the raw values are perturbed directly.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    """Container for scenario simulation results."""
    scenario_name: str
    baseline_proba: np.ndarray
    scenario_proba: np.ndarray
    baseline_predictions: np.ndarray
    scenario_predictions: np.ndarray
    high_risk_prob_change: np.ndarray  # Change in P(high-risk)
    transition_matrix: np.ndarray
    district_impacts: Optional[pd.DataFrame] = None


@dataclass
class ScenarioEngine:
    """
    Simulate hydrochemical scenarios and measure prediction impacts.

    Supports:
    - Percentage perturbations (e.g., +10% TDS) — always in raw-feature space
    - Threshold-based scenarios (e.g., RSC crossing thresholds)
    - Combined multi-parameter scenarios

    Attributes:
        feature_mapping: Feature name -> column index in X
        high_risk_indices: Encoded class indices that are high-risk
        scaler: Optional sklearn scaler used when building X.  When
            provided, perturbations are computed in raw space and then
            re-scaled, producing physically correct "what-if" values.
        scaler_feature_names: Ordered list of feature names the scaler
            was fitted on (needed to locate each feature's column inside
            the scaler's internal arrays).
    """

    feature_mapping: Dict[str, int] = field(default_factory=dict)
    high_risk_indices: List[int] = field(default_factory=list)
    scaler: Optional[Any] = field(default=None)
    scaler_feature_names: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Internal helpers for scale-aware perturbation
    # ------------------------------------------------------------------

    def _perturb_column_raw(
        self,
        X: np.ndarray,
        col_idx: int,
        feature_name: str,
        multiplier: float
    ) -> np.ndarray:
        """
        Return a copy of X with one column perturbed by *multiplier* in
        raw-feature space.

        If a scaler is attached and the feature is in its vocabulary the
        perturbation is:
            1. Extract the scaled column.
            2. Inverse-transform to raw values using the feature's
               per-column mean and std stored in the scaler.
            3. Apply the multiplier to the raw values.
            4. Re-scale the result back.

        For unscaled data (no scaler, or feature not in scaler), the
        column is multiplied directly — which is already correct.
        """
        X_out = X.copy()

        if self.scaler is not None and feature_name in self.scaler_feature_names:
            scaler_col = self.scaler_feature_names.index(feature_name)
            mean = self.scaler.mean_[scaler_col]
            std = self.scaler.scale_[scaler_col]  # StandardScaler stores std in .scale_

            # x_raw = x_scaled * std + mean
            raw_vals = X_out[:, col_idx] * std + mean

            # Perturb in raw space
            raw_vals_new = raw_vals * multiplier

            # Re-scale: x_scaled_new = (x_raw_new - mean) / std
            X_out[:, col_idx] = (raw_vals_new - mean) / std
        else:
            # Raw features (tree models) — direct multiplication is correct
            X_out[:, col_idx] = X_out[:, col_idx] * multiplier

        return X_out

    def simulate_percentage_change(
        self,
        model: Any,
        X: np.ndarray,
        feature_name: str,
        percentage: float,
        direction: str = "increase"
    ) -> ScenarioResult:
        """
        Simulate percentage change in a feature (perturbation in raw space).

        Args:
            model: Fitted model
            X: Feature data (may be scaled or raw; see ``scaler`` attribute)
            feature_name: Name of feature to perturb
            percentage: Fractional change magnitude (e.g., 0.1 for 10%)
            direction: "increase" or "decrease"

        Returns:
            ScenarioResult with impact analysis
        """
        if feature_name not in self.feature_mapping:
            raise ValueError(f"Feature {feature_name} not in mapping")

        col_idx = self.feature_mapping[feature_name]
        multiplier = 1.0 + percentage if direction == "increase" else 1.0 - percentage

        X_perturbed = self._perturb_column_raw(X, col_idx, feature_name, multiplier)

        return self._evaluate_scenario(
            model, X, X_perturbed,
            f"{feature_name}_{direction}_{int(percentage * 100)}pct"
        )

    def simulate_threshold_crossing(
        self,
        model: Any,
        X: np.ndarray,
        feature_name: str,
        threshold: float,
        cross_direction: str = "above"
    ) -> ScenarioResult:
        """
        Simulate feature values crossing a threshold (in raw space).

        The threshold is expressed in raw (physical) units.  When a
        scaler is attached, the threshold is converted to scaled space
        before applying the mask.

        Args:
            model: Fitted model
            X: Feature data (may be scaled or raw)
            feature_name: Name of feature
            threshold: Threshold in raw physical units (e.g. mg/L for TDS)
            cross_direction: "above" or "below"

        Returns:
            ScenarioResult with impact analysis
        """
        if feature_name not in self.feature_mapping:
            raise ValueError(f"Feature {feature_name} not in mapping")

        col_idx = self.feature_mapping[feature_name]
        X_perturbed = X.copy()

        # Convert threshold to the same space as X
        if self.scaler is not None and feature_name in self.scaler_feature_names:
            scaler_col = self.scaler_feature_names.index(feature_name)
            mean = self.scaler.mean_[scaler_col]
            std = self.scaler.scale_[scaler_col]
            threshold_scaled = (threshold - mean) / std
            nudge_above = threshold_scaled + (0.01 * std)  # small absolute nudge
            nudge_below = threshold_scaled - (0.01 * std)
        else:
            threshold_scaled = threshold
            nudge_above = threshold * 1.01
            nudge_below = threshold * 0.99

        if cross_direction == "above":
            mask = X_perturbed[:, col_idx] < threshold_scaled
            X_perturbed[mask, col_idx] = nudge_above
        else:
            mask = X_perturbed[:, col_idx] > threshold_scaled
            X_perturbed[mask, col_idx] = nudge_below

        return self._evaluate_scenario(
            model, X, X_perturbed,
            f"{feature_name}_threshold_{threshold}_{cross_direction}"
        )

    def simulate_combined_scenario(
        self,
        model: Any,
        X: np.ndarray,
        perturbations: Dict[str, Tuple[float, str]],
        scenario_name: str
    ) -> ScenarioResult:
        """
        Simulate combined multi-parameter scenario.

        Args:
            model: Fitted model
            X: Original feature data
            perturbations: Dict of feature_name -> (percentage, direction)
            scenario_name: Name for this scenario

        Returns:
            ScenarioResult with impact analysis
        """
        X_perturbed = X.copy()

        for feature_name, (percentage, direction) in perturbations.items():
            if feature_name not in self.feature_mapping:
                logger.warning(f"Skipping unknown feature: {feature_name}")
                continue

            col_idx = self.feature_mapping[feature_name]
            multiplier = 1.0 + percentage if direction == "increase" else 1.0 - percentage
            X_perturbed = self._perturb_column_raw(X_perturbed, col_idx, feature_name, multiplier)

        return self._evaluate_scenario(model, X, X_perturbed, scenario_name)

    def _evaluate_scenario(
        self,
        model: Any,
        X_baseline: np.ndarray,
        X_scenario: np.ndarray,
        scenario_name: str
    ) -> ScenarioResult:
        """Evaluate baseline vs scenario predictions."""
        # Get predictions
        baseline_proba = model.predict_proba(X_baseline)
        scenario_proba = model.predict_proba(X_scenario)

        baseline_pred = np.argmax(baseline_proba, axis=1)
        scenario_pred = np.argmax(scenario_proba, axis=1)

        # Compute high-risk probability change (per sample)
        high_risk_prob_baseline = self._compute_high_risk_prob(baseline_proba)
        high_risk_prob_scenario = self._compute_high_risk_prob(scenario_proba)
        high_risk_prob_change = high_risk_prob_scenario - high_risk_prob_baseline

        # Compute transition matrix
        n_classes = baseline_proba.shape[1]
        transition_matrix = self._compute_transition_matrix(
            baseline_pred, scenario_pred, n_classes
        )

        return ScenarioResult(
            scenario_name=scenario_name,
            baseline_proba=baseline_proba,
            scenario_proba=scenario_proba,
            baseline_predictions=baseline_pred,
            scenario_predictions=scenario_pred,
            high_risk_prob_change=high_risk_prob_change,
            transition_matrix=transition_matrix
        )

    def compute_safe_to_risky_escalation(self, result: ScenarioResult) -> float:
        """
        Fraction of currently-safe (non-T3) wells that become high-risk
        under the scenario.

        This is more interpretable than mean_prob_change, which averages
        over wells already classified as T3 (where further increases have
        no additional impact on the metric).

        Returns a value in [0, 1].
        """
        if not self.high_risk_indices:
            return 0.0
        baseline_safe = ~np.isin(result.baseline_predictions, self.high_risk_indices)
        n_safe = baseline_safe.sum()
        if n_safe == 0:
            return 0.0
        escalated = baseline_safe & np.isin(result.scenario_predictions, self.high_risk_indices)
        return float(escalated.sum() / n_safe)

    def _compute_high_risk_prob(self, proba: np.ndarray) -> np.ndarray:
        """Compute probability of high-risk class."""
        if not self.high_risk_indices:
            return np.zeros(len(proba))

        return np.sum(proba[:, self.high_risk_indices], axis=1)

    def _compute_transition_matrix(
        self,
        baseline_pred: np.ndarray,
        scenario_pred: np.ndarray,
        n_classes: int
    ) -> np.ndarray:
        """Compute class transition matrix."""
        matrix = np.zeros((n_classes, n_classes), dtype=int)

        for base, scen in zip(baseline_pred, scenario_pred):
            if 0 <= base < n_classes and 0 <= scen < n_classes:
                matrix[base, scen] += 1

        return matrix

    def compute_district_impacts(
        self,
        results: List[ScenarioResult],
        districts: np.ndarray
    ) -> pd.DataFrame:
        """
        Compute district-level vulnerability rankings.

        Args:
            results: List of scenario results
            districts: Array of district labels for each sample

        Returns:
            DataFrame with district vulnerability rankings
        """
        unique_districts = np.unique(districts)
        records = []

        for result in results:
            for district in unique_districts:
                mask = districts == district

                if mask.sum() == 0:
                    continue

                # Compute district statistics
                district_prob_change = result.high_risk_prob_change[mask]

                records.append({
                    'scenario': result.scenario_name,
                    'district': district,
                    'n_samples': mask.sum(),
                    'mean_risk_change': np.mean(district_prob_change),
                    'max_risk_change': np.max(district_prob_change),
                    'pct_increased_risk': np.mean(district_prob_change > 0) * 100,
                    'baseline_high_risk_rate': np.mean(
                        np.isin(result.baseline_predictions[mask], self.high_risk_indices)
                    ),
                    'scenario_high_risk_rate': np.mean(
                        np.isin(result.scenario_predictions[mask], self.high_risk_indices)
                    )
                })

        df = pd.DataFrame(records)

        # Rank districts by vulnerability
        if not df.empty:
            df['vulnerability_rank'] = df.groupby('scenario')['mean_risk_change'].rank(
                ascending=False
            )

        return df

    def run_standard_scenarios(
        self,
        model: Any,
        X: np.ndarray
    ) -> Dict[str, ScenarioResult]:
        """
        Run standard set of scenarios.

        Args:
            model: Fitted model
            X: Feature data

        Returns:
            Dictionary of scenario name to result
        """
        results = {}

        # TDS scenarios
        for pct in [0.1, 0.2, 0.3]:
            if 'TDS' in self.feature_mapping:
                result = self.simulate_percentage_change(
                    model, X, 'TDS', pct, 'increase'
                )
                results[result.scenario_name] = result

        # SAR scenarios
        for pct in [0.1, 0.2]:
            if 'SAR' in self.feature_mapping:
                result = self.simulate_percentage_change(
                    model, X, 'SAR', pct, 'increase'
                )
                results[result.scenario_name] = result

        # RSC threshold scenarios
        if 'RSC' in self.feature_mapping:
            for threshold in [1.25, 2.5]:
                result = self.simulate_threshold_crossing(
                    model, X, 'RSC', threshold, 'above'
                )
                results[result.scenario_name] = result

        # Combined high salinity scenario
        if 'TDS' in self.feature_mapping and 'SAR' in self.feature_mapping:
            result = self.simulate_combined_scenario(
                model, X,
                {'TDS': (0.2, 'increase'), 'SAR': (0.1, 'increase')},
                'high_salinity_combined'
            )
            results[result.scenario_name] = result

        return results

    def generate_scenario_report(
        self,
        results: Dict[str, ScenarioResult],
        idx_to_label: Dict[int, str]
    ) -> pd.DataFrame:
        """Generate summary report of all scenarios."""
        records = []

        for name, result in results.items():
            baseline_high_risk = np.mean(
                np.isin(result.baseline_predictions, self.high_risk_indices)
            )
            scenario_high_risk = np.mean(
                np.isin(result.scenario_predictions, self.high_risk_indices)
            )

            # safe_to_risky: fraction of currently-safe wells that escalate to T3.
            # This is the decision-relevant metric — it measures NEW risk, not
            # diluted average across already-risky wells.
            safe_to_risky = self.compute_safe_to_risky_escalation(result)

            records.append({
                'scenario': name,
                'n_samples': len(result.baseline_predictions),
                'baseline_high_risk_rate': baseline_high_risk,
                'scenario_high_risk_rate': scenario_high_risk,
                'high_risk_rate_change': scenario_high_risk - baseline_high_risk,
                'safe_to_risky_escalation': safe_to_risky,
                'mean_prob_change': np.mean(result.high_risk_prob_change),
                'max_prob_change': np.max(result.high_risk_prob_change),
                'pct_class_changed': np.mean(
                    result.baseline_predictions != result.scenario_predictions
                ) * 100
            })

        return pd.DataFrame(records)
