"""
Scenario Simulation Engine
==========================

Simulates "what-if" scenarios by perturbing key hydrochemical
features and measuring impacts on groundwater quality predictions.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    - Percentage perturbations (e.g., +10% TDS)
    - Threshold-based scenarios (e.g., RSC crossing thresholds)
    - Combined multi-parameter scenarios
    """

    feature_mapping: Dict[str, int] = field(default_factory=dict)  # Feature name -> column index
    high_risk_indices: List[int] = field(default_factory=list)

    def simulate_percentage_change(
        self,
        model: Any,
        X: np.ndarray,
        feature_name: str,
        percentage: float,
        direction: str = "increase"
    ) -> ScenarioResult:
        """
        Simulate percentage change in a feature.

        Args:
            model: Fitted model
            X: Original feature data
            feature_name: Name of feature to perturb
            percentage: Percentage change (e.g., 0.1 for 10%)
            direction: "increase" or "decrease"

        Returns:
            ScenarioResult with impact analysis
        """
        if feature_name not in self.feature_mapping:
            raise ValueError(f"Feature {feature_name} not in mapping")

        col_idx = self.feature_mapping[feature_name]

        # Create perturbed data
        X_perturbed = X.copy()
        multiplier = 1 + percentage if direction == "increase" else 1 - percentage
        X_perturbed[:, col_idx] = X_perturbed[:, col_idx] * multiplier

        return self._evaluate_scenario(
            model, X, X_perturbed,
            f"{feature_name}_{direction}_{int(percentage*100)}pct"
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
        Simulate feature values crossing a threshold.

        Args:
            model: Fitted model
            X: Original feature data
            feature_name: Name of feature
            threshold: Threshold value
            cross_direction: "above" or "below"

        Returns:
            ScenarioResult with impact analysis
        """
        if feature_name not in self.feature_mapping:
            raise ValueError(f"Feature {feature_name} not in mapping")

        col_idx = self.feature_mapping[feature_name]

        # Create perturbed data
        X_perturbed = X.copy()

        if cross_direction == "above":
            # Set values below threshold to just above
            mask = X_perturbed[:, col_idx] < threshold
            X_perturbed[mask, col_idx] = threshold * 1.01
        else:
            # Set values above threshold to just below
            mask = X_perturbed[:, col_idx] > threshold
            X_perturbed[mask, col_idx] = threshold * 0.99

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
            multiplier = 1 + percentage if direction == "increase" else 1 - percentage
            X_perturbed[:, col_idx] = X_perturbed[:, col_idx] * multiplier

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

        # Compute high-risk probability change
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

            records.append({
                'scenario': name,
                'n_samples': len(result.baseline_predictions),
                'baseline_high_risk_rate': baseline_high_risk,
                'scenario_high_risk_rate': scenario_high_risk,
                'high_risk_rate_change': scenario_high_risk - baseline_high_risk,
                'mean_prob_change': np.mean(result.high_risk_prob_change),
                'max_prob_change': np.max(result.high_risk_prob_change),
                'pct_class_changed': np.mean(
                    result.baseline_predictions != result.scenario_predictions
                ) * 100
            })

        return pd.DataFrame(records)
