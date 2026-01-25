"""
Model Selection Module
======================

Implements two-level model selection:
1. Select best configuration per model (VIKOR on Pareto front)
2. Select best model among best-configured models (VIKOR on test performance)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .vikor import VIKOR, VIKORResult

logger = logging.getLogger(__name__)


@dataclass
class ConfigurationResult:
    """Result for a single model configuration."""
    model_name: str
    params: Dict[str, Any]
    feature_mask: Optional[np.ndarray]
    cv_objectives: np.ndarray  # Mean objectives from CV
    cv_std: np.ndarray  # Std of objectives from CV
    test_objectives: Optional[np.ndarray] = None


@dataclass
class ModelSelectionResult:
    """Final model selection result."""
    best_model_name: str
    best_config: ConfigurationResult
    all_results: Dict[str, ConfigurationResult]
    level1_rankings: Dict[str, VIKORResult]  # Per-model config rankings
    level2_ranking: VIKORResult  # Cross-model ranking


@dataclass
class ModelSelector:
    """
    Two-level model selection using VIKOR.

    Level 1: Select best configuration for each model type
    Level 2: Select best model among best-configured models
    """

    vikor_v: float = 0.5
    objective_weights: Optional[np.ndarray] = None

    def select_best_config(
        self,
        pareto_front: List[Tuple[np.ndarray, np.ndarray]],
        model_name: str
    ) -> Tuple[int, VIKORResult]:
        """
        Select best configuration from Pareto front for a single model.

        Args:
            pareto_front: List of (position, objectives) tuples
            model_name: Name of the model

        Returns:
            Tuple of (best index, VIKOR result)
        """
        if not pareto_front:
            raise ValueError(f"Empty Pareto front for {model_name}")

        objectives = np.array([obj for _, obj in pareto_front])

        vikor = VIKOR(v=self.vikor_v, weights=self.objective_weights)
        result = vikor.rank(objectives)

        logger.info(f"Model {model_name}: Selected config {result.best_idx} from {len(pareto_front)} Pareto solutions")

        return result.best_idx, result

    def select_best_model(
        self,
        model_results: Dict[str, ConfigurationResult]
    ) -> ModelSelectionResult:
        """
        Select best model from best-configured models.

        Args:
            model_results: Dictionary mapping model name to best ConfigurationResult

        Returns:
            ModelSelectionResult with selected model and rankings
        """
        model_names = list(model_results.keys())

        # Use test objectives if available, otherwise CV objectives
        objectives = []
        for name in model_names:
            result = model_results[name]
            obj = result.test_objectives if result.test_objectives is not None else result.cv_objectives
            objectives.append(obj)

        objectives = np.array(objectives)

        # Apply VIKOR
        vikor = VIKOR(v=self.vikor_v, weights=self.objective_weights)
        level2_result = vikor.rank(objectives)

        best_model_name = model_names[level2_result.best_idx]

        logger.info(f"Best model selected: {best_model_name}")
        logger.info(f"Model rankings: {[model_names[i] for i in level2_result.rankings]}")

        return ModelSelectionResult(
            best_model_name=best_model_name,
            best_config=model_results[best_model_name],
            all_results=model_results,
            level1_rankings={},  # Will be populated during full selection
            level2_ranking=level2_result
        )

    def full_selection(
        self,
        model_pareto_fronts: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
        position_decoders: Dict[str, callable],
        test_evaluator: Optional[callable] = None
    ) -> ModelSelectionResult:
        """
        Run complete two-level selection.

        Args:
            model_pareto_fronts: Dict mapping model name to Pareto front
            position_decoders: Dict mapping model name to position decoder function
            test_evaluator: Optional function to evaluate on test set

        Returns:
            Complete ModelSelectionResult
        """
        level1_rankings = {}
        best_configs = {}

        # Level 1: Select best config per model
        for model_name, front in model_pareto_fronts.items():
            if not front:
                logger.warning(f"Skipping {model_name}: empty Pareto front")
                continue

            best_idx, vikor_result = self.select_best_config(front, model_name)
            level1_rankings[model_name] = vikor_result

            # Get best configuration
            best_position, best_objectives = front[best_idx]
            params, feature_mask = position_decoders[model_name](best_position)

            # Evaluate on test if evaluator provided
            test_objectives = None
            if test_evaluator:
                test_objectives = test_evaluator(model_name, params, feature_mask)

            best_configs[model_name] = ConfigurationResult(
                model_name=model_name,
                params=params,
                feature_mask=feature_mask,
                cv_objectives=best_objectives,
                cv_std=np.zeros_like(best_objectives),  # Would need proper tracking
                test_objectives=test_objectives
            )

        # Level 2: Select best model
        if not best_configs:
            raise ValueError("No valid model configurations found")

        level2_result = self.select_best_model(best_configs)
        level2_result.level1_rankings = level1_rankings

        return level2_result

    def generate_comparison_table(
        self,
        result: ModelSelectionResult,
        objective_names: List[str]
    ) -> pd.DataFrame:
        """
        Generate comparison table of best models.

        Args:
            result: ModelSelectionResult
            objective_names: Names of objectives

        Returns:
            DataFrame comparing models
        """
        rows = []

        for model_name, config in result.all_results.items():
            row = {
                'Model': model_name,
                'Rank': int(np.where(result.level2_ranking.rankings ==
                          list(result.all_results.keys()).index(model_name))[0][0] + 1),
                'Q_Score': result.level2_ranking.Q[
                    list(result.all_results.keys()).index(model_name)
                ]
            }

            # Add objectives
            obj = config.test_objectives if config.test_objectives is not None else config.cv_objectives
            for i, name in enumerate(objective_names):
                if i < len(obj):
                    row[name] = obj[i]

            # Add key parameters
            row['n_features'] = int(np.sum(config.feature_mask)) if config.feature_mask is not None else 'N/A'

            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values('Rank')

        return df
