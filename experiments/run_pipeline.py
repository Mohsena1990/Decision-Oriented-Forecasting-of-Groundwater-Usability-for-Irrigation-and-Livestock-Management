"""
Main Pipeline Runner
====================

Single entrypoint for the groundwater quality forecasting framework.

Pipeline Stages:
    A. Data Ingestion + Harmonization
    B. Transition Building
    C. Data Quality Validation
    D. Preprocessing
    E. Feature Selection (Filter Warm-Start)
    F. PSO-GWO Multi-Objective Optimization (for all 4 models)
    G. VIKOR Model Selection (two-level)
    H. SHAP Explainability
    I. Scenario Simulation
    J. Paper Outputs

Usage:
    python -m experiments.run_pipeline --config configs/main.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.utils.config import load_config, get_config_value
from src.utils.logging import setup_logging
from src.utils.reproducibility import set_seed

from src.data import DataIngestion, DataHarmonizer, TransitionBuilder
from src.data_quality import DataQualityValidator, DataCleaner, DataQualityReport
from src.preprocessing import PreprocessingPipeline, LabelParser, OrdinalEncoder
from src.evaluation import TemporalSplitter, CrossValidator, MetricsCalculator
from src.imbalance import ImbalanceHandler
from src.feature_selection import FilterSelector

# All 4 models
from src.models.trees import CatBoostForecaster, LightGBMForecaster
from src.models.base import ForecasterConfig, compute_class_weights

# Deep models
try:
    from src.models.deep import GRUForecaster, LSTMForecaster
    DEEP_MODELS_AVAILABLE = True
except ImportError:
    DEEP_MODELS_AVAILABLE = False
    GRUForecaster = None
    LSTMForecaster = None

# Optimization & Decision Making
from src.optimization.pso_gwo import PSOGWO, OptimizationResult, create_search_space, decode_position
from src.decision.vikor import VIKOR, VIKORResult
from src.decision.model_selection import ModelSelector, ConfigurationResult, ModelSelectionResult

from src.objectives import ObjectiveCalculator
from src.objectives.definitions import get_high_risk_indices, get_ordinal_mapping, HIGH_RISK_CLASSES

# Explainability & Scenarios
from src.explain.shap_tree import TreeSHAPExplainer
from src.explain.surrogate import SurrogateExplainer
from src.scenarios.engine import ScenarioEngine

from src.reporting import FigureGenerator, TableGenerator, ManagerialInsights

logger = logging.getLogger(__name__)


# =============================================================================
# MODEL FACTORIES
# =============================================================================

def create_catboost_factory(n_classes: int, class_weights: Dict[int, float], random_state: int):
    """Factory function for CatBoost models."""
    def factory(params: Dict[str, Any]) -> CatBoostForecaster:
        config = ForecasterConfig(
            params={
                'iterations': int(params.get('iterations', 500)),
                'depth': int(params.get('depth', 6)),
                'learning_rate': params.get('learning_rate', 0.1),
                'l2_leaf_reg': params.get('l2_leaf_reg', 3),
                'early_stopping_rounds': 50
            },
            n_classes=n_classes,
            class_weights=class_weights,
            random_state=random_state
        )
        return CatBoostForecaster(config=config)
    return factory


def create_lightgbm_factory(n_classes: int, class_weights: Dict[int, float], random_state: int):
    """Factory function for LightGBM models."""
    def factory(params: Dict[str, Any]) -> LightGBMForecaster:
        config = ForecasterConfig(
            params={
                'n_estimators': int(params.get('n_estimators', 500)),
                'max_depth': int(params.get('max_depth', 6)),
                'learning_rate': params.get('learning_rate', 0.1),
                'num_leaves': int(params.get('num_leaves', 31)),
                'min_child_samples': int(params.get('min_child_samples', 20)),
                'early_stopping_rounds': 50
            },
            n_classes=n_classes,
            class_weights=class_weights,
            random_state=random_state
        )
        return LightGBMForecaster(config=config)
    return factory


def create_gru_factory(n_classes: int, class_weights: Dict[int, float], random_state: int, categorical_indices: List[int]):
    """Factory function for GRU models."""
    def factory(params: Dict[str, Any]) -> GRUForecaster:
        config = ForecasterConfig(
            params={
                'hidden_size': int(params.get('hidden_size', 64)),
                'num_layers': int(params.get('num_layers', 1)),
                'dropout': params.get('dropout', 0.2),
                'learning_rate': params.get('learning_rate', 0.001),
                'batch_size': int(params.get('batch_size', 32)),
                'embedding_dim': int(params.get('embedding_dim', 16)),
                'max_epochs': 200,
                'patience': 20
            },
            n_classes=n_classes,
            class_weights=class_weights,
            random_state=random_state
        )
        model = GRUForecaster(config=config)
        model._categorical_indices_hint = categorical_indices
        return model
    return factory


def create_lstm_factory(n_classes: int, class_weights: Dict[int, float], random_state: int, categorical_indices: List[int]):
    """Factory function for LSTM models."""
    def factory(params: Dict[str, Any]) -> LSTMForecaster:
        config = ForecasterConfig(
            params={
                'hidden_size': int(params.get('hidden_size', 64)),
                'num_layers': int(params.get('num_layers', 1)),
                'dropout': params.get('dropout', 0.2),
                'learning_rate': params.get('learning_rate', 0.001),
                'batch_size': int(params.get('batch_size', 32)),
                'embedding_dim': int(params.get('embedding_dim', 16)),
                'max_epochs': 200,
                'patience': 20
            },
            n_classes=n_classes,
            class_weights=class_weights,
            random_state=random_state
        )
        model = LSTMForecaster(config=config)
        model._categorical_indices_hint = categorical_indices
        return model
    return factory


# =============================================================================
# SEARCH SPACE DEFINITIONS
# =============================================================================

def get_catboost_search_space(n_features: int):
    """Get search space for CatBoost."""
    param_bounds = {
        'iterations': (100, 1000),
        'depth': (4, 10),
        'learning_rate': (-2, -0.5),  # log scale: 10^-2 to 10^-0.5
        'l2_leaf_reg': (1, 10)
    }
    param_types = {
        'iterations': 'int',
        'depth': 'int',
        'learning_rate': 'log',
        'l2_leaf_reg': 'float'
    }
    return param_bounds, param_types


def get_lightgbm_search_space(n_features: int):
    """Get search space for LightGBM."""
    param_bounds = {
        'n_estimators': (100, 1000),
        'max_depth': (4, 10),
        'learning_rate': (-2, -0.5),  # log scale
        'num_leaves': (15, 127),
        'min_child_samples': (5, 50)
    }
    param_types = {
        'n_estimators': 'int',
        'max_depth': 'int',
        'learning_rate': 'log',
        'num_leaves': 'int',
        'min_child_samples': 'int'
    }
    return param_bounds, param_types


def get_gru_search_space(n_features: int):
    """Get search space for GRU."""
    param_bounds = {
        'hidden_size': (32, 128),
        'num_layers': (1, 2),
        'dropout': (0.1, 0.4),
        'learning_rate': (-4, -2),  # log scale: 10^-4 to 10^-2
        'batch_size': (16, 64),
        'embedding_dim': (8, 32)
    }
    param_types = {
        'hidden_size': 'int',
        'num_layers': 'int',
        'dropout': 'float',
        'learning_rate': 'log',
        'batch_size': 'int',
        'embedding_dim': 'int'
    }
    return param_bounds, param_types


def get_lstm_search_space(n_features: int):
    """Get search space for LSTM."""
    # Same as GRU
    return get_gru_search_space(n_features)


# =============================================================================
# OBJECTIVE FUNCTION CREATION
# =============================================================================

def create_objective_function(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_factory: Callable,
    calculator: ObjectiveCalculator,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    param_bounds: Dict[str, Tuple[float, float]],
    param_types: Dict[str, str],
    n_features: int,
    include_feature_selection: bool = True,
    categorical_indices: Optional[List[int]] = None
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create objective function for PSO-GWO optimization.

    Args:
        X_train: Training features
        y_train: Training labels
        model_factory: Function to create model instances
        calculator: ObjectiveCalculator instance
        cv_splits: Pre-computed CV splits
        param_bounds: Parameter bounds
        param_types: Parameter types
        n_features: Number of features
        include_feature_selection: Whether to include feature selection
        categorical_indices: Indices of categorical features (for deep models)

    Returns:
        Objective function that takes position and returns objectives
    """
    # Create dimension mapping
    lower, upper, mapping = create_search_space(
        param_bounds, n_features, include_feature_mask=include_feature_selection
    )

    def objective_fn(position: np.ndarray) -> np.ndarray:
        # Decode position
        params, feature_mask = decode_position(position, mapping, param_types)

        # Apply feature mask
        if include_feature_selection and feature_mask.sum() > 0:
            selected_features = feature_mask.astype(bool)
            X_masked = X_train[:, selected_features]
            n_selected = int(feature_mask.sum())

            # Adjust categorical indices for masked features
            if categorical_indices:
                cat_idx_masked = []
                original_idx = 0
                for i, selected in enumerate(selected_features):
                    if selected:
                        if i in categorical_indices:
                            cat_idx_masked.append(original_idx)
                        original_idx += 1
            else:
                cat_idx_masked = None
        else:
            X_masked = X_train
            n_selected = n_features
            cat_idx_masked = categorical_indices

        # Cross-validation evaluation
        all_objectives = []

        for train_idx, val_idx in cv_splits:
            X_tr, X_val = X_masked[train_idx], X_masked[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            try:
                # Create and train model
                model = model_factory(params)

                # For deep models, pass categorical indices
                if hasattr(model, '_categorical_indices_hint'):
                    model.fit(X_tr, y_tr, X_val, y_val, categorical_features=cat_idx_masked)
                else:
                    model.fit(X_tr, y_tr, X_val, y_val)

                # Predict
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)

                # Compute objectives
                objectives = calculator.compute_objectives(
                    y_val, y_pred, y_proba,
                    n_features=n_selected,
                    model_complexity=model.get_model_complexity()
                )
                all_objectives.append(objectives)

            except Exception as e:
                logger.warning(f"CV fold failed: {e}")
                # Return worst-case objectives
                all_objectives.append(np.array([10.0, 1.0, 1.0, 1.0]))

        # Average across folds
        return np.mean(all_objectives, axis=0)

    return objective_fn, lower, upper, mapping


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(config_path: str):
    """
    Run the complete forecasting pipeline.

    Args:
        config_path: Path to configuration YAML file
    """
    # Load configuration
    config = load_config(config_path)

    # Setup
    setup_logging(
        level=get_config_value(config, 'logging', 'level', default='INFO'),
        log_file=get_config_value(config, 'output', 'logs', default='outputs/logs') + '/pipeline.log'
    )

    set_seed(get_config_value(config, 'reproducibility', 'global_seed', default=42))

    logger.info("=" * 60)
    logger.info("GROUNDWATER QUALITY FORECASTING PIPELINE")
    logger.info("=" * 60)

    # Create output directories
    output_dir = Path(get_config_value(config, 'output', 'base_dir', default='outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = output_dir / 'checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # STAGE A: Data Ingestion + Harmonization
    # =========================================================================
    logger.info("\n[STAGE A] Data Ingestion + Harmonization")

    data_config = config.get('data', {})
    base_dir = data_config.get('base_dir')
    files = data_config.get('files', {})

    # Convert year keys to integers
    files_int = {int(k): v for k, v in files.items()}

    ingestion = DataIngestion(base_dir=base_dir, files=files_int)
    raw_data = ingestion.load_all()

    harmonizer = DataHarmonizer()
    data = harmonizer.harmonize_all(raw_data)

    logger.info(f"Loaded and harmonized data: {list(data.keys())}")

    # =========================================================================
    # STAGE B: Location ID Resolution + Transition Building
    # =========================================================================
    logger.info("\n[STAGE B] Transition Building")

    location_keys = data_config.get('location_keys', ['district', 'mandal', 'village'])
    transition_builder = TransitionBuilder(location_keys=location_keys)

    transitions = transition_builder.build_all_transitions(data, target_col='Classification')
    logger.info(f"Built transitions: {list(transitions.keys())}")

    # =========================================================================
    # STAGE C: Data Quality Validation
    # =========================================================================
    logger.info("\n[STAGE C] Data Quality Validation")

    dq_config = config.get('data_quality', {})
    validator = DataQualityValidator(
        expected_columns=dq_config.get('expected_columns', []),
        high_missing_threshold=dq_config.get('high_missing_threshold', 0.3),
        outlier_method=dq_config.get('outlier_method', 'iqr'),
        range_checks=dq_config.get('range_checks', {})
    )

    validation_results = validator.validate_all(data)

    # =========================================================================
    # STAGE D: Data Cleaning + Preprocessing
    # =========================================================================
    logger.info("\n[STAGE D] Data Cleaning + Preprocessing")

    label_config = config.get('labels', {})
    cleaner = DataCleaner(
        label_typo_mapping=label_config.get('typo_mapping', {}),
        rare_class_handling=dq_config.get('cleaning', {}).get('handle_rare_classes', 'merge'),
        rare_class_threshold=dq_config.get('cleaning', {}).get('rare_class_threshold', 5)
    )

    # Clean data (fit on 2018)
    cleaned_data = cleaner.clean_all(data, fit_year=2018)
    label_encoder = cleaner.get_label_encoder()

    logger.info(f"Label encoder: {label_encoder}")

    # Build transitions from cleaned data
    transitions = transition_builder.build_all_transitions(cleaned_data, target_col='Classification')

    # Temporal split
    eval_config = config.get('evaluation', {})
    splitter = TemporalSplitter(
        train_transitions=eval_config.get('temporal_split', {}).get('train_transitions', ['2018_2019']),
        test_transitions=eval_config.get('temporal_split', {}).get('test_transitions', ['2019_2020'])
    )

    train_df, test_df = splitter.split(transitions)
    logger.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    # Setup preprocessing pipeline
    preproc_config = config.get('preprocessing', {})
    pipeline = PreprocessingPipeline(
        numeric_features=preproc_config.get('numeric_features', []),
        categorical_features=preproc_config.get('categorical_features', []),
        spatial_features=preproc_config.get('spatial_features', []),
        scaling_method=preproc_config.get('scaling', {}).get('method', 'standard'),
        target_column='Classification_target'
    )

    # Fit on training, transform both
    X_train, y_train, feature_names = pipeline.fit_transform(train_df, scale_features=True)
    X_test, y_test, _ = pipeline.transform(test_df, scale_features=True)

    # Handle missing targets
    train_mask = y_train >= 0
    test_mask = y_test >= 0 if y_test is not None else np.ones(len(X_test), dtype=bool)

    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]

    logger.info(f"Final shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Classes: {np.unique(y_train)}")

    # Get categorical feature indices
    categorical_indices = pipeline.get_categorical_indices() if hasattr(pipeline, 'get_categorical_indices') else []

    # =========================================================================
    # STAGE E: Setup for Optimization
    # =========================================================================
    logger.info("\n[STAGE E] Setup for Optimization")

    # Setup imbalance handling
    imbalance_config = config.get('imbalance', {})
    imbalance_handler = ImbalanceHandler(
        strategy=imbalance_config.get('strategy', 'class_weights'),
        weight_method=imbalance_config.get('class_weights', {}).get('method', 'balanced')
    )
    imbalance_handler.fit(y_train)
    class_weights = imbalance_handler.get_class_weights()

    n_classes = len(np.unique(y_train))
    n_features = X_train.shape[1]
    random_state = get_config_value(config, 'reproducibility', 'global_seed', default=42)

    # Setup metrics calculator
    ordinal_mapping = get_ordinal_mapping(label_encoder)
    idx_to_label = {v: k for k, v in label_encoder.items()}
    high_risk_idx = get_high_risk_indices(label_encoder)

    # Setup objective calculator
    obj_config = config.get('objectives', {})
    obj_weights = config.get('mcdm', {}).get('objective_weights', {})

    objective_calculator = ObjectiveCalculator(
        label_to_ordinal=ordinal_mapping,
        idx_to_label=idx_to_label,
        high_risk_indices=high_risk_idx,
        objective_weights={
            'ordinal_distance': obj_weights.get('ordinal_distance', 0.25),
            'severe_fnr': obj_weights.get('severe_fnr', 0.35),
            'macro_f1_complement': obj_weights.get('macro_f1', 0.25),
            'complexity': obj_weights.get('complexity', 0.15)
        }
    )

    # Setup cross-validation splits
    cv_config = eval_config.get('inner_cv', {})
    n_splits = cv_config.get('n_splits', 5)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_splits = list(cv.split(X_train, y_train))

    # Get optimization config
    opt_config = config.get('optimization', {})
    population_size = opt_config.get('population_size', 30)
    max_iterations = opt_config.get('max_iterations', 50)

    # =========================================================================
    # STAGE F: PSO-GWO Optimization for Each Model
    # =========================================================================
    logger.info("\n[STAGE F] PSO-GWO Multi-Objective Optimization")

    model_configs = config.get('models', {})
    optimization_results = {}  # Store Pareto fronts per model
    model_factories = {}  # Store factory functions
    position_decoders = {}  # Store decoder functions

    # Define models to optimize
    models_to_optimize = []

    if model_configs.get('catboost', {}).get('enabled', True):
        models_to_optimize.append(('CatBoost', 'catboost'))

    if model_configs.get('lightgbm', {}).get('enabled', True):
        models_to_optimize.append(('LightGBM', 'lightgbm'))

    if DEEP_MODELS_AVAILABLE:
        if model_configs.get('gru', {}).get('enabled', True):
            models_to_optimize.append(('GRU', 'gru'))

        if model_configs.get('lstm', {}).get('enabled', True):
            models_to_optimize.append(('LSTM', 'lstm'))
    else:
        logger.warning("PyTorch not available - skipping GRU and LSTM models")

    # Run optimization for each model
    for model_name, model_key in models_to_optimize:
        logger.info(f"\n--- Optimizing {model_name} ---")

        # Get search space
        if model_key == 'catboost':
            param_bounds, param_types = get_catboost_search_space(n_features)
            factory = create_catboost_factory(n_classes, class_weights, random_state)
            cat_idx = None
        elif model_key == 'lightgbm':
            param_bounds, param_types = get_lightgbm_search_space(n_features)
            factory = create_lightgbm_factory(n_classes, class_weights, random_state)
            cat_idx = None
        elif model_key == 'gru':
            param_bounds, param_types = get_gru_search_space(n_features)
            factory = create_gru_factory(n_classes, class_weights, random_state, categorical_indices)
            cat_idx = categorical_indices
        elif model_key == 'lstm':
            param_bounds, param_types = get_lstm_search_space(n_features)
            factory = create_lstm_factory(n_classes, class_weights, random_state, categorical_indices)
            cat_idx = categorical_indices
        else:
            continue

        model_factories[model_name] = factory

        # Create objective function
        obj_fn, lower, upper, mapping = create_objective_function(
            X_train, y_train, factory, objective_calculator,
            cv_splits, param_bounds, param_types, n_features,
            include_feature_selection=True,
            categorical_indices=cat_idx
        )

        # Create position decoder for this model
        def make_decoder(param_types_local, mapping_local):
            def decoder(position):
                return decode_position(position, mapping_local, param_types_local)
            return decoder

        position_decoders[model_name] = make_decoder(param_types, mapping)

        # Initialize PSO-GWO optimizer
        optimizer = PSOGWO(
            population_size=population_size,
            max_iterations=max_iterations,
            w=opt_config.get('pso', {}).get('w', 0.7),
            c1=opt_config.get('pso', {}).get('c1', 1.5),
            c2=opt_config.get('pso', {}).get('c2', 1.5),
            a_start=opt_config.get('gwo', {}).get('a_start', 2.0),
            a_end=opt_config.get('gwo', {}).get('a_end', 0.0),
            hybrid_weight=opt_config.get('hybrid_weight', 0.5),
            random_state=random_state
        )

        optimizer.initialize(lower, upper)

        # Run optimization
        result = optimizer.optimize(obj_fn, verbose=True)

        optimization_results[model_name] = result

        logger.info(f"{model_name} optimization complete:")
        logger.info(f"  Best objectives: {result.best_objectives}")
        logger.info(f"  Pareto front size: {len(result.pareto_front)}")
        logger.info(f"  Total evaluations: {result.n_evaluations}")

    # =========================================================================
    # STAGE G: VIKOR Model Selection (Two-Level)
    # =========================================================================
    logger.info("\n[STAGE G] VIKOR Model Selection")

    mcdm_config = config.get('mcdm', {})
    vikor_v = mcdm_config.get('vikor', {}).get('v', 0.5)
    obj_weights_array = np.array([
        obj_weights.get('ordinal_distance', 0.25),
        obj_weights.get('severe_fnr', 0.35),
        obj_weights.get('macro_f1', 0.25),
        obj_weights.get('complexity', 0.15)
    ])

    model_selector = ModelSelector(
        vikor_v=vikor_v,
        objective_weights=obj_weights_array
    )

    # Level 1: Select best configuration per model
    best_configs = {}
    level1_rankings = {}

    for model_name, opt_result in optimization_results.items():
        pareto_front = opt_result.pareto_front

        if not pareto_front:
            logger.warning(f"Empty Pareto front for {model_name}, skipping")
            continue

        best_idx, vikor_result = model_selector.select_best_config(pareto_front, model_name)
        level1_rankings[model_name] = vikor_result

        # Get best configuration
        best_position, best_cv_objectives = pareto_front[best_idx]
        params, feature_mask = position_decoders[model_name](best_position)

        logger.info(f"\n{model_name} Best Configuration:")
        logger.info(f"  Parameters: {params}")
        logger.info(f"  Selected features: {int(feature_mask.sum())}/{n_features}")
        logger.info(f"  CV Objectives: {best_cv_objectives}")

        best_configs[model_name] = ConfigurationResult(
            model_name=model_name,
            params=params,
            feature_mask=feature_mask,
            cv_objectives=best_cv_objectives,
            cv_std=np.zeros_like(best_cv_objectives),
            test_objectives=None  # Will be computed below
        )

    # Evaluate best configurations on test set
    logger.info("\n--- Evaluating Best Configurations on Test Set ---")

    final_models = {}  # Store trained models for later use

    for model_name, config_result in best_configs.items():
        params = config_result.params
        feature_mask = config_result.feature_mask

        # Apply feature mask
        if feature_mask is not None and feature_mask.sum() > 0:
            selected = feature_mask.astype(bool)
            X_train_selected = X_train[:, selected]
            X_test_selected = X_test[:, selected]
            n_selected = int(feature_mask.sum())

            # Adjust categorical indices
            if model_name in ['GRU', 'LSTM'] and categorical_indices:
                cat_idx_selected = []
                original_idx = 0
                for i, is_selected in enumerate(selected):
                    if is_selected:
                        if i in categorical_indices:
                            cat_idx_selected.append(original_idx)
                        original_idx += 1
            else:
                cat_idx_selected = None
        else:
            X_train_selected = X_train
            X_test_selected = X_test
            n_selected = n_features
            cat_idx_selected = categorical_indices if model_name in ['GRU', 'LSTM'] else None

        # Train model with best params
        model = model_factories[model_name](params)

        if hasattr(model, '_categorical_indices_hint'):
            model.fit(X_train_selected, y_train, X_test_selected, y_test,
                     categorical_features=cat_idx_selected)
        else:
            model.fit(X_train_selected, y_train, X_test_selected, y_test)

        final_models[model_name] = (model, selected if feature_mask is not None else None)

        # Evaluate on test set
        y_pred = model.predict(X_test_selected)
        y_proba = model.predict_proba(X_test_selected)

        test_objectives = objective_calculator.compute_objectives(
            y_test, y_pred, y_proba,
            n_features=n_selected,
            model_complexity=model.get_model_complexity()
        )

        config_result.test_objectives = test_objectives

        logger.info(f"\n{model_name} Test Results:")
        logger.info(f"  Test Objectives: {test_objectives}")

    # Level 2: Select best model among best-configured models
    logger.info("\n--- Level 2: Cross-Model Selection ---")

    final_selection = model_selector.select_best_model(best_configs)
    final_selection.level1_rankings = level1_rankings

    logger.info(f"\nBest Model: {final_selection.best_model_name}")
    logger.info(f"Model Rankings: {[list(best_configs.keys())[i] for i in final_selection.level2_ranking.rankings]}")

    # Generate comparison table
    comparison_df = model_selector.generate_comparison_table(
        final_selection,
        objective_names=['ordinal_distance', 'severe_fnr', 'macro_f1_complement', 'complexity']
    )
    logger.info(f"\nModel Comparison:\n{comparison_df.to_string()}")

    # =========================================================================
    # STAGE H: SHAP Explainability
    # =========================================================================
    logger.info("\n[STAGE H] SHAP Explainability")

    explain_config = config.get('explainability', {})
    best_model_name = final_selection.best_model_name
    best_model, best_feature_mask = final_models[best_model_name]

    # Apply feature mask for SHAP
    if best_feature_mask is not None:
        X_test_shap = X_test[:, best_feature_mask]
        selected_feature_names = [f for i, f in enumerate(feature_names) if best_feature_mask[i]]
    else:
        X_test_shap = X_test
        selected_feature_names = feature_names

    shap_results = None

    if best_model_name in ['CatBoost', 'LightGBM']:
        # TreeSHAP for tree models
        if explain_config.get('tree_shap', {}).get('enabled', True):
            try:
                max_samples = explain_config.get('tree_shap', {}).get('max_samples', 1000)
                shap_explainer = TreeSHAPExplainer(
                    max_samples=max_samples,
                    top_k_features=10
                )

                X_sample = X_test_shap[:min(max_samples, len(X_test_shap))]
                shap_result_obj = shap_explainer.explain(best_model, X_sample, selected_feature_names)

                importance_dict = dict(zip(
                    shap_result_obj.global_importance['feature'],
                    shap_result_obj.global_importance['importance']
                ))
                shap_results = {
                    'method': 'TreeSHAP',
                    'shap_result': shap_result_obj,
                    'values': shap_result_obj.shap_values,
                    'importance': importance_dict
                }

                logger.info(f"Top SHAP features: {list(importance_dict.items())[:10]}")
            except Exception as e:
                logger.warning(f"TreeSHAP failed: {e}")

    elif best_model_name in ['GRU', 'LSTM']:
        # Surrogate SHAP for deep models
        if explain_config.get('surrogate', {}).get('enabled', True):
            try:
                surrogate_explainer = SurrogateExplainer(
                    surrogate_type=explain_config.get('surrogate', {}).get('surrogate_model', 'lightgbm')
                )

                X_sample = X_test_shap[:min(500, len(X_test_shap))]
                surrogate_result = surrogate_explainer.explain(best_model, X_sample, selected_feature_names)

                importance_dict = dict(zip(
                    surrogate_result.shap_result.global_importance['feature'],
                    surrogate_result.shap_result.global_importance['importance']
                ))
                shap_results = {
                    'method': 'Surrogate SHAP',
                    'shap_result': surrogate_result.shap_result,
                    'values': surrogate_result.shap_result.shap_values,
                    'importance': importance_dict,
                    'fidelity': surrogate_result.fidelity_metrics
                }

                logger.info(f"Surrogate fidelity: {surrogate_result.fidelity_metrics}")
                logger.info(f"Top SHAP features: {list(importance_dict.items())[:10]}")
            except Exception as e:
                logger.warning(f"Surrogate SHAP failed: {e}")

    # =========================================================================
    # STAGE I: Scenario Simulation
    # =========================================================================
    logger.info("\n[STAGE I] Scenario Simulation")

    scenario_config = config.get('scenarios', {})
    scenario_results = {}

    try:
        feature_mapping = {name: i for i, name in enumerate(selected_feature_names)}
        scenario_engine = ScenarioEngine(
            feature_mapping=feature_mapping,
            high_risk_indices=high_risk_idx
        )

        # Run TDS scenarios
        if 'tds' in scenario_config and 'TDS' in feature_mapping:
            for pct in scenario_config['tds'].get('perturbations', [0.1, 0.2]):
                result = scenario_engine.simulate_percentage_change(
                    best_model, X_test_shap, 'TDS', pct, direction='increase'
                )
                scenario_name = f"TDS_+{int(pct*100)}%"
                scenario_results[scenario_name] = {
                    'risk_change': float(np.mean(result.high_risk_prob_change)),
                    'scenario_name': result.scenario_name,
                    'pct_class_changed': float(
                        np.mean(result.baseline_predictions != result.scenario_predictions) * 100
                    )
                }
                logger.info(f"{scenario_name}: Risk change = {scenario_results[scenario_name]['risk_change']:.4f}")

        # Run SAR scenarios
        if 'sar' in scenario_config and 'SAR' in feature_mapping:
            for pct in scenario_config['sar'].get('perturbations', [0.1]):
                result = scenario_engine.simulate_percentage_change(
                    best_model, X_test_shap, 'SAR', pct, direction='increase'
                )
                scenario_name = f"SAR_+{int(pct*100)}%"
                scenario_results[scenario_name] = {
                    'risk_change': float(np.mean(result.high_risk_prob_change)),
                    'scenario_name': result.scenario_name,
                    'pct_class_changed': float(
                        np.mean(result.baseline_predictions != result.scenario_predictions) * 100
                    )
                }
                logger.info(f"{scenario_name}: Risk change = {scenario_results[scenario_name]['risk_change']:.4f}")

    except Exception as e:
        logger.warning(f"Scenario simulation failed: {e}")

    # =========================================================================
    # STAGE J: Paper Outputs
    # =========================================================================
    logger.info("\n[STAGE J] Generating Paper Outputs")

    paper_output_dir = Path(get_config_value(config, 'output', 'paper_outputs', default='outputs/paper_outputs'))
    paper_output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare results DataFrame
    results = []
    metrics_calc = MetricsCalculator(
        label_to_ordinal=ordinal_mapping,
        idx_to_label=idx_to_label,
        high_risk_indices=high_risk_idx
    )

    for model_name, (model, feature_mask) in final_models.items():
        X_eval = X_test[:, feature_mask] if feature_mask is not None else X_test
        y_pred = model.predict(X_eval)
        y_proba = model.predict_proba(X_eval)

        metrics = metrics_calc.compute_all(y_test, y_pred, y_proba)
        metrics['Model'] = model_name
        metrics['Rank'] = int(np.where(final_selection.level2_ranking.rankings ==
                                        list(best_configs.keys()).index(model_name))[0][0] + 1)
        metrics['VIKOR_Q'] = final_selection.level2_ranking.Q[
            list(best_configs.keys()).index(model_name)
        ]
        results.append(metrics)

    results_df = pd.DataFrame(results).sort_values('Rank')

    # Generate figures
    fig_gen = FigureGenerator(output_dir=paper_output_dir / 'figures')

    # F1: Flowchart
    fig_gen.create_mermaid_flowchart()

    # F2: Class distribution
    fig_gen.f2_class_distribution(cleaned_data, target_col='Classification')

    # F3: Temporal schematic
    fig_gen.f3_temporal_forecasting_schematic()

    # F4: Model comparison (all 4 models) — show decision-relevant metrics + VIKOR winner
    fig_gen.f4_model_comparison(
        results_df,
        metrics=['macro_f1', 'severe_fnr', 'ordinal_distance_mean'],
        best_model=best_model_name
    )

    # F5: Pareto fronts
    try:
        fig_gen.f5_pareto_fronts(optimization_results)
    except Exception as e:
        logger.warning(f"Pareto front figure failed: {e}")

    # F6: VIKOR rankings
    try:
        fig_gen.f6_vikor_rankings(comparison_df)
    except Exception as e:
        logger.warning(f"VIKOR ranking figure failed: {e}")

    # F7: SHAP summary + extended SHAP figures (F7b beeswarm, F7d dependence)
    if shap_results:
        try:
            fig_gen.f7_shap_summary(shap_results['importance'], selected_feature_names)
        except Exception as e:
            logger.warning(f"SHAP summary figure failed: {e}")

        if 'shap_result' in shap_results:
            try:
                class_names = [idx_to_label.get(i, str(i)) for i in sorted(idx_to_label.keys())]
                fig_gen.generate_all_shap_figures(
                    shap_results['shap_result'],
                    X_test_shap,
                    high_risk_importance=None,
                    class_names=class_names,
                    top_k=min(15, len(selected_feature_names))
                )
            except Exception as e:
                logger.warning(f"Extended SHAP figures failed: {e}")

    # F4b: Confusion matrix for best model
    try:
        y_pred_best = best_model.predict(X_test_shap)
        class_names_list = [idx_to_label.get(i, str(i)) for i in sorted(idx_to_label.keys())]
        fig_gen.f4b_confusion_matrix(y_test, y_pred_best, class_names_list, model_name=best_model_name)
    except Exception as e:
        logger.warning(f"Confusion matrix figure failed: {e}")

    # F4c: Per-class metrics for best model
    try:
        metrics_best = metrics_calc.compute_all(y_test, y_pred_best, best_model.predict_proba(X_test_shap))
        per_class = metrics_calc.get_per_class_metrics(y_test, y_pred_best)
        if per_class:
            labeled_per_class = {idx_to_label.get(k, str(k)): v for k, v in per_class.items()}
            fig_gen.f4c_per_class_metrics(labeled_per_class, model_name=best_model_name)
    except Exception as e:
        logger.warning(f"Per-class metrics figure failed: {e}")

    # F4d: Severity comparison across all models
    try:
        fig_gen.f4d_severity_comparison(results_df)
    except Exception as e:
        logger.warning(f"Severity comparison figure failed: {e}")

    # ROC curves for best model
    try:
        y_proba_best = best_model.predict_proba(X_test_shap)
        class_names_roc = [idx_to_label.get(i, str(i)) for i in sorted(idx_to_label.keys())]
        fig_gen.f_roc_curves(y_test, y_proba_best, class_names_roc, model_name=best_model_name)
    except Exception as e:
        logger.warning(f"ROC curves figure failed: {e}")

    # Precision-Recall curves for best model
    try:
        fig_gen.f_pr_curves(y_test, y_proba_best, class_names_roc, model_name=best_model_name)
    except Exception as e:
        logger.warning(f"PR curves figure failed: {e}")

    # Learning curves for deep models
    if best_model_name in ['LSTM', 'GRU'] and hasattr(best_model, 'training_history'):
        try:
            fig_gen.f_learning_curves(best_model.training_history, model_name=best_model_name)
        except Exception as e:
            logger.warning(f"Learning curves figure failed: {e}")

    # Generate tables
    table_gen = TableGenerator(output_dir=paper_output_dir / 'tables')

    # T1: Dataset overview
    table_gen.t1_dataset_overview(cleaned_data)

    # T2: Label mapping
    table_gen.t2_label_mapping(label_encoder, HIGH_RISK_CLASSES)

    # T3: Hyperparameter bounds
    try:
        table_gen.t3_hyperparameter_bounds(config.get('models', {}))
    except Exception as e:
        logger.warning(f"Hyperparameter table failed: {e}")

    # T4: Best configurations
    try:
        table_gen.t4_best_configurations(best_configs)
    except Exception as e:
        logger.warning(f"Best config table failed: {e}")

    # T5: Test performance
    results_df.to_csv(paper_output_dir / 'tables' / 'model_results.csv', index=False)

    # T6: SHAP drivers
    if shap_results:
        pd.DataFrame(list(shap_results['importance'].items()),
                    columns=['Feature', 'Importance']).to_csv(
            paper_output_dir / 'tables' / 'shap_importance.csv', index=False
        )

    # T7: Scenario outcomes
    if scenario_results:
        pd.DataFrame([
            {'Scenario': k, **v} for k, v in scenario_results.items()
        ]).to_csv(paper_output_dir / 'tables' / 'scenario_results.csv', index=False)

    # Save comparison table
    comparison_df.to_csv(paper_output_dir / 'tables' / 'model_comparison_vikor.csv', index=False)

    # Generate insights
    insights_gen = ManagerialInsights(output_dir=paper_output_dir)

    findings = insights_gen.generate_key_findings({
        'best_model': final_selection.best_model_name,
        'test_metrics': results[0] if results else {},
        'n_models_compared': len(final_models),
        'optimization_method': 'PSO-GWO',
        'selection_method': 'VIKOR MCDM'
    })

    top_features = list(shap_results['importance'].keys())[:5] if shap_results else selected_feature_names[:5]
    recommendations = insights_gen.generate_recommendations(
        shap_top_features=top_features,
        vulnerable_districts=[],
        high_impact_scenarios=list(scenario_results.keys())[:3] if scenario_results else []
    )

    insights_gen.save_insights(findings, recommendations)

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Best Model: {final_selection.best_model_name}")
    logger.info(f"Outputs saved to: {paper_output_dir}")

    return results_df, final_selection


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Groundwater Quality Forecasting Pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/main.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        results, selection = run_pipeline(args.config)
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"\nBest Model: {selection.best_model_name}")
        print("\nModel Comparison:")
        print(results.to_string())
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
