"""
Main Pipeline Runner
====================

Single entrypoint for the groundwater quality forecasting framework.

Pipeline Stages:
    A. Data Ingestion + Harmonization
    B. Transition Building
    C. Data Quality Validation
    D. Preprocessing + Imbalance Handling
    E. PSO-GWO Multi-Objective Optimization (for all 4 models)
    F. VIKOR Model Selection (two-level)
    G. SHAP Explainability
    H. Scenario Simulation
    I. Paper Outputs (figures, tables, animations)

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

def create_catboost_factory(n_classes: int, class_weights: Dict[int, float], random_state: int,
                            categorical_indices: Optional[List[int]] = None):
    """Factory function for CatBoost models."""
    def factory(params: Dict[str, Any]) -> CatBoostForecaster:
        config = ForecasterConfig(
            params={
                'iterations': int(params.get('iterations', 5000)),
                'depth': int(params.get('depth', 6)),
                'learning_rate': params.get('learning_rate', 0.003),
                'l2_leaf_reg': params.get('l2_leaf_reg', 3),
                'min_data_in_leaf': int(params.get('min_data_in_leaf', 20)),
                'early_stopping_rounds': 100
            },
            n_classes=n_classes,
            class_weights=class_weights,
            random_state=random_state
        )
        model = CatBoostForecaster(config=config)
        # Mark for categorical feature passing (same pattern as GRU/LSTM)
        model._categorical_indices_hint = categorical_indices or []
        return model
    return factory


def create_lightgbm_factory(n_classes: int, class_weights: Dict[int, float], random_state: int):
    """Factory function for LightGBM models."""
    def factory(params: Dict[str, Any]) -> LightGBMForecaster:
        config = ForecasterConfig(
            params={
                'n_estimators': int(params.get('n_estimators', 2000)),
                'max_depth': int(params.get('max_depth', 6)),
                'learning_rate': params.get('learning_rate', 0.01),
                'num_leaves': int(params.get('num_leaves', 31)),
                'min_child_samples': int(params.get('min_child_samples', 20)),
                'reg_alpha': params.get('reg_alpha', 0.1),
                'reg_lambda': params.get('reg_lambda', 0.1),
                'early_stopping_rounds': 100
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
    """Get search space for CatBoost (Sample-1 inspired: high iters, low LR)."""
    param_bounds = {
        'iterations': (1000, 14000),           # up to 14k like Sample 1
        'depth': (4, 8),
        'learning_rate': (-3.5, -0.5),         # log scale: 0.0003 to 0.316
        'l2_leaf_reg': (1, 10),
        'min_data_in_leaf': (10, 30),          # prevents overfitting on small data
    }
    param_types = {
        'iterations': 'int',
        'depth': 'int',
        'learning_rate': 'log',
        'l2_leaf_reg': 'float',
        'min_data_in_leaf': 'int',
    }
    return param_bounds, param_types


def get_lightgbm_search_space(n_features: int):
    """Get search space for LightGBM (extended for small-dataset slow learning)."""
    param_bounds = {
        'n_estimators': (500, 5000),           # higher ceiling for slow LR
        'max_depth': (4, 10),
        'learning_rate': (-3.0, -0.5),         # log scale: 0.001 to 0.316
        'num_leaves': (15, 127),
        'min_child_samples': (5, 50),
        'reg_alpha': (0.0, 1.0),
        'reg_lambda': (0.0, 1.0),
    }
    param_types = {
        'n_estimators': 'int',
        'max_depth': 'int',
        'learning_rate': 'log',
        'num_leaves': 'int',
        'min_child_samples': 'int',
        'reg_alpha': 'float',
        'reg_lambda': 'float',
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
    categorical_indices: Optional[List[int]] = None
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create objective function for PSO-GWO optimization.
    Feature selection is disabled — all features are always used.
    """
    lower, upper, mapping = create_search_space(
        param_bounds, n_features, include_feature_mask=False
    )

    def objective_fn(position: np.ndarray) -> np.ndarray:
        params, _ = decode_position(position, mapping, param_types)

        all_objectives = []

        for train_idx, val_idx in cv_splits:
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            try:
                model = model_factory(params)

                if hasattr(model, '_categorical_indices_hint'):
                    model.fit(X_tr, y_tr, X_val, y_val, categorical_features=categorical_indices)
                else:
                    model.fit(X_tr, y_tr, X_val, y_val)

                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)

                objectives = calculator.compute_objectives(
                    y_val, y_pred, y_proba,
                    n_features=n_features,
                    model_complexity=model.get_model_complexity()
                )
                all_objectives.append(objectives)

            except Exception as e:
                logger.warning(f"CV fold failed: {e}")
                all_objectives.append(np.array([10.0, 1.0, 1.0, 1.0]))

        return np.mean(all_objectives, axis=0)

    return objective_fn, lower, upper, mapping


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(config_path: str):
    """Run the complete forecasting pipeline."""
    config = load_config(config_path)

    setup_logging(
        level=get_config_value(config, 'logging', 'level', default='INFO'),
        log_file=get_config_value(config, 'output', 'logs', default='outputs/logs') + '/pipeline.log'
    )

    set_seed(get_config_value(config, 'reproducibility', 'global_seed', default=42))

    logger.info("=" * 60)
    logger.info("GROUNDWATER QUALITY FORECASTING PIPELINE")
    logger.info("=" * 60)

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
        label_typo_mapping=label_config.get('typo_mapping', {})
    )

    cleaned_data = cleaner.clean_all(data, fit_year=2018)
    label_encoder = cleaner.get_label_encoder()

    logger.info(f"Label encoder: {label_encoder}")

    transitions = transition_builder.build_all_transitions(cleaned_data, target_col='Classification')

    eval_config = config.get('evaluation', {})
    splitter = TemporalSplitter(
        train_transitions=eval_config.get('temporal_split', {}).get('train_transitions', ['2018_2019']),
        test_transitions=eval_config.get('temporal_split', {}).get('test_transitions', ['2019_2020'])
    )

    train_df, test_df = splitter.split(transitions)
    logger.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    preproc_config = config.get('preprocessing', {})
    pipeline = PreprocessingPipeline(
        numeric_features=preproc_config.get('numeric_features', []),
        categorical_features=preproc_config.get('categorical_features', []),
        spatial_features=preproc_config.get('spatial_features', []),
        scaling_method=preproc_config.get('scaling', {}).get('method', 'standard'),
        target_column='Classification_target'
    )

    X_train, y_train, feature_names = pipeline.fit_transform(train_df, scale_features=True)
    X_test, y_test, _ = pipeline.transform(test_df, scale_features=True)

    train_mask = y_train >= 0
    test_mask = y_test >= 0 if y_test is not None else np.ones(len(X_test), dtype=bool)

    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]

    logger.info(f"Final shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Classes: {np.unique(y_train)}")

    categorical_indices = pipeline.get_categorical_indices() if hasattr(pipeline, 'get_categorical_indices') else []

    # =========================================================================
    # STAGE D2: Imbalance Handling (class weights + resampling)
    # =========================================================================
    logger.info("\n[STAGE D2] Imbalance Handling")

    imbalance_config = config.get('imbalance', {})
    imbalance_handler = ImbalanceHandler(
        strategy=imbalance_config.get('strategy', 'smote_tomek'),
        weight_method=imbalance_config.get('class_weights', {}).get('method', 'balanced'),
        smote_k_neighbors=imbalance_config.get('smote', {}).get('k_neighbors', 5),
        gan_epochs=imbalance_config.get('gan', {}).get('epochs', 300),
        random_state=get_config_value(config, 'reproducibility', 'global_seed', default=42)
    )
    imbalance_handler.fit(y_train)
    class_weights = imbalance_handler.get_class_weights()
    logger.info(f"Class weights: {class_weights}")

    # Apply resampling to training data
    X_train_balanced, y_train_balanced = imbalance_handler.resample(X_train, y_train)
    logger.info(
        f"Resampled train: {len(y_train)} -> {len(y_train_balanced)} samples | "
        f"class dist: {dict(zip(*np.unique(y_train_balanced, return_counts=True)))}"
    )

    n_classes = len(np.unique(y_train_balanced))
    n_features = X_train_balanced.shape[1]
    random_state = get_config_value(config, 'reproducibility', 'global_seed', default=42)

    # =========================================================================
    # STAGE E: Setup for Optimization
    # =========================================================================
    logger.info("\n[STAGE E] Setup for Optimization")

    ordinal_mapping = get_ordinal_mapping(label_encoder)
    idx_to_label = {v: k for k, v in label_encoder.items()}
    high_risk_idx = get_high_risk_indices(label_encoder)

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

    cv_config = eval_config.get('inner_cv', {})
    n_splits = cv_config.get('n_splits', 5)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_splits = list(cv.split(X_train_balanced, y_train_balanced))

    opt_config = config.get('optimization', {})
    population_size = opt_config.get('population_size', 30)
    max_iterations = opt_config.get('max_iterations', 50)

    # =========================================================================
    # STAGE F: PSO-GWO Optimization for Each Model
    # =========================================================================
    logger.info("\n[STAGE F] PSO-GWO Multi-Objective Optimization")

    model_configs = config.get('models', {})
    optimization_results = {}
    model_factories = {}
    position_decoders = {}

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

    for model_name, model_key in models_to_optimize:
        logger.info(f"\n--- Optimizing {model_name} ---")

        if model_key == 'catboost':
            param_bounds, param_types = get_catboost_search_space(n_features)
            factory = create_catboost_factory(n_classes, class_weights, random_state,
                                              categorical_indices=categorical_indices)
            cat_idx = categorical_indices  # CatBoost now uses native categorical handling
        elif model_key == 'lightgbm':
            param_bounds, param_types = get_lightgbm_search_space(n_features)
            factory = create_lightgbm_factory(n_classes, class_weights, random_state)
            cat_idx = categorical_indices
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

        obj_fn, lower, upper, mapping = create_objective_function(
            X_train_balanced, y_train_balanced, factory, objective_calculator,
            cv_splits, param_bounds, param_types, n_features,
            categorical_indices=cat_idx
        )

        def make_decoder(param_types_local, mapping_local):
            def decoder(position):
                return decode_position(position, mapping_local, param_types_local)
            return decoder

        position_decoders[model_name] = make_decoder(param_types, mapping)

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

    best_configs = {}
    level1_rankings = {}

    for model_name, opt_result in optimization_results.items():
        pareto_front = opt_result.pareto_front

        if not pareto_front:
            logger.warning(f"Empty Pareto front for {model_name}, skipping")
            continue

        best_idx, vikor_result = model_selector.select_best_config(pareto_front, model_name)
        level1_rankings[model_name] = vikor_result

        best_position, best_cv_objectives = pareto_front[best_idx]
        params, _ = position_decoders[model_name](best_position)

        logger.info(f"\n{model_name} Best Configuration:")
        logger.info(f"  Parameters: {params}")
        logger.info(f"  CV Objectives: {best_cv_objectives}")

        best_configs[model_name] = ConfigurationResult(
            model_name=model_name,
            params=params,
            feature_mask=None,  # No feature selection
            cv_objectives=best_cv_objectives,
            cv_std=np.zeros_like(best_cv_objectives),
            test_objectives=None
        )

    # Evaluate best configurations on test set
    logger.info("\n--- Evaluating Best Configurations on Test Set ---")

    final_models = {}
    model_histories = {}  # Track training histories for animation

    for model_name, config_result in best_configs.items():
        params = config_result.params

        model = model_factories[model_name](params)

        if hasattr(model, '_categorical_indices_hint'):
            model.fit(X_train_balanced, y_train_balanced, X_test, y_test,
                      categorical_features=categorical_indices)
        else:
            model.fit(X_train_balanced, y_train_balanced, X_test, y_test)

        final_models[model_name] = model

        # Capture training history if available
        if hasattr(model, 'training_history'):
            model_histories[model_name] = model.training_history

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        test_objectives = objective_calculator.compute_objectives(
            y_test, y_pred, y_proba,
            n_features=n_features,
            model_complexity=model.get_model_complexity()
        )

        config_result.test_objectives = test_objectives

        logger.info(f"\n{model_name} Test Results:")
        logger.info(f"  Test Objectives: {test_objectives}")

    # Level 2: Select best model
    logger.info("\n--- Level 2: Cross-Model Selection ---")

    final_selection = model_selector.select_best_model(best_configs)
    final_selection.level1_rankings = level1_rankings

    logger.info(f"\nBest Model: {final_selection.best_model_name}")
    logger.info(f"Model Rankings: {[list(best_configs.keys())[i] for i in final_selection.level2_ranking.rankings]}")

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
    best_model = final_models[best_model_name]

    shap_results = None

    if best_model_name in ['CatBoost', 'LightGBM']:
        if explain_config.get('tree_shap', {}).get('enabled', True):
            try:
                max_samples = explain_config.get('tree_shap', {}).get('max_samples', 1000)
                shap_explainer = TreeSHAPExplainer(
                    max_samples=max_samples,
                    top_k_features=10
                )

                X_sample = X_test[:min(max_samples, len(X_test))]
                shap_result_obj = shap_explainer.explain(best_model, X_sample, feature_names)

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
        if explain_config.get('surrogate', {}).get('enabled', True):
            try:
                surrogate_explainer = SurrogateExplainer(
                    surrogate_type=explain_config.get('surrogate', {}).get('surrogate_model', 'lightgbm')
                )

                X_sample = X_test[:min(500, len(X_test))]
                surrogate_result = surrogate_explainer.explain(best_model, X_sample, feature_names)

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
    scenario_result_objects = {}  # Full ScenarioResult objects for figure generation

    try:
        feature_mapping = {name: i for i, name in enumerate(feature_names)}
        scenario_engine = ScenarioEngine(
            feature_mapping=feature_mapping,
            high_risk_indices=high_risk_idx
        )

        # Run TDS scenarios
        if 'tds' in scenario_config and 'TDS' in feature_mapping:
            for pct in scenario_config['tds'].get('perturbations', [0.1, 0.2]):
                result = scenario_engine.simulate_percentage_change(
                    best_model, X_test, 'TDS', pct, direction='increase'
                )
                scenario_name = f"TDS_+{int(pct*100)}%"
                scenario_results[scenario_name] = {
                    'risk_change': float(np.mean(result.high_risk_prob_change)),
                    'scenario_name': result.scenario_name,
                    'pct_class_changed': float(
                        np.mean(result.baseline_predictions != result.scenario_predictions) * 100
                    )
                }
                scenario_result_objects[scenario_name] = result
                logger.info(f"{scenario_name}: Risk change = {scenario_results[scenario_name]['risk_change']:.4f}")

        # Run SAR scenarios
        if 'sar' in scenario_config and 'SAR' in feature_mapping:
            for pct in scenario_config['sar'].get('perturbations', [0.1]):
                result = scenario_engine.simulate_percentage_change(
                    best_model, X_test, 'SAR', pct, direction='increase'
                )
                scenario_name = f"SAR_+{int(pct*100)}%"
                scenario_results[scenario_name] = {
                    'risk_change': float(np.mean(result.high_risk_prob_change)),
                    'scenario_name': result.scenario_name,
                    'pct_class_changed': float(
                        np.mean(result.baseline_predictions != result.scenario_predictions) * 100
                    )
                }
                scenario_result_objects[scenario_name] = result
                logger.info(f"{scenario_name}: Risk change = {scenario_results[scenario_name]['risk_change']:.4f}")

        # RSC threshold scenarios
        if 'rsc' in scenario_config and 'RSC' in feature_mapping:
            for threshold in scenario_config['rsc'].get('thresholds', [1.25, 2.5]):
                result = scenario_engine.simulate_threshold_crossing(
                    best_model, X_test, 'RSC', threshold, 'above'
                )
                scenario_name = f"RSC_above_{threshold}"
                scenario_results[scenario_name] = {
                    'risk_change': float(np.mean(result.high_risk_prob_change)),
                    'scenario_name': result.scenario_name,
                    'pct_class_changed': float(
                        np.mean(result.baseline_predictions != result.scenario_predictions) * 100
                    )
                }
                scenario_result_objects[scenario_name] = result
                logger.info(f"{scenario_name}: Risk change = {scenario_results[scenario_name]['risk_change']:.4f}")

        # If no scenario ran (features not available), run generic scenarios
        if not scenario_results:
            logger.info("Named features not found — running generic percentage scenarios on top features")
            top_feature = feature_names[0] if feature_names else None
            if top_feature and top_feature in feature_mapping:
                for pct in [0.1, 0.2, 0.3]:
                    result = scenario_engine.simulate_percentage_change(
                        best_model, X_test, top_feature, pct, direction='increase'
                    )
                    scenario_name = f"{top_feature}_+{int(pct*100)}%"
                    scenario_results[scenario_name] = {
                        'risk_change': float(np.mean(result.high_risk_prob_change)),
                        'scenario_name': result.scenario_name,
                        'pct_class_changed': float(
                            np.mean(result.baseline_predictions != result.scenario_predictions) * 100
                        )
                    }
                    scenario_result_objects[scenario_name] = result

    except Exception as e:
        logger.warning(f"Scenario simulation failed: {e}", exc_info=True)

    # =========================================================================
    # STAGE J: Paper Outputs
    # =========================================================================
    logger.info("\n[STAGE J] Generating Paper Outputs")

    paper_output_dir = Path(get_config_value(config, 'output', 'paper_outputs', default='outputs/paper_outputs'))
    paper_output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    metrics_calc = MetricsCalculator(
        label_to_ordinal=ordinal_mapping,
        idx_to_label=idx_to_label,
        high_risk_indices=high_risk_idx
    )

    for model_name, model in final_models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        metrics = metrics_calc.compute_all(y_test, y_pred, y_proba)
        metrics['Model'] = model_name
        metrics['Rank'] = int(np.where(final_selection.level2_ranking.rankings ==
                                        list(best_configs.keys()).index(model_name))[0][0] + 1)
        metrics['VIKOR_Q'] = final_selection.level2_ranking.Q[
            list(best_configs.keys()).index(model_name)
        ]
        results.append(metrics)

    results_df = pd.DataFrame(results).sort_values('Rank')

    fig_gen = FigureGenerator(output_dir=paper_output_dir / 'figures')

    # F1: Flowchart
    fig_gen.create_mermaid_flowchart()

    # F2: Class distribution
    fig_gen.f2_class_distribution(cleaned_data, target_col='Classification')

    # F3: Temporal schematic
    fig_gen.f3_temporal_forecasting_schematic()

    # F4: Model comparison
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

    # F7: SHAP summary
    if shap_results:
        try:
            fig_gen.f7_shap_summary(shap_results['importance'], feature_names)
        except Exception as e:
            logger.warning(f"SHAP summary figure failed: {e}")

        if 'shap_result' in shap_results:
            try:
                class_names = [idx_to_label.get(i, str(i)) for i in sorted(idx_to_label.keys())]
                fig_gen.generate_all_shap_figures(
                    shap_results['shap_result'],
                    X_test,
                    high_risk_importance=None,
                    class_names=class_names,
                    top_k=min(15, len(feature_names))
                )
            except Exception as e:
                logger.warning(f"Extended SHAP figures failed: {e}")

    # F4b: Confusion matrix for best model
    try:
        y_pred_best = best_model.predict(X_test)
        class_names_list = [idx_to_label.get(i, str(i)) for i in sorted(idx_to_label.keys())]
        fig_gen.f4b_confusion_matrix(y_test, y_pred_best, class_names_list, model_name=best_model_name)
    except Exception as e:
        logger.warning(f"Confusion matrix figure failed: {e}")

    # F4c: Per-class metrics for best model
    try:
        y_proba_best = best_model.predict_proba(X_test)
        metrics_best = metrics_calc.compute_all(y_test, y_pred_best, y_proba_best)
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

    # ROC curves
    try:
        y_proba_best = best_model.predict_proba(X_test)
        class_names_roc = [idx_to_label.get(i, str(i)) for i in sorted(idx_to_label.keys())]
        fig_gen.f_roc_curves(y_test, y_proba_best, class_names_roc, model_name=best_model_name)
    except Exception as e:
        logger.warning(f"ROC curves figure failed: {e}")

    # Precision-Recall curves
    try:
        fig_gen.f_pr_curves(y_test, y_proba_best, class_names_roc, model_name=best_model_name)
    except Exception as e:
        logger.warning(f"PR curves figure failed: {e}")

    # F8: Scenario simulation figures
    if scenario_result_objects:
        try:
            class_names_scen = [idx_to_label.get(i, str(i)) for i in sorted(idx_to_label.keys())]
            fig_gen.f8_scenario_simulation(
                scenario_result_objects,
                class_names=class_names_scen,
                idx_to_label=idx_to_label,
                high_risk_indices=high_risk_idx
            )
            logger.info("Scenario simulation figures generated")
        except Exception as e:
            logger.warning(f"Scenario figure failed: {e}", exc_info=True)

    # F9: Imbalance handling figure (before/after class distribution)
    try:
        fig_gen.f9_imbalance_handling(
            y_train, y_train_balanced,
            idx_to_label=idx_to_label,
            strategy=imbalance_handler.strategy
        )
    except Exception as e:
        logger.warning(f"Imbalance figure failed: {e}")

    # Learning curves for deep models
    if best_model_name in ['LSTM', 'GRU'] and hasattr(best_model, 'training_history'):
        try:
            fig_gen.f_learning_curves(best_model.training_history, model_name=best_model_name)
        except Exception as e:
            logger.warning(f"Learning curves figure failed: {e}")

    # Training animations for all models
    try:
        from src.reporting.animation import TrainingAnimator
        animator = TrainingAnimator(output_dir=paper_output_dir / 'figures')
        for m_name, m_history in model_histories.items():
            animator.animate_training(m_history, model_name=m_name)
        # Also animate a summary across all models with available histories
        if model_histories:
            animator.animate_multi_model_comparison(model_histories, metric='val_loss')
    except Exception as e:
        logger.warning(f"Training animation failed: {e}", exc_info=True)

    # Generate tables
    table_gen = TableGenerator(output_dir=paper_output_dir / 'tables')

    table_gen.t1_dataset_overview(cleaned_data)
    table_gen.t2_label_mapping(label_encoder, HIGH_RISK_CLASSES)

    try:
        table_gen.t3_hyperparameter_bounds(config.get('models', {}))
    except Exception as e:
        logger.warning(f"Hyperparameter table failed: {e}")

    try:
        table_gen.t4_best_configurations(best_configs)
    except Exception as e:
        logger.warning(f"Best config table failed: {e}")

    results_df.to_csv(paper_output_dir / 'tables' / 'model_results.csv', index=False)

    if shap_results:
        pd.DataFrame(list(shap_results['importance'].items()),
                     columns=['Feature', 'Importance']).to_csv(
            paper_output_dir / 'tables' / 'shap_importance.csv', index=False
        )

    if scenario_results:
        pd.DataFrame([
            {'Scenario': k, **v} for k, v in scenario_results.items()
        ]).to_csv(paper_output_dir / 'tables' / 'scenario_results.csv', index=False)

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

    top_features = list(shap_results['importance'].keys())[:5] if shap_results else feature_names[:5]
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
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
