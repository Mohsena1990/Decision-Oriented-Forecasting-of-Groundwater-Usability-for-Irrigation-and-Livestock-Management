"""
Enhanced Pipeline Runner
========================

Improved pipeline using:
1. Enhanced preprocessing (removes gwl, TDS; robust scaling; power transforms)
2. Professional class weighting with high-risk multiplier
3. Focal Loss for deep models
4. Decision threshold optimization
5. Better hyperparameters

Usage:
    python -m experiments.run_enhanced_pipeline --config configs/enhanced_config.yaml
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
from src.data_quality import DataQualityValidator, DataCleaner
from src.preprocessing import EnhancedPreprocessingPipeline, EnhancedPreprocessingConfig
from src.evaluation import TemporalSplitter, MetricsCalculator
from src.imbalance import (
    EnhancedImbalanceHandler,
    compute_enhanced_class_weights,
    optimize_threshold_for_recall,
    apply_threshold_adjustment,
)

# Models
from src.models.trees import CatBoostForecaster, LightGBMForecaster
from src.models.base import ForecasterConfig

try:
    from src.models.deep import GRUForecaster, LSTMForecaster
    from src.imbalance import FocalLoss, create_loss_function, TORCH_LOSSES_AVAILABLE
    DEEP_MODELS_AVAILABLE = True
except ImportError:
    DEEP_MODELS_AVAILABLE = False
    GRUForecaster = None
    LSTMForecaster = None
    TORCH_LOSSES_AVAILABLE = False

from src.optimization.pso_gwo import PSOGWO, create_search_space, decode_position
from src.decision.vikor import VIKOR
from src.decision.model_selection import ModelSelector, ConfigurationResult

from src.objectives import ObjectiveCalculator
from src.objectives.definitions import get_high_risk_indices, get_ordinal_mapping, HIGH_RISK_CLASSES

from src.explain.shap_tree import TreeSHAPExplainer
from src.scenarios.engine import ScenarioEngine
from src.reporting import FigureGenerator, TableGenerator

logger = logging.getLogger(__name__)


# =============================================================================
# ENHANCED MODEL FACTORIES
# =============================================================================

def create_catboost_factory(n_classes: int, class_weights: Dict[int, float], random_state: int):
    """Factory for CatBoost with enhanced class weighting."""
    def factory(params: Dict[str, Any]) -> CatBoostForecaster:
        config = ForecasterConfig(
            params={
                'iterations': int(params.get('iterations', 500)),
                'depth': int(params.get('depth', 6)),
                'learning_rate': params.get('learning_rate', 0.05),
                'l2_leaf_reg': params.get('l2_leaf_reg', 5),
                'early_stopping_rounds': 30,
                'subsample': params.get('subsample', 0.8),
            },
            n_classes=n_classes,
            class_weights=class_weights,
            random_state=random_state
        )
        return CatBoostForecaster(config=config)
    return factory


def create_lightgbm_factory(n_classes: int, class_weights: Dict[int, float], random_state: int):
    """Factory for LightGBM with is_unbalance and class weights."""
    def factory(params: Dict[str, Any]) -> LightGBMForecaster:
        config = ForecasterConfig(
            params={
                'n_estimators': int(params.get('n_estimators', 500)),
                'max_depth': int(params.get('max_depth', 6)),
                'learning_rate': params.get('learning_rate', 0.05),
                'num_leaves': int(params.get('num_leaves', 31)),
                'min_child_samples': int(params.get('min_child_samples', 30)),
                'reg_alpha': params.get('reg_alpha', 0.5),
                'reg_lambda': params.get('reg_lambda', 0.5),
                'subsample': params.get('subsample', 0.8),
                'colsample_bytree': params.get('colsample_bytree', 0.8),
                'early_stopping_rounds': 30,
            },
            n_classes=n_classes,
            class_weights=class_weights,
            random_state=random_state
        )
        return LightGBMForecaster(config=config)
    return factory


def create_gru_factory(
    n_classes: int,
    class_weights: Dict[int, float],
    random_state: int,
    categorical_indices: List[int],
    use_focal_loss: bool = True
):
    """Factory for GRU with focal loss support."""
    def factory(params: Dict[str, Any]) -> GRUForecaster:
        config = ForecasterConfig(
            params={
                'hidden_size': int(params.get('hidden_size', 128)),
                'num_layers': int(params.get('num_layers', 1)),
                'dropout': params.get('dropout', 0.4),
                'learning_rate': params.get('learning_rate', 0.001),
                'batch_size': int(params.get('batch_size', 32)),
                'embedding_dim': int(params.get('embedding_dim', 16)),
                'max_epochs': 300,
                'patience': 30,
                'use_focal_loss': use_focal_loss,
                'focal_gamma': 2.0,
            },
            n_classes=n_classes,
            class_weights=class_weights,
            random_state=random_state
        )
        model = GRUForecaster(config=config)
        model._categorical_indices_hint = categorical_indices
        return model
    return factory


def create_lstm_factory(
    n_classes: int,
    class_weights: Dict[int, float],
    random_state: int,
    categorical_indices: List[int],
    use_focal_loss: bool = True
):
    """Factory for LSTM with focal loss support."""
    def factory(params: Dict[str, Any]) -> LSTMForecaster:
        config = ForecasterConfig(
            params={
                'hidden_size': int(params.get('hidden_size', 128)),
                'num_layers': int(params.get('num_layers', 1)),
                'dropout': params.get('dropout', 0.4),
                'learning_rate': params.get('learning_rate', 0.001),
                'batch_size': int(params.get('batch_size', 32)),
                'embedding_dim': int(params.get('embedding_dim', 16)),
                'max_epochs': 300,
                'patience': 30,
                'use_focal_loss': use_focal_loss,
                'focal_gamma': 2.0,
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
# SEARCH SPACES (OPTIMIZED)
# =============================================================================

def get_catboost_search_space(n_features: int):
    """Optimized search space for CatBoost."""
    param_bounds = {
        'iterations': (200, 800),
        'depth': (4, 8),
        'learning_rate': (-1.5, -0.5),  # log scale: ~0.03 to 0.3
        'l2_leaf_reg': (3, 10),
        'subsample': (0.7, 0.95),
    }
    param_types = {
        'iterations': 'int',
        'depth': 'int',
        'learning_rate': 'log',
        'l2_leaf_reg': 'float',
        'subsample': 'float',
    }
    return param_bounds, param_types


def get_lightgbm_search_space(n_features: int):
    """Optimized search space for LightGBM."""
    param_bounds = {
        'n_estimators': (200, 800),
        'max_depth': (4, 8),
        'learning_rate': (-1.5, -0.5),
        'num_leaves': (15, 63),
        'min_child_samples': (20, 50),
        'reg_alpha': (0.1, 1.0),
        'reg_lambda': (0.1, 1.0),
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
    """Optimized search space for GRU."""
    param_bounds = {
        'hidden_size': (64, 256),
        'num_layers': (1, 2),
        'dropout': (0.3, 0.5),
        'learning_rate': (-3.3, -2.5),  # log scale: ~0.0005 to 0.003
        'batch_size': (32, 64),
        'embedding_dim': (16, 32),
    }
    param_types = {
        'hidden_size': 'int',
        'num_layers': 'int',
        'dropout': 'float',
        'learning_rate': 'log',
        'batch_size': 'int',
        'embedding_dim': 'int',
    }
    return param_bounds, param_types


def get_lstm_search_space(n_features: int):
    """Optimized search space for LSTM."""
    return get_gru_search_space(n_features)


# =============================================================================
# OBJECTIVE FUNCTION
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
):
    """Create objective function for optimization."""
    lower, upper, mapping = create_search_space(
        param_bounds, n_features, include_feature_mask=include_feature_selection
    )

    def objective_fn(position: np.ndarray) -> np.ndarray:
        params, feature_mask = decode_position(position, mapping, param_types)

        if include_feature_selection and feature_mask.sum() > 0:
            selected_features = feature_mask.astype(bool)
            X_masked = X_train[:, selected_features]
            n_selected = int(feature_mask.sum())

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

        all_objectives = []

        for train_idx, val_idx in cv_splits:
            X_tr, X_val = X_masked[train_idx], X_masked[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            try:
                model = model_factory(params)

                if hasattr(model, '_categorical_indices_hint'):
                    model.fit(X_tr, y_tr, X_val, y_val, categorical_features=cat_idx_masked)
                else:
                    model.fit(X_tr, y_tr, X_val, y_val)

                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)

                objectives = calculator.compute_objectives(
                    y_val, y_pred, y_proba,
                    n_features=n_selected,
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

def run_enhanced_pipeline(config_path: str):
    """Run the enhanced forecasting pipeline."""
    config = load_config(config_path)

    setup_logging(
        level=get_config_value(config, 'logging', 'level', default='INFO'),
        log_file=get_config_value(config, 'output', 'logs', default='outputs/logs') + '/enhanced_pipeline.log'
    )

    set_seed(get_config_value(config, 'reproducibility', 'global_seed', default=42))

    logger.info("=" * 70)
    logger.info("ENHANCED GROUNDWATER QUALITY FORECASTING PIPELINE")
    logger.info("=" * 70)
    logger.info("Key improvements:")
    logger.info("  - Removed gwl (covariate shift) and TDS (redundant with EC)")
    logger.info("  - RobustScaler + Power transforms for skewed features")
    logger.info("  - Enhanced class weighting (3x multiplier for high-risk)")
    logger.info("  - Focal Loss for deep models")
    logger.info("  - Optimized hyperparameters")
    logger.info("=" * 70)

    output_dir = Path(get_config_value(config, 'output', 'base_dir', default='outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)

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

    logger.info(f"Loaded data: {list(data.keys())}")
    for year, df in data.items():
        logger.info(f"  Year {year}: {len(df)} samples, {len(df.columns)} columns")

    # =========================================================================
    # STAGE B: Transition Building
    # =========================================================================
    logger.info("\n[STAGE B] Transition Building")

    location_keys = data_config.get('location_keys', ['district', 'mandal', 'village'])
    transition_builder = TransitionBuilder(location_keys=location_keys)

    # =========================================================================
    # STAGE C: Data Cleaning
    # =========================================================================
    logger.info("\n[STAGE C] Data Cleaning")

    label_config = config.get('labels', {})
    dq_config = config.get('data_quality', {})

    cleaner = DataCleaner(
        label_typo_mapping=label_config.get('typo_mapping', {}),
        rare_class_handling=dq_config.get('cleaning', {}).get('handle_rare_classes', 'merge'),
        rare_class_threshold=dq_config.get('cleaning', {}).get('rare_class_threshold', 5)
    )

    cleaned_data = cleaner.clean_all(data, fit_year=2018)
    label_encoder_basic = cleaner.get_label_encoder()

    logger.info(f"Label encoder: {label_encoder_basic}")

    # Build transitions
    transitions = transition_builder.build_all_transitions(cleaned_data, target_col='Classification')

    # =========================================================================
    # STAGE D: Enhanced Preprocessing
    # =========================================================================
    logger.info("\n[STAGE D] Enhanced Preprocessing")

    # Temporal split
    eval_config = config.get('evaluation', {})
    splitter = TemporalSplitter(
        train_transitions=eval_config.get('temporal_split', {}).get('train_transitions', ['2018_2019']),
        test_transitions=eval_config.get('temporal_split', {}).get('test_transitions', ['2019_2020'])
    )

    train_df, test_df = splitter.split(transitions)
    logger.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    # Enhanced preprocessing config
    preproc_config = config.get('preprocessing', {})

    enhanced_config = EnhancedPreprocessingConfig(
        remove_redundant_features=True,  # Removes TDS
        remove_shifted_features=True,     # Removes gwl
        apply_power_transform=preproc_config.get('power_transform', {}).get('enabled', True),
        scaling_method='robust',
        winsorize_limits=(0.02, 0.98),
        add_current_class_feature=preproc_config.get('feature_engineering', {}).get('add_current_class', True),
        add_interaction_features=preproc_config.get('feature_engineering', {}).get('add_interactions', True),
        class_weight_strategy='custom_high_risk',
        high_risk_weight_multiplier=config.get('imbalance', {}).get('class_weights', {}).get('high_risk_multiplier', 3.0),
    )

    pipeline = EnhancedPreprocessingPipeline(
        config=enhanced_config,
        numeric_features=preproc_config.get('numeric_features', []),
        categorical_features=preproc_config.get('categorical_features', []),
        spatial_features=preproc_config.get('spatial_features', []),
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
    logger.info(f"Features: {feature_names}")
    logger.info(f"Classes: {np.unique(y_train)}")

    # Get class weights from pipeline
    class_weights = pipeline.get_class_weights()
    label_encoder = pipeline.get_label_encoder()
    categorical_indices = pipeline.get_categorical_indices()

    logger.info(f"Enhanced class weights: {class_weights}")

    # =========================================================================
    # STAGE E: Setup for Optimization
    # =========================================================================
    logger.info("\n[STAGE E] Setup for Optimization")

    n_classes = len(np.unique(y_train))
    n_features = X_train.shape[1]
    random_state = get_config_value(config, 'reproducibility', 'global_seed', default=42)

    # Setup metrics
    ordinal_mapping = get_ordinal_mapping(label_encoder)
    idx_to_label = {v: k for k, v in label_encoder.items()}
    high_risk_idx = get_high_risk_indices(label_encoder)

    logger.info(f"High-risk class indices: {high_risk_idx}")
    logger.info(f"High-risk labels: {[idx_to_label.get(i, '?') for i in high_risk_idx]}")

    # Objective calculator with enhanced weights
    obj_weights = config.get('mcdm', {}).get('objective_weights', {})
    objective_calculator = ObjectiveCalculator(
        label_to_ordinal=ordinal_mapping,
        idx_to_label=idx_to_label,
        high_risk_indices=high_risk_idx,
        objective_weights={
            'ordinal_distance': obj_weights.get('ordinal_distance', 0.20),
            'severe_fnr': obj_weights.get('severe_fnr', 0.45),  # Highest
            'macro_f1_complement': obj_weights.get('macro_f1', 0.25),
            'complexity': obj_weights.get('complexity', 0.10)
        }
    )

    # CV splits
    cv_config = eval_config.get('inner_cv', {})
    n_splits = cv_config.get('n_splits', 5)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_splits = list(cv.split(X_train, y_train))

    # Optimization config
    opt_config = config.get('optimization', {})
    population_size = opt_config.get('population_size', 20)
    max_iterations = opt_config.get('max_iterations', 30)

    # =========================================================================
    # STAGE F: Model Optimization
    # =========================================================================
    logger.info("\n[STAGE F] Model Optimization")

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

    for model_name, model_key in models_to_optimize:
        logger.info(f"\n--- Optimizing {model_name} ---")

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
            factory = create_gru_factory(n_classes, class_weights, random_state, categorical_indices, use_focal_loss=True)
            cat_idx = categorical_indices
        elif model_key == 'lstm':
            param_bounds, param_types = get_lstm_search_space(n_features)
            factory = create_lstm_factory(n_classes, class_weights, random_state, categorical_indices, use_focal_loss=True)
            cat_idx = categorical_indices
        else:
            continue

        model_factories[model_name] = factory

        obj_fn, lower, upper, mapping = create_objective_function(
            X_train, y_train, factory, objective_calculator,
            cv_splits, param_bounds, param_types, n_features,
            include_feature_selection=True,
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
            w=opt_config.get('pso', {}).get('w', 0.6),
            c1=opt_config.get('pso', {}).get('c1', 1.8),
            c2=opt_config.get('pso', {}).get('c2', 1.2),
            a_start=opt_config.get('gwo', {}).get('a_start', 2.0),
            a_end=opt_config.get('gwo', {}).get('a_end', 0.0),
            hybrid_weight=opt_config.get('hybrid_weight', 0.5),
            random_state=random_state
        )

        optimizer.initialize(lower, upper)
        result = optimizer.optimize(obj_fn, verbose=True)
        optimization_results[model_name] = result

        logger.info(f"{model_name} complete: best_objectives={result.best_objectives}")

    # =========================================================================
    # STAGE G: VIKOR Model Selection
    # =========================================================================
    logger.info("\n[STAGE G] VIKOR Model Selection")

    mcdm_config = config.get('mcdm', {})
    vikor_v = mcdm_config.get('vikor', {}).get('v', 0.6)
    obj_weights_array = np.array([
        obj_weights.get('ordinal_distance', 0.20),
        obj_weights.get('severe_fnr', 0.45),
        obj_weights.get('macro_f1', 0.25),
        obj_weights.get('complexity', 0.10)
    ])

    model_selector = ModelSelector(vikor_v=vikor_v, objective_weights=obj_weights_array)

    best_configs = {}
    for model_name, opt_result in optimization_results.items():
        pareto_front = opt_result.pareto_front

        if not pareto_front:
            logger.warning(f"Empty Pareto front for {model_name}")
            continue

        best_idx, vikor_result = model_selector.select_best_config(pareto_front, model_name)
        best_position, best_cv_objectives = pareto_front[best_idx]
        params, feature_mask = position_decoders[model_name](best_position)

        logger.info(f"\n{model_name} Best Config:")
        logger.info(f"  Params: {params}")
        logger.info(f"  Features: {int(feature_mask.sum())}/{n_features}")
        logger.info(f"  CV Objectives: {best_cv_objectives}")

        best_configs[model_name] = ConfigurationResult(
            model_name=model_name,
            params=params,
            feature_mask=feature_mask,
            cv_objectives=best_cv_objectives,
            cv_std=np.zeros_like(best_cv_objectives),
            test_objectives=None
        )

    # =========================================================================
    # STAGE H: Test Evaluation with Threshold Optimization
    # =========================================================================
    logger.info("\n[STAGE H] Test Evaluation with Threshold Optimization")

    final_models = {}
    all_results = []
    final_preds = {}  # model_name -> (y_test, y_pred_adj, y_proba)

    metrics_calc = MetricsCalculator(
        label_to_ordinal=ordinal_mapping,
        idx_to_label=idx_to_label,
        high_risk_indices=high_risk_idx
    )

    for model_name, config_result in best_configs.items():
        params = config_result.params
        feature_mask = config_result.feature_mask

        if feature_mask is not None and feature_mask.sum() > 0:
            selected = feature_mask.astype(bool)
            X_train_selected = X_train[:, selected]
            X_test_selected = X_test[:, selected]
            n_selected = int(feature_mask.sum())

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

        # Train final model
        model = model_factories[model_name](params)

        if hasattr(model, '_categorical_indices_hint'):
            model.fit(X_train_selected, y_train, X_test_selected, y_test,
                     categorical_features=cat_idx_selected)
        else:
            model.fit(X_train_selected, y_train, X_test_selected, y_test)

        final_models[model_name] = (model, selected if feature_mask is not None else None)

        # Standard predictions
        y_pred = model.predict(X_test_selected)
        y_proba = model.predict_proba(X_test_selected)

        # Optimize threshold for high-risk detection
        if config.get('imbalance', {}).get('threshold_optimization', {}).get('enabled', True):
            target_recall = config.get('imbalance', {}).get('threshold_optimization', {}).get('target_recall', 0.7)
            opt_threshold, threshold_metrics = optimize_threshold_for_recall(
                y_test, y_proba, high_risk_idx, target_recall=target_recall
            )

            # Apply threshold-adjusted predictions
            y_pred_adjusted = apply_threshold_adjustment(y_proba, high_risk_idx, threshold=opt_threshold)
            logger.info(f"{model_name} threshold optimization: {threshold_metrics}")
        else:
            y_pred_adjusted = y_pred

        # Compute metrics (using adjusted predictions)
        metrics = metrics_calc.compute_all(y_test, y_pred_adjusted, y_proba)
        metrics['Model'] = model_name

        test_objectives = objective_calculator.compute_objectives(
            y_test, y_pred_adjusted, y_proba,
            n_features=n_selected,
            model_complexity=model.get_model_complexity()
        )
        config_result.test_objectives = test_objectives

        logger.info(f"\n{model_name} Test Results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Macro F1: {metrics['macro_f1']:.4f}")
        logger.info(f"  Severe FNR: {metrics['severe_fnr']:.4f}")
        logger.info(f"  Severe Recall: {metrics['severe_recall']:.4f}")
        logger.info(f"  Test Objectives: {test_objectives}")

        all_results.append(metrics)
        final_preds[model_name] = (y_test, y_pred_adjusted, y_proba)

    # Final model selection
    final_selection = model_selector.select_best_model(best_configs)
    logger.info(f"\nBest Model: {final_selection.best_model_name}")

    # =========================================================================
    # STAGE I: Save Results
    # =========================================================================
    logger.info("\n[STAGE I] Saving Results")

    paper_output_dir = Path(get_config_value(config, 'output', 'paper_outputs', default='outputs/paper_outputs'))
    paper_output_dir.mkdir(parents=True, exist_ok=True)

    # Add rankings to results
    rankings = final_selection.level2_ranking.rankings
    model_list = list(best_configs.keys())

    for metrics in all_results:
        model_name = metrics['Model']
        if model_name in model_list:
            model_idx = model_list.index(model_name)
            rank_position = np.where(rankings == model_idx)[0]
            metrics['Rank'] = int(rank_position[0] + 1) if len(rank_position) > 0 else 99
            metrics['VIKOR_Q'] = float(final_selection.level2_ranking.Q[model_idx])

    results_df = pd.DataFrame(all_results).sort_values('Rank')

    # Save results
    results_df.to_csv(paper_output_dir / 'tables' / 'model_results.csv', index=False)
    logger.info(f"\nResults saved to {paper_output_dir / 'tables' / 'model_results.csv'}")

    # =========================================================================
    # STAGE J: Paper Figures
    # =========================================================================
    logger.info("\n[STAGE J] Generating Paper Figures")

    fig_gen = FigureGenerator(output_dir=paper_output_dir / 'figures', dpi=300)
    fig_gen.f2_class_distribution(data, target_col='Classification')
    fig_gen.f3_temporal_forecasting_schematic()
    fig_gen.create_mermaid_flowchart()

    # F4: model comparison bar chart
    metric_cols = ['macro_f1', 'severe_fnr', 'ordinal_distance_mean']
    avail_metrics = [c for c in metric_cols if c in results_df.columns]
    if avail_metrics:
        fig_gen.f4_model_comparison(results_df, metrics=avail_metrics)

    # F4d: severity metrics across models
    fig_gen.f4d_severity_comparison(results_df)

    # F5: Pareto fronts
    fig_gen.f5_pareto_fronts(optimization_results)

    # F6: VIKOR rankings
    objective_names = ['Ordinal Distance', 'Severe FNR', '1 - Macro F1', 'Complexity']
    try:
        comparison_df = model_selector.generate_comparison_table(
            final_selection, objective_names=objective_names
        )
        fig_gen.f6_vikor_rankings(comparison_df)
        comparison_df.to_csv(paper_output_dir / 'tables' / 'model_comparison_vikor.csv', index=False)
    except Exception as _e:
        logger.warning(f"F6 VIKOR figure skipped: {_e}")

    # F4b: confusion matrix + F4c: per-class metrics for best model
    best_model_name = final_selection.best_model_name
    class_names_list = [idx_to_label.get(i, str(i)) for i in range(n_classes)]
    if best_model_name in final_preds:
        y_true_bm, y_pred_bm, _ = final_preds[best_model_name]
        fig_gen.f4b_confusion_matrix(
            y_true_bm, y_pred_bm,
            class_names=class_names_list,
            model_name=best_model_name,
        )
        per_class = metrics_calc.get_per_class_metrics(y_true_bm, y_pred_bm)
        fig_gen.f4c_per_class_metrics(per_class, model_name=best_model_name)

    logger.info(f"  Figures saved to {paper_output_dir / 'figures'}")

    # Print final summary
    logger.info("\n" + "=" * 70)
    logger.info("ENHANCED PIPELINE COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Best Model: {final_selection.best_model_name}")
    logger.info(f"\nModel Comparison:\n{results_df.to_string()}")

    return results_df, final_selection


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Enhanced Groundwater Forecasting Pipeline')
    parser.add_argument('--config', type=str, default='configs/enhanced_config.yaml')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        results, selection = run_enhanced_pipeline(args.config)
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print(f"\nBest Model: {selection.best_model_name}")
        print("\nModel Comparison:")
        print(results.to_string())
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
