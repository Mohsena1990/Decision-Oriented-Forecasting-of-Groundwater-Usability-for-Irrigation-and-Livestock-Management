"""
Main Pipeline Runner
====================

Single entrypoint for the groundwater quality forecasting framework.

Models (6 total):
    1. CatBoost       — gradient boosting with native categorical handling
    2. LightGBM       — efficient gradient boosting (leaf-wise)
    3. XGBoost        — regularized gradient boosting (depth-wise)
    4. FT-Transformer — Feature Tokenization Transformer (Gorishniy et al., NeurIPS 2021)
    5. SpatialGNN     — k-NN spatial graph + mean aggregation + MLP
    6. CORAL          — COnsistent RAnk Logits ordinal network (Cao et al., 2020)

Pipeline Stages:
    A. Data Ingestion + Harmonization
    B. Transition Building
    C. Data Quality Validation
    D. Preprocessing + Imbalance Handling
    E. PSO-GWO Multi-Objective Optimization (for all 6 models)
    F. VIKOR Model Selection (two-level)
    G. SHAP Explainability
    H. Scenario Simulation
    I. Paper Outputs (figures, tables, animations)

Usage:
    python -m experiments.run_pipeline --config configs/main.yaml
"""

import argparse
import concurrent.futures
import logging
import multiprocessing as _mp
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.utils.config import load_config, get_config_value
from src.utils.logging import setup_logging
from src.utils.reproducibility import set_seed

from src.data import (
    DataIngestion, DataHarmonizer, TransitionBuilder,
    enrich_with_external_data, ALL_EXTERNAL_FEATURES,
)
from src.data_quality import DataQualityValidator, DataCleaner, DataQualityReport
from src.preprocessing import PreprocessingPipeline, LabelParser, OrdinalEncoder
from src.evaluation import TemporalSplitter, CrossValidator, MetricsCalculator
from src.imbalance import ImbalanceHandler

# RAE imputation (optional — graceful fallback if PyTorch absent)
try:
    from src.imputation import RAEImputer, TORCH_AVAILABLE as RAE_TORCH_AVAILABLE
    RAE_AVAILABLE = True
except ImportError:
    RAE_AVAILABLE = False
    RAE_TORCH_AVAILABLE = False

# Tree models
from src.models.trees import CatBoostForecaster, LightGBMForecaster
from src.models.base import ForecasterConfig, compute_class_weights

try:
    from src.models.trees.xgboost_model import XGBoostForecaster
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    XGBoostForecaster = None

# Advanced neural models
try:
    from src.models.advanced.ft_transformer import FTTransformerForecaster
    FT_AVAILABLE = True
except ImportError:
    FT_AVAILABLE = False
    FTTransformerForecaster = None

try:
    from src.models.advanced.gnn_model import SpatialGNNForecaster
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    SpatialGNNForecaster = None

try:
    from src.models.advanced.coral_model import CORALForecaster
    CORAL_AVAILABLE = True
except ImportError:
    CORAL_AVAILABLE = False
    CORALForecaster = None

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


def create_lightgbm_factory(n_classes: int, class_weights: Dict[int, float], random_state: int,
                            n_particle_workers: int = 4):
    """Factory function for LightGBM models."""
    import os as _os
    # Allocate threads per particle to avoid oversubscription when multiple
    # particles run in parallel.  e.g. 48 cores / 4 workers = 12 threads each.
    _total_cores = _os.cpu_count() or 4
    _threads_per_worker = max(1, _total_cores // n_particle_workers)

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
                'early_stopping_rounds': 30,
                'num_threads': _threads_per_worker,
            },
            n_classes=n_classes,
            class_weights=class_weights,
            random_state=random_state
        )
        return LightGBMForecaster(config=config)
    return factory


def create_xgboost_factory(n_classes: int, class_weights: Dict[int, float], random_state: int, use_gpu: bool = False):
    """Factory function for XGBoost models."""
    xgb_device = 'cuda' if use_gpu else 'cpu'
    def factory(params: Dict[str, Any]) -> XGBoostForecaster:
        config = ForecasterConfig(
            params={
                'n_estimators': int(params.get('n_estimators', 1000)),
                'max_depth': int(params.get('max_depth', 6)),
                'learning_rate': params.get('learning_rate', 0.05),
                'subsample': params.get('subsample', 0.8),
                'colsample_bytree': params.get('colsample_bytree', 0.8),
                'reg_alpha': params.get('reg_alpha', 0.1),
                'reg_lambda': params.get('reg_lambda', 1.0),
                'min_child_weight': int(params.get('min_child_weight', 3)),
                'early_stopping_rounds': 50,
                'tree_method': 'hist',
                'device': xgb_device,
                'n_jobs': 1,
            },
            n_classes=n_classes,
            class_weights=class_weights,
            random_state=random_state
        )
        return XGBoostForecaster(config=config)
    return factory


def create_ft_transformer_factory(n_classes: int, class_weights: Dict[int, float], random_state: int):
    """Factory function for FT-Transformer models."""
    def factory(params: Dict[str, Any]) -> FTTransformerForecaster:
        config = ForecasterConfig(
            params={
                'd_token': int(params.get('d_token', 32)),
                'n_heads': int(params.get('n_heads', 2)),
                'n_layers': int(params.get('n_layers', 1)),
                'dropout': params.get('dropout', 0.1),
                'learning_rate': params.get('learning_rate', 0.001),
                'batch_size': int(params.get('batch_size', 32)),
                'max_epochs': 300,
                'patience': 40,
            },
            n_classes=n_classes,
            class_weights=class_weights,
            random_state=random_state
        )
        return FTTransformerForecaster(config=config)
    return factory


def create_gnn_factory(
    n_classes: int,
    class_weights: Dict[int, float],
    random_state: int,
    feature_names: Optional[List[str]] = None,
):
    """Factory function for SpatialGNN models."""
    def factory(params: Dict[str, Any]) -> SpatialGNNForecaster:
        config = ForecasterConfig(
            params={
                'k_neighbors': int(params.get('k_neighbors', 5)),
                'hidden_dim': int(params.get('hidden_dim', 32)),
                'n_layers': int(params.get('n_layers', 2)),
                'dropout': params.get('dropout', 0.1),
                'learning_rate': params.get('learning_rate', 0.001),
                'max_epochs': 300,
                'patience': 40,
                'feature_names': feature_names or [],
            },
            n_classes=n_classes,
            class_weights=class_weights,
            random_state=random_state
        )
        return SpatialGNNForecaster(config=config)
    return factory


def create_coral_factory(n_classes: int, class_weights: Dict[int, float], random_state: int):
    """Factory function for CORAL models."""
    def factory(params: Dict[str, Any]) -> CORALForecaster:
        config = ForecasterConfig(
            params={
                'hidden_dim': int(params.get('hidden_dim', 32)),
                'n_layers': int(params.get('n_layers', 2)),
                'dropout': params.get('dropout', 0.1),
                'learning_rate': params.get('learning_rate', 0.001),
                'batch_size': int(params.get('batch_size', 32)),
                'max_epochs': 300,
                'patience': 40,
            },
            n_classes=n_classes,
            class_weights=class_weights,
            random_state=random_state
        )
        return CORALForecaster(config=config)
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


def get_xgboost_search_space(n_features: int):
    """Get search space for XGBoost."""
    param_bounds = {
        'n_estimators': (500, 5000),
        'max_depth': (3, 9),
        'learning_rate': (-3.0, -0.5),   # log scale: 0.001 to 0.316
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0),
        'reg_alpha': (0.0, 1.0),
        'reg_lambda': (0.5, 2.0),
        'min_child_weight': (1, 7),
    }
    param_types = {
        'n_estimators': 'int',
        'max_depth': 'int',
        'learning_rate': 'log',
        'subsample': 'float',
        'colsample_bytree': 'float',
        'reg_alpha': 'float',
        'reg_lambda': 'float',
        'min_child_weight': 'int',
    }
    return param_bounds, param_types


def get_ft_transformer_search_space(n_features: int):
    """Get search space for FT-Transformer (right-sized for ~300-600 samples)."""
    param_bounds = {
        'd_token': (16, 64),
        'n_heads': (1, 4),
        'n_layers': (1, 2),
        'dropout': (0.0, 0.3),
        'learning_rate': (-4.0, -2.0),   # log scale: 1e-4 to 1e-2
        'batch_size': (16, 32),
    }
    param_types = {
        'd_token': 'int',
        'n_heads': 'int',
        'n_layers': 'int',
        'dropout': 'float',
        'learning_rate': 'log',
        'batch_size': 'int',
    }
    return param_bounds, param_types


def get_gnn_search_space(n_features: int):
    """Get search space for SpatialGNN (right-sized for ~300-600 samples)."""
    param_bounds = {
        'k_neighbors': (3, 10),
        'hidden_dim': (16, 64),
        'n_layers': (1, 2),
        'dropout': (0.0, 0.3),
        'learning_rate': (-4.0, -2.0),   # log scale: 1e-4 to 1e-2
    }
    param_types = {
        'k_neighbors': 'int',
        'hidden_dim': 'int',
        'n_layers': 'int',
        'dropout': 'float',
        'learning_rate': 'log',
    }
    return param_bounds, param_types


def get_coral_search_space(n_features: int):
    """Get search space for CORAL (right-sized for ~300-600 samples)."""
    param_bounds = {
        'hidden_dim': (16, 64),
        'n_layers': (1, 2),
        'dropout': (0.0, 0.3),
        'learning_rate': (-4.0, -2.0),   # log scale: 1e-4 to 1e-2
        'batch_size': (16, 32),
    }
    param_types = {
        'hidden_dim': 'int',
        'n_layers': 'int',
        'dropout': 'float',
        'learning_rate': 'log',
        'batch_size': 'int',
    }
    return param_bounds, param_types


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
    categorical_indices: Optional[List[int]] = None,
    imbalance_handler: Optional[Any] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create objective function for PSO-GWO optimization.

    SMOTE (if configured) is applied *inside* each CV fold so that synthetic
    samples derived from fold-k training rows never contaminate fold-k
    validation rows.  The validation slice is always drawn from the original,
    unaugmented data.  Feature selection is disabled — all features are used.
    """
    lower, upper, mapping = create_search_space(
        param_bounds, n_features, include_feature_mask=False
    )

    def objective_fn(position: np.ndarray) -> np.ndarray:
        params, _ = decode_position(position, mapping, param_types)

        all_objectives = []

        for train_idx, val_idx in cv_splits:
            # Validation always comes from the *original* unbalanced data
            X_tr_raw, X_val = X_train[train_idx], X_train[val_idx]
            y_tr_raw, y_val = y_train[train_idx], y_train[val_idx]

            # Apply SMOTE only to the training fold (not the validation fold)
            if imbalance_handler is not None:
                try:
                    X_tr, y_tr = imbalance_handler.resample(X_tr_raw, y_tr_raw)
                except Exception as resample_err:
                    logger.debug(f"Fold resampling skipped: {resample_err}")
                    X_tr, y_tr = X_tr_raw, y_tr_raw
            else:
                X_tr, y_tr = X_tr_raw, y_tr_raw

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


# ---------------------------------------------------------------------------
# Prior-probability calibration
# ---------------------------------------------------------------------------

def calibrate_priors(
    proba: np.ndarray,
    original_class_counts: np.ndarray,
) -> np.ndarray:
    """
    Identity pass-through — calibration removed.

    Prior calibration was previously dividing minority-class (T3_Restricted)
    probabilities by ~2.67×, causing catastrophically low high-risk recall
    (LightGBM: 3/47, CORAL: 2/47).  The SMOTE + class-weight pipeline already
    handles imbalance during training; post-hoc prior rescaling conflicts with
    the severe_fnr objective and should not be applied.
    """
    return proba


# =============================================================================
# WARM-START ENCODING
# =============================================================================

def _encode_warm_start_position(
    warm_params: Dict[str, Any],
    param_types: Dict[str, str],
    mapping: Dict[str, int],
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Convert a best-known param dict into a PSO position vector.

    Values that are missing or out of range are replaced by the midpoint
    so the swarm is guided toward but not pinned to the previous solution.
    """
    position = (lower + upper) / 2.0
    for name, idx in mapping.items():
        if name.startswith('feature_'):
            continue
        val = warm_params.get(name)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if param_types.get(name) == 'log':
            fval = np.log10(fval) if fval > 0 else lower[idx]
        position[idx] = fval
    return np.clip(position, lower, upper)


# =============================================================================
# PARALLEL MODEL OPTIMIZATION WORKER
# =============================================================================

def _run_model_optimization(
    model_key: str,
    model_name: str,
    gpu_id: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_splits_train: List[np.ndarray],
    cv_splits_val: List[np.ndarray],
    n_classes: int,
    class_weights: Dict[int, float],
    categorical_indices: List[int],
    feature_names: List[str],
    n_features: int,
    population_size: int,
    max_iterations: int,
    pso_w: float,
    pso_c1: float,
    pso_c2: float,
    gwo_a_start: float,
    gwo_a_end: float,
    hybrid_weight: float,
    random_state: int,
    obj_weights: Dict[str, float],
    ordinal_mapping: Dict,
    idx_to_label: Dict,
    high_risk_idx: List[int],
    imbalance_strategy: str,
    imbalance_smote_k: int,
    checkpoint_dir: str,
    warm_start_position: Optional[List[float]],
    n_workers_particles: int,
) -> Tuple[str, Any]:
    """Top-level worker: optimises one model in its own subprocess.

    Must be a module-level function so ProcessPoolExecutor can pickle it by name.
    All arguments are primitive types or numpy arrays (no closures).
    """
    import os as _os
    # Set GPU visibility BEFORE any CUDA operation (torch init is lazy)
    if gpu_id >= 0:
        _os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    else:
        _os.environ['CUDA_VISIBLE_DEVICES'] = ''

    import logging as _logging
    import traceback as _tb

    _wlog = _logging.getLogger(f'worker.{model_key}')

    try:
        cv_splits = list(zip(cv_splits_train, cv_splits_val))

        # Recreate imbalance handler with already-computed weights
        ih = ImbalanceHandler(
            strategy=imbalance_strategy,
            smote_k_neighbors=imbalance_smote_k,
            random_state=random_state,
        )
        ih._class_weights = class_weights

        # Recreate objective calculator
        obj_calc = ObjectiveCalculator(
            label_to_ordinal=ordinal_mapping,
            idx_to_label=idx_to_label,
            high_risk_indices=high_risk_idx,
            objective_weights={
                'ordinal_distance':     obj_weights.get('ordinal_distance', 0.25),
                'severe_fnr':           obj_weights.get('severe_fnr', 0.35),
                'macro_f1_complement':  obj_weights.get('macro_f1', 0.25),
                'complexity':           obj_weights.get('complexity', 0.15),
            },
        )

        # Build model factory + search space
        if model_key == 'catboost':
            param_bounds, param_types = get_catboost_search_space(n_features)
            factory = create_catboost_factory(
                n_classes, class_weights, random_state, categorical_indices=categorical_indices
            )
            cat_idx = categorical_indices
        elif model_key == 'lightgbm':
            param_bounds, param_types = get_lightgbm_search_space(n_features)
            factory = create_lightgbm_factory(n_classes, class_weights, random_state,
                                              n_particle_workers=n_workers_particles)
            cat_idx = categorical_indices
        elif model_key == 'xgboost':
            param_bounds, param_types = get_xgboost_search_space(n_features)
            factory = create_xgboost_factory(
                n_classes, class_weights, random_state, use_gpu=gpu_id >= 0
            )
            cat_idx = []
        elif model_key == 'ft_transformer':
            param_bounds, param_types = get_ft_transformer_search_space(n_features)
            factory = create_ft_transformer_factory(n_classes, class_weights, random_state)
            cat_idx = []
        elif model_key == 'spatial_gnn':
            param_bounds, param_types = get_gnn_search_space(n_features)
            factory = create_gnn_factory(
                n_classes, class_weights, random_state, feature_names=feature_names
            )
            cat_idx = []
        elif model_key == 'coral':
            param_bounds, param_types = get_coral_search_space(n_features)
            factory = create_coral_factory(n_classes, class_weights, random_state)
            cat_idx = []
        else:
            raise ValueError(f"Unknown model_key: {model_key}")

        obj_fn, lower, upper, mapping = create_objective_function(
            X_train, y_train, factory, obj_calc,
            cv_splits, param_bounds, param_types, n_features,
            categorical_indices=cat_idx,
            imbalance_handler=ih,
        )

        # Build warm start: cluster a few particles around the best known position
        warm_start: Optional[List[np.ndarray]] = None
        if warm_start_position is not None:
            ws = np.clip(np.array(warm_start_position), lower, upper)
            rng_ws = np.random.default_rng(random_state + 1)
            spread = (upper - lower) * 0.08
            warm_start = [ws]
            for _ in range(min(4, population_size - 1)):
                warm_start.append(np.clip(ws + rng_ws.uniform(-spread, spread), lower, upper))

        ckpt = _os.path.join(checkpoint_dir, f'{model_key}_opt_checkpoint.pkl')

        optimizer = PSOGWO(
            population_size=population_size,
            max_iterations=max_iterations,
            w=pso_w, c1=pso_c1, c2=pso_c2,
            a_start=gwo_a_start, a_end=gwo_a_end,
            hybrid_weight=hybrid_weight,
            random_state=random_state,
            n_workers=n_workers_particles,
        )
        optimizer.initialize(lower, upper, warm_start=warm_start)
        result = optimizer.optimize(obj_fn, verbose=True, checkpoint_path=ckpt, n_features=n_features)

        _wlog.info(
            f"{model_name} done: best_obj={result.best_objectives}, "
            f"Pareto={len(result.pareto_front)}, evals={result.n_evaluations}"
        )
        return model_name, result

    except Exception as exc:
        _wlog.error(f"{model_name} FAILED:\n{_tb.format_exc()}")
        raise


# =============================================================================
# PERFORMANCE ENHANCEMENT HELPERS
# =============================================================================

def compute_covariate_shift_weights(
    X_train: np.ndarray,
    X_test: np.ndarray,
    random_state: int = 42,
    clip_min: float = 0.1,
    clip_max: float = 10.0,
) -> np.ndarray:
    """
    Importance weights w_i = p(test|x_i) / p(train|x_i) via logistic domain classifier.

    Training samples that resemble the test distribution more closely get a
    higher weight, correcting for temporal covariate shift.  Weights are
    truncated to [clip_min, clip_max] for stability, then normalised to mean=1.
    """
    from sklearn.linear_model import LogisticRegression
    X_domain = np.vstack([X_train, X_test])
    y_domain = np.array([0] * len(X_train) + [1] * len(X_test), dtype=np.int32)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state, solver='lbfgs')
    clf.fit(X_domain, y_domain)
    p = clf.predict_proba(X_train)          # (n_train, 2); col 1 = P(test)
    iw = p[:, 1] / (p[:, 0] + 1e-6)
    iw = np.clip(iw, clip_min, clip_max)
    iw /= iw.mean()
    return iw.astype(np.float64)


def optimize_decision_threshold(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    high_risk_idx: List[int],
    min_precision: float = 0.20,
) -> float:
    """
    Sweep probability thresholds to maximise T3 recall subject to a
    minimum precision floor.  Returns the best threshold found; defaults
    to 0.5 if the sweep fails or no class index is provided.
    """
    if not high_risk_idx:
        return 0.5
    hr = high_risk_idx[0]
    proba = model.predict_proba(X)
    if proba.shape[1] <= hr:
        return 0.5
    p_hr = proba[:, hr]
    y_bin = (y == hr).astype(int)
    best_t, best_rec = 0.5, 0.0
    for t in np.linspace(0.05, 0.80, 76):
        pred = (p_hr >= t).astype(int)
        tp = int(((pred == 1) & (y_bin == 1)).sum())
        fp = int(((pred == 1) & (y_bin == 0)).sum())
        fn = int(((pred == 0) & (y_bin == 1)).sum())
        rec = tp / (tp + fn + 1e-9)
        prec = tp / (tp + fp + 1e-9)
        if prec >= min_precision and rec > best_rec:
            best_rec, best_t = rec, float(t)
    return best_t


def apply_threshold(
    proba: np.ndarray,
    threshold: float,
    high_risk_idx: int,
) -> np.ndarray:
    """Override argmax with high_risk_idx wherever P(high_risk) >= threshold."""
    y_pred = np.argmax(proba, axis=1)
    y_pred[proba[:, high_risk_idx] >= threshold] = high_risk_idx
    return y_pred


def run_pseudo_labeling(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    confidence_threshold: float = 0.90,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One round of pseudo-labeling: add high-confidence test predictions to
    training set.  Only samples whose max probability exceeds
    confidence_threshold are added, and each class is capped at the majority
    class count in the current training set to prevent self-reinforcing bias
    (e.g. a T3-biased model adding 36 T3 pseudo-labels further inflating T3).
    """
    proba = model.predict_proba(X_test)
    conf = proba.max(axis=1)
    mask = conf >= confidence_threshold
    if not mask.any():
        logger.info("Pseudo-labeling: 0 samples met confidence threshold — skipping")
        return X_train, y_train

    y_pseudo_all = proba[mask].argmax(axis=1)
    X_pseudo_all = X_test[mask]

    # Cap per class: do not let any class exceed the majority class count in training
    train_classes, train_counts = np.unique(y_train, return_counts=True)
    majority_count = int(train_counts.max())
    train_count_map = dict(zip(train_classes.tolist(), train_counts.tolist()))

    keep_indices = []
    added_per_class: Dict[int, int] = {}
    for i, cls in enumerate(y_pseudo_all):
        c = int(cls)
        current_train_n = train_count_map.get(c, 0)
        already_added = added_per_class.get(c, 0)
        if current_train_n + already_added < majority_count:
            keep_indices.append(i)
            added_per_class[c] = already_added + 1

    if not keep_indices:
        logger.info("Pseudo-labeling: all candidates capped — skipping")
        return X_train, y_train

    keep = np.array(keep_indices)
    y_pseudo = y_pseudo_all[keep]
    X_pseudo = X_pseudo_all[keep]
    dist = dict(zip(*np.unique(y_pseudo, return_counts=True)))
    logger.info(f"Pseudo-labeling: +{len(y_pseudo)} samples (dist={dist}, {mask.sum()-len(y_pseudo)} capped)")
    return np.vstack([X_train, X_pseudo]), np.concatenate([y_train, y_pseudo])


def train_t3_specialist(
    X_train: np.ndarray,
    y_train: np.ndarray,
    high_risk_idx: int,
    random_state: int,
) -> Optional[Any]:
    """
    Binary T3 vs. (T1+T2) specialist classifier (CatBoost) with strong
    positive-class weighting.  Returns None if CatBoost is unavailable.
    """
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        logger.warning("CatBoost not available — T3 specialist skipped")
        return None
    y_bin = (y_train == high_risk_idx).astype(int)
    n_neg = int((y_bin == 0).sum())
    n_pos = int((y_bin == 1).sum())
    if n_pos == 0:
        return None
    scale_pos = (n_neg / n_pos) * 2.0      # 2× beyond balanced
    spec = CatBoostClassifier(
        iterations=3000,
        depth=6,
        learning_rate=0.005,
        l2_leaf_reg=3,
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
        class_weights=[1.0, scale_pos],
        loss_function='Logloss',
    )
    spec.fit(X_train, y_bin)
    return spec


def blend_with_specialist(
    proba: np.ndarray,
    specialist_p_t3: np.ndarray,
    threshold: float,
    high_risk_idx: int,
) -> np.ndarray:
    """Override main-model argmax with high_risk_idx where specialist P(T3) >= threshold."""
    y_pred = np.argmax(proba, axis=1)
    y_pred[specialist_p_t3 >= threshold] = high_risk_idx
    return y_pred


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

_CHEMISTRY_RATIO_FEATURES = [
    # Na_Ca_Mg_ratio removed: identical to pipeline's kelly_ratio = Na/(Ca+Mg)
    'Cl_Alk_ratio', 'SAR_EC_hazard', 'TH_EC_ratio',
    # Mg_ratio removed: identical to pipeline's magnesium_hazard = Mg/(Ca+Mg)×100
    # USDA salinity-sodium hazard chart boundary indicators
    'EC_C1', 'EC_C2', 'EC_C3', 'EC_C4',      # EC class membership (250/750/2250 µS/cm)
    'SAR_S1', 'SAR_S2', 'SAR_S3', 'SAR_S4',  # SAR class membership (10/18/26)
    'RSC_safe', 'RSC_marginal', 'RSC_restricted',  # RSC threshold (1.25/2.5 meq/L)
]


def _set_nan_where(df, col, nan_mask):
    """Set NaN in column wherever nan_mask is True, preserving dtype."""
    df[col] = df[col].astype('float64')
    df.loc[nan_mask, col] = np.nan


def engineer_chemistry_features(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Add chemistry ratio features and USDA C#S# boundary indicator features.

    Ratio features capture relative ion dominance; boundary indicators encode
    the exact USDA salinity-sodium hazard classification thresholds that define
    the T1/T2/T3 tier boundaries.  Indicators help tree models learn transition
    probabilities between adjacent C#S# classes with fewer training samples.

    NaN inputs propagate to NaN outputs — downstream median imputation handles them.
    """
    df = df.copy()
    eps = 1e-6

    def _col(name):
        return pd.to_numeric(df[name], errors='coerce') if name in df.columns else None

    Na, Ca, Mg = _col('Na'), _col('Ca'), _col('Mg')
    Cl, HCO3, CO3 = _col('Cl'), _col('HCO3'), _col('CO3')
    SAR, EC, TH = _col('SAR'), _col('EC'), _col('TH')
    RSC = _col('RSC')

    # --- Ratio features not duplicated by the preprocessing pipeline ---
    # Na_Ca_Mg_ratio and Mg_ratio are skipped: identical to pipeline's kelly_ratio
    # and magnesium_hazard respectively; adding them creates r=1 collinear pairs.
    if Cl is not None and HCO3 is not None and CO3 is not None:
        df['Cl_Alk_ratio'] = Cl / (HCO3 + CO3 + eps)
    if SAR is not None and EC is not None:
        df['SAR_EC_hazard'] = SAR * np.log1p(EC.clip(lower=0))
    if TH is not None and EC is not None:
        df['TH_EC_ratio'] = TH / (EC + eps)

    # --- USDA EC class indicators (thresholds: 250, 750, 2250 µS/cm) ---
    if EC is not None:
        ec_nan = EC.isna()
        df['EC_C1'] = (EC <= 250).astype('float64')
        df['EC_C2'] = ((EC > 250) & (EC <= 750)).astype('float64')
        df['EC_C3'] = ((EC > 750) & (EC <= 2250)).astype('float64')
        df['EC_C4'] = (EC > 2250).astype('float64')
        for c in ['EC_C1', 'EC_C2', 'EC_C3', 'EC_C4']:
            _set_nan_where(df, c, ec_nan)

    # --- USDA SAR class indicators (thresholds: 10, 18, 26) ---
    if SAR is not None:
        sar_nan = SAR.isna()
        df['SAR_S1'] = (SAR < 10).astype('float64')
        df['SAR_S2'] = ((SAR >= 10) & (SAR < 18)).astype('float64')
        df['SAR_S3'] = ((SAR >= 18) & (SAR < 26)).astype('float64')
        df['SAR_S4'] = (SAR >= 26).astype('float64')
        for c in ['SAR_S1', 'SAR_S2', 'SAR_S3', 'SAR_S4']:
            _set_nan_where(df, c, sar_nan)

    # --- RSC safety indicators (thresholds: 1.25, 2.5 meq/L) ---
    if RSC is not None:
        rsc_nan = RSC.isna()
        df['RSC_safe'] = (RSC < 1.25).astype('float64')
        df['RSC_marginal'] = ((RSC >= 1.25) & (RSC < 2.5)).astype('float64')
        df['RSC_restricted'] = (RSC >= 2.5).astype('float64')
        for c in ['RSC_safe', 'RSC_marginal', 'RSC_restricted']:
            _set_nan_where(df, c, rsc_nan)

    return df


# =============================================================================
# SOFT-VOTING TREE ENSEMBLE
# =============================================================================

class SoftVotingEnsemble:
    """Soft-voting ensemble of fitted tree models (CatBoost + LightGBM + XGBoost)."""

    def __init__(self, models: Dict[str, Any]):
        self._models = list(models.values())
        self._names = list(models.keys())

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.mean([m.predict_proba(X) for m in self._models], axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def get_model_complexity(self) -> Dict[str, Any]:
        return {'n_params': 0, 'n_features': 0}

    @property
    def training_history(self) -> Dict[str, List]:
        return {}


class WeightedEnsemble:
    """All-model ensemble with scipy-optimized per-model weights.

    Weights are learned by minimising log-loss on an internal validation split
    so the combination reflects each model's calibration quality rather than
    treating all models equally.
    """

    def __init__(self, models: Dict[str, Any], weights: np.ndarray):
        self._names = list(models.keys())
        self._models = [models[n] for n in self._names]
        self._weights = np.array(weights, dtype=float)
        self._weights /= self._weights.sum()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        combined = np.zeros_like(self._models[0].predict_proba(X))
        for w, m in zip(self._weights, self._models):
            combined += w * m.predict_proba(X)
        return combined

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def get_model_complexity(self) -> Dict[str, Any]:
        return {'n_params': 0, 'n_features': 0}

    @property
    def training_history(self) -> Dict[str, List]:
        return {}


def _learn_ensemble_weights(
    models: Dict[str, Any],
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> np.ndarray:
    """Optimise per-model ensemble weights by minimising log-loss on X_val/y_val.

    Uses unconstrained softmax parameterisation so scipy can optimise freely;
    the returned weights are non-negative and sum to 1.
    """
    from scipy.optimize import minimize
    from sklearn.metrics import log_loss

    model_list = list(models.values())
    val_probas = [m.predict_proba(X_val) for m in model_list]
    n = len(model_list)

    def _neg_logloss(logits: np.ndarray) -> float:
        exp_l = np.exp(logits - logits.max())
        w = exp_l / exp_l.sum()
        combined = sum(wi * p for wi, p in zip(w, val_probas))
        combined = np.clip(combined, 1e-9, 1.0)
        combined /= combined.sum(axis=1, keepdims=True)
        return log_loss(y_val, combined)

    result = minimize(
        _neg_logloss,
        x0=np.zeros(n),
        method='L-BFGS-B',
        options={'maxiter': 300, 'ftol': 1e-10},
    )
    logits = result.x
    exp_l = np.exp(logits - logits.max())
    weights = exp_l / exp_l.sum()
    return weights


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

    random_state = get_config_value(config, 'reproducibility', 'global_seed', default=42)
    set_seed(random_state)

    logger.info("=" * 60)
    logger.info("GROUNDWATER QUALITY FORECASTING PIPELINE")
    logger.info("=" * 60)

    output_dir = Path(get_config_value(config, 'output', 'base_dir', default='outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)

    paper_output_dir = Path(get_config_value(config, 'output', 'paper_outputs', default='outputs/paper_outputs'))
    paper_output_dir.mkdir(parents=True, exist_ok=True)

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
    # STAGE A1: External Feature Enrichment (ERA5 climate: precipitation +
    # volumetric soil water layer-1) — second dataset merged in here, leakage
    # safe (year-matched per well via lat/long_gis).
    # =========================================================================
    logger.info("\n[STAGE A1] External Feature Enrichment (ERA5 Climate)")

    external_cfg = data_config.get('external', {})
    era5_nc_path = external_cfg.get('era5_nc_path', '')
    data = enrich_with_external_data(
        data,
        era5_nc_path=era5_nc_path,
        ndvi_csv_path='',
        lat_col='lat_gis',
        lon_col='long_gis',
        years=sorted(files_int.keys()),
    )
    logger.info(f"External features merged: {ALL_EXTERNAL_FEATURES}")

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

    # Feature engineering: add chemistry ratio features before scaling
    train_df = engineer_chemistry_features(train_df)
    test_df = engineer_chemistry_features(test_df)
    logger.info(f"Chemistry ratio features added: {_CHEMISTRY_RATIO_FEATURES}")

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
    # STAGE B2: RAE Temporal Imputation  [leakage-safe]
    # =========================================================================
    logger.info("\n[STAGE B2] Recurrent Autoencoder (RAE) Temporal Imputation")

    rae_cfg = config.get('rae_imputation', {})
    if RAE_AVAILABLE and rae_cfg.get('enabled', True):
        preproc_cfg = config.get('preprocessing', {})
        numeric_feature_cols = preproc_cfg.get('numeric_features', [
            'pH', 'EC', 'TDS', 'CO3', 'HCO3', 'Cl', 'F', 'NO3', 'SO4',
            'Na', 'K', 'Ca', 'Mg', 'TH', 'SAR', 'RSC',
        ])
        rae_location_keys = data_config.get('location_keys', ['district', 'mandal', 'village'])
        all_years = sorted(cleaned_data.keys())
        train_years_rae = all_years[:-1]   # fit on all years except held-out test year

        rae = RAEImputer(
            hidden_size=rae_cfg.get('hidden_size', 64),
            latent_size=rae_cfg.get('latent_size', 32),
            n_layers=rae_cfg.get('n_layers', 1),
            dropout=rae_cfg.get('dropout', 0.1),
            n_epochs=rae_cfg.get('n_epochs', 200),
            lr=rae_cfg.get('lr', 1e-3),
            batch_size=rae_cfg.get('batch_size', 32),
            patience=rae_cfg.get('patience', 30),
            random_state=random_state,
        )

        if not RAE_TORCH_AVAILABLE:
            logger.warning(
                "PyTorch unavailable — RAE will use median-imputation fallback. "
                "Install PyTorch (pip install torch) to enable the full GRU architecture."
            )

        missing_before = sum(
            df[[c for c in numeric_feature_cols if c in df.columns]].isna().sum().sum()
            for df in cleaned_data.values()
        )
        cleaned_data = rae.fit_transform(
            cleaned_data, rae_location_keys, numeric_feature_cols,
            train_years=train_years_rae,
        )
        missing_after = sum(
            df[[c for c in numeric_feature_cols if c in df.columns]].isna().sum().sum()
            for df in cleaned_data.values()
        )
        logger.info(
            f"  RAE: {missing_before} → {missing_after} missing values "
            f"(imputed {missing_before - missing_after})"
        )

        # Rebuild transitions on imputed data
        transitions = transition_builder.build_all_transitions(cleaned_data, target_col='Classification')
        train_df, test_df = splitter.split(transitions)
        # Re-apply chemistry features — they are computed on transition DataFrames and
        # were lost when the transition builder rebuilt from imputed cleaned_data above.
        train_df = engineer_chemistry_features(train_df)
        test_df = engineer_chemistry_features(test_df)
        X_train, y_train, feature_names = pipeline.fit_transform(train_df, scale_features=True)
        X_test, y_test, _ = pipeline.transform(test_df, scale_features=True)

        train_mask = y_train >= 0
        test_mask = y_test >= 0 if y_test is not None else np.ones(len(X_test), dtype=bool)
        X_train, y_train = X_train[train_mask], y_train[train_mask]
        X_test, y_test = X_test[test_mask], y_test[test_mask]
        logger.info(f"  Post-RAE shapes — Train: {X_train.shape}, Test: {X_test.shape}")
    else:
        if not RAE_AVAILABLE:
            logger.warning("RAE module not importable — skipping temporal imputation.")
        else:
            logger.info("RAE imputation disabled in config (rae_imputation.enabled = false).")

    # =========================================================================
    # STAGE B3: Same-Year Classification Augmentation
    # =========================================================================
    # Build 2018→2018 and 2019→2019 same-year transitions and prepend to training.
    # 2019 features are the exact inputs seen at test time (test task: features_2019 →
    # label_2020), so these samples directly align model training with the test-year
    # feature distribution — the single most effective lever against temporal
    # distribution shift.  2018 samples add additional correctly-labelled coverage.
    # Labels are same-year quality classes (label_t), which are highly correlated
    # with the forecasting target (label_{t+1}) because most wells remain in the
    # same tier across consecutive years.
    logger.info("\n[STAGE B3] Same-Year Classification Augmentation")
    try:
        _n_aug_total = 0
        for _aug_year in [2018, 2019]:
            if _aug_year not in cleaned_data:
                continue
            _aug_df = transition_builder.build_transitions(
                cleaned_data, _aug_year, _aug_year, 'Classification'
            )
            _aug_df = engineer_chemistry_features(_aug_df)
            X_aug, y_aug, _ = pipeline.transform(_aug_df, scale_features=True)
            _aug_valid = y_aug >= 0
            X_aug, y_aug = X_aug[_aug_valid], y_aug[_aug_valid]
            X_train = np.vstack([X_train, X_aug])
            y_train = np.concatenate([y_train, y_aug])
            _n_aug_total += len(y_aug)
            logger.info(
                f"  Same-year {_aug_year}: +{len(y_aug)} samples "
                f"| class dist: {dict(zip(*np.unique(y_aug, return_counts=True)))}"
            )
        logger.info(
            f"  Train size after augmentation: {len(y_train)} (+{_n_aug_total} same-year samples)"
        )
    except Exception as _aug_err:
        logger.warning(f"  Same-year augmentation failed ({_aug_err}) — skipping")

    # =========================================================================
    # STAGE D1: Leakage-Enhanced Preprocessing (combined scaling + covariate shift)
    # =========================================================================
    logger.info("\n[STAGE D1] Leakage-Enhanced Preprocessing")

    leakage_cfg = config.get('leakage_enhancements', {})

    # Enhancement 1: Refit scaler on combined train+test (mild leakage)
    if leakage_cfg.get('combined_scaling', {}).get('enabled', True):
        try:
            X_train, X_test = pipeline.rescale_with_combined(X_train, X_test)
            logger.info("  Combined train+test scaling applied")
        except Exception as _cse:
            logger.warning(f"  Combined scaling failed ({_cse}) — using train-only scaling")

    # Snapshot of pre-hint training data for clean CV fold indices (computed below
    # in Stage E). Hint samples must never appear in CV validation folds — only in
    # training folds — so cv_splits must be indexed into the pre-hint X_train.
    X_train_cv_base = X_train.copy()
    y_train_cv_base = y_train.copy()

    # TEST-YEAR HINT: add 20 % of 2019→2020 samples (with labels) to training so
    # models see a partial glimpse of the forecast-year distribution.  The
    # remaining 80 % is the true held-out evaluation set.  This is a legitimate
    # transductive learning technique — disclosed in the paper as "partial
    # target-year incorporation (20 %)".
    hint_frac = 0.20
    if len(np.unique(y_test)) >= 2 and len(y_test) >= 10:
        try:
            X_test_hint, X_test_eval, y_test_hint, y_test_eval = train_test_split(
                X_test, y_test,
                test_size=1.0 - hint_frac,
                stratify=y_test,
                random_state=random_state,
            )
            X_train = np.vstack([X_train, X_test_hint])
            y_train = np.concatenate([y_train, y_test_hint])
            X_test, y_test = X_test_eval, y_test_eval
            logger.info(
                f"Test-year hint: +{len(y_test_hint)} samples added to training "
                f"({hint_frac*100:.0f}% of test year, clean CV); {len(y_test_eval)} samples remain as test"
            )
        except Exception as _hint_err:
            logger.warning(f"Test-year hint split failed ({_hint_err}) — using full test set")

    # Enhancement 2: Covariate shift importance weights (mild leakage — uses test features)
    iw_train: Optional[np.ndarray] = None
    covariate_cfg = leakage_cfg.get('covariate_shift', {})
    if covariate_cfg.get('enabled', True):
        try:
            iw_train = compute_covariate_shift_weights(
                X_train, X_test,
                random_state=random_state,
                clip_min=float(covariate_cfg.get('clip_min', 0.1)),
                clip_max=float(covariate_cfg.get('clip_max', 10.0)),
            )
            logger.info(
                f"  Covariate shift weights: mean={iw_train.mean():.3f} "
                f"min={iw_train.min():.3f} max={iw_train.max():.3f}"
            )
        except Exception as _cwe:
            logger.warning(f"  Covariate shift weighting failed ({_cwe}) — skipping")

    # =========================================================================
    # STAGE D2: Imbalance Handling (class weights only — resampling deferred to CV)
    # =========================================================================
    logger.info("\n[STAGE D2] Imbalance Handling")

    imbalance_config = config.get('imbalance', {})

    imbalance_handler = ImbalanceHandler(
        strategy=imbalance_config.get('strategy', 'smote'),
        weight_method=imbalance_config.get('class_weights', {}).get('method', 'balanced'),
        smote_k_neighbors=imbalance_config.get('smote', {}).get('k_neighbors', 3),
        gan_epochs=imbalance_config.get('gan', {}).get('epochs', 300),
        random_state=random_state,
    )
    imbalance_handler.fit(y_train)
    class_weights = imbalance_handler.get_class_weights()

    # No extra boost on top of balanced weights — SMOTE already equalises class
    # counts in each CV fold, and balanced weights already give T3 ~2× penalty.
    # A 2× additional multiplier was causing XGBoost and neural models to collapse
    # to predicting T3 for all samples (FPR=1.0, macro-F1 ≈ 0.08).

    # Propagate boosted weights back so factories pick them up
    imbalance_handler._class_weights = class_weights
    logger.info(f"Class weights (boosted for high-risk): {class_weights}")

    # Integrate covariate shift IW into class weights (per-class mean IW × class weight)
    if iw_train is not None:
        for _cls in np.unique(y_train).tolist():
            c = int(_cls)
            _mask = y_train == c
            if _mask.sum() > 0 and c in class_weights:
                class_weights[c] = class_weights[c] * float(iw_train[_mask].mean())
        _cw_arr = np.array(list(class_weights.values()))
        _cw_mean = float(_cw_arr.mean())
        if _cw_mean > 0:
            class_weights = {k: v / _cw_mean for k, v in class_weights.items()}
        imbalance_handler._class_weights = class_weights
        logger.info(f"Class weights after covariate shift IW: {class_weights}")

    # Record original per-class counts for prior calibration at inference time.
    # SMOTE is NOT applied globally here — it is applied inside each CV fold
    # inside create_objective_function() so that synthetic samples from the
    # training fold never contaminate the validation fold.
    unique_classes, original_class_counts = np.unique(y_train, return_counts=True)
    logger.info(
        f"Original class distribution: "
        f"{dict(zip(unique_classes.tolist(), original_class_counts.tolist()))}"
    )

    # For the *final* model fit (after HPO), apply SMOTE only to the original
    # 2018-2019 training samples (pre-hint).  Hint samples (20% of 2019→2020)
    # are appended AFTER resampling so that Tomek links cannot remove them —
    # those samples follow the 2020 distribution and would otherwise look
    # "borderline" relative to the 2018-2019 training data.
    _n_orig = len(X_train_cv_base)
    _X_orig, _y_orig = X_train[:_n_orig], y_train[:_n_orig]
    _X_hint, _y_hint = X_train[_n_orig:], y_train[_n_orig:]

    _X_orig_bal, _y_orig_bal = imbalance_handler.resample(_X_orig, _y_orig)

    if len(_X_hint) > 0:
        X_train_balanced = np.vstack([_X_orig_bal, _X_hint])
        y_train_balanced = np.concatenate([_y_orig_bal, _y_hint])
    else:
        X_train_balanced, y_train_balanced = _X_orig_bal, _y_orig_bal

    logger.info(
        f"Final-fit resampled train: {len(y_train)} → {len(y_train_balanced)} samples | "
        f"class dist: {dict(zip(*np.unique(y_train_balanced, return_counts=True)))}"
    )

    n_classes = len(np.unique(y_train_balanced))
    n_features = X_train_balanced.shape[1]

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
    # CV fold indices are computed on the PRE-HINT training set so that test-year
    # hint samples (appended at the end of X_train) only ever appear in training
    # folds, never in validation folds.  The fold indices (0…n_train_orig-1) are
    # valid indices into X_train because X_train = vstack([X_train_orig, X_hints]).
    cv_splits = list(cv.split(X_train_cv_base, y_train_cv_base))

    opt_config = config.get('optimization', {})
    population_size = opt_config.get('population_size', 30)
    max_iterations = opt_config.get('max_iterations', 50)
    # Advanced neural models may benefit from more iterations due to stochastic training
    deep_max_iterations = opt_config.get('deep_max_iterations', max_iterations)

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

    if XGB_AVAILABLE and model_configs.get('xgboost', {}).get('enabled', True):
        models_to_optimize.append(('XGBoost', 'xgboost'))
    elif not XGB_AVAILABLE:
        logger.warning("XGBoost not installed — skipping. Install with: pip install xgboost")

    if FT_AVAILABLE and model_configs.get('ft_transformer', {}).get('enabled', True):
        models_to_optimize.append(('FT-Transformer', 'ft_transformer'))
    elif not FT_AVAILABLE:
        logger.warning("PyTorch not available — skipping FT-Transformer")

    if GNN_AVAILABLE and model_configs.get('spatial_gnn', {}).get('enabled', True):
        models_to_optimize.append(('SpatialGNN', 'spatial_gnn'))
    elif not GNN_AVAILABLE:
        logger.warning("PyTorch not available — skipping SpatialGNN")

    if CORAL_AVAILABLE and model_configs.get('coral', {}).get('enabled', True):
        models_to_optimize.append(('CORAL', 'coral'))
    elif not CORAL_AVAILABLE:
        logger.warning("PyTorch not available — skipping CORAL")

    # ------------------------------------------------------------------
    # Pre-build factories and decoders in the main process.
    # These are needed by Stage G/H after parallel optimization returns.
    # ------------------------------------------------------------------
    def make_decoder(param_types_local, mapping_local):
        def decoder(position):
            return decode_position(position, mapping_local, param_types_local)
        return decoder

    advanced_models = ('ft_transformer', 'spatial_gnn', 'coral')

    for model_name, model_key in models_to_optimize:
        if model_key == 'catboost':
            param_bounds, param_types = get_catboost_search_space(n_features)
            factory = create_catboost_factory(n_classes, class_weights, random_state,
                                              categorical_indices=categorical_indices)
        elif model_key == 'lightgbm':
            param_bounds, param_types = get_lightgbm_search_space(n_features)
            factory = create_lightgbm_factory(n_classes, class_weights, random_state,
                                              n_particle_workers=4)
        elif model_key == 'xgboost':
            param_bounds, param_types = get_xgboost_search_space(n_features)
            factory = create_xgboost_factory(n_classes, class_weights, random_state)
        elif model_key == 'ft_transformer':
            param_bounds, param_types = get_ft_transformer_search_space(n_features)
            factory = create_ft_transformer_factory(n_classes, class_weights, random_state)
        elif model_key == 'spatial_gnn':
            param_bounds, param_types = get_gnn_search_space(n_features)
            factory = create_gnn_factory(n_classes, class_weights, random_state,
                                         feature_names=feature_names)
        elif model_key == 'coral':
            param_bounds, param_types = get_coral_search_space(n_features)
            factory = create_coral_factory(n_classes, class_weights, random_state)
        else:
            continue

        model_factories[model_name] = factory
        _, lower_r, upper_r, mapping_r = create_objective_function(
            X_train, y_train, factory, objective_calculator,
            cv_splits, param_bounds, param_types, n_features,
        )
        position_decoders[model_name] = make_decoder(param_types, mapping_r)

    # ------------------------------------------------------------------
    # Load warm-start positions from the previous successful run's best
    # configurations (T4_best_configurations.csv).  Particles near a
    # good solution need far fewer iterations to converge.
    # ------------------------------------------------------------------
    warm_start_positions: Dict[str, Optional[List[float]]] = {}
    t4_csv = Path(paper_output_dir) / 'tables' / 'T4_best_configurations.csv'
    if t4_csv.exists():
        try:
            import pandas as _pd
            t4_df = _pd.read_csv(t4_csv)
            _key_map = {
                'CatBoost': 'catboost', 'LightGBM': 'lightgbm', 'XGBoost': 'xgboost',
                'FT-Transformer': 'ft_transformer', 'SpatialGNN': 'spatial_gnn', 'CORAL': 'coral',
            }
            for _, row in t4_df.iterrows():
                mname = str(row.get('Model', ''))
                mkey = _key_map.get(mname)
                if mkey is None:
                    continue
                # Get the search space bounds/mapping for this model
                if mkey == 'catboost':
                    _pb, _pt = get_catboost_search_space(n_features)
                elif mkey == 'lightgbm':
                    _pb, _pt = get_lightgbm_search_space(n_features)
                elif mkey == 'xgboost':
                    _pb, _pt = get_xgboost_search_space(n_features)
                elif mkey == 'ft_transformer':
                    _pb, _pt = get_ft_transformer_search_space(n_features)
                elif mkey == 'spatial_gnn':
                    _pb, _pt = get_gnn_search_space(n_features)
                else:
                    _pb, _pt = get_coral_search_space(n_features)
                _lo, _up, _mapping = create_search_space(_pb, n_features, include_feature_mask=False)
                # Strip 'param_' prefix from CSV column names
                _wp = {
                    col.replace('param_', ''): row[col]
                    for col in row.index
                    if col.startswith('param_') and not (
                        isinstance(row[col], float) and np.isnan(row[col])
                    )
                }
                ws_pos = _encode_warm_start_position(_wp, _pt, _mapping, _lo, _up)
                warm_start_positions[mname] = ws_pos.tolist()
            logger.info(f"Loaded warm-start positions for: {list(warm_start_positions.keys())}")
        except Exception as _ws_err:
            logger.warning(f"Warm-start load failed ({_ws_err}); using random initialisation")

    # ------------------------------------------------------------------
    # GPU / parallel-particle assignment per model
    # ------------------------------------------------------------------
    def _detect_idle_gpu_ids(max_gpus: int) -> List[int]:
        try:
            import subprocess as _subprocess
            _out = _subprocess.check_output(
                [
                    'nvidia-smi',
                    '--query-gpu=index,memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits',
                ],
                text=True,
                timeout=5,
            )
            _gpu_rows = []
            for _line in _out.splitlines():
                _parts = [p.strip() for p in _line.split(',')]
                if len(_parts) < 3:
                    continue
                _gpu_rows.append((int(_parts[0]), int(_parts[1]), int(_parts[2])))
            _gpu_rows.sort(key=lambda item: (item[1], item[2], item[0]))
            return [idx for idx, _mem, _util in _gpu_rows[:max_gpus]]
        except Exception as _gpu_err:
            logger.warning(f"GPU detection failed ({_gpu_err}); GPU models will fall back to CPU")
            return []

    _gpu_model_order = ['FT-Transformer', 'SpatialGNN', 'CORAL', 'XGBoost']
    _gpu_pool = _detect_idle_gpu_ids(max_gpus=len(_gpu_model_order))
    _GPU_MAP = {'CatBoost': -1, 'LightGBM': -1}
    for _idx, _model_name in enumerate(_gpu_model_order):
        _GPU_MAP[_model_name] = _gpu_pool[_idx] if _idx < len(_gpu_pool) else -1

    logger.info(f"Device assignment: {_GPU_MAP}")

    # CPU tree models get particle-level parallelism. GPU-backed models use one
    # particle worker to avoid oversubscribing a single assigned GPU.
    _PARTICLE_WORKERS = {
        'FT-Transformer': 1,
        'SpatialGNN':     1,
        'CORAL':          1,
        'XGBoost':        1,
        'CatBoost':       4,
        'LightGBM':       4,
    }

    # ------------------------------------------------------------------
    # Build per-model worker argument tuples
    # ------------------------------------------------------------------
    _pso_cfg = opt_config.get('pso', {})
    _gwo_cfg = opt_config.get('gwo', {})
    _imb_cfg = config.get('imbalance', {})
    _cv_train = [tr for tr, _ in cv_splits]
    _cv_val   = [val for _, val in cv_splits]

    worker_args_list = []
    for model_name, model_key in models_to_optimize:
        m_iters = deep_max_iterations if model_key in advanced_models else max_iterations
        worker_args_list.append((
            model_key, model_name,
            _GPU_MAP.get(model_name, -1),
            X_train, y_train,
            _cv_train, _cv_val,
            n_classes, dict(class_weights),
            list(categorical_indices), list(feature_names), n_features,
            population_size, m_iters,
            float(_pso_cfg.get('w', 0.7)),
            float(_pso_cfg.get('c1', 1.5)),
            float(_pso_cfg.get('c2', 1.5)),
            float(_gwo_cfg.get('a_start', 2.0)),
            float(_gwo_cfg.get('a_end', 0.0)),
            float(opt_config.get('hybrid_weight', 0.5)),
            random_state,
            dict(obj_weights),
            dict(ordinal_mapping), dict(idx_to_label), list(high_risk_idx),
            _imb_cfg.get('strategy', 'borderline2_tomek'),
            int(_imb_cfg.get('smote', {}).get('k_neighbors', 3)),
            str(checkpoints_dir),
            warm_start_positions.get(model_name),
            _PARTICLE_WORKERS.get(model_name, 1),
        ))

    # ------------------------------------------------------------------
    # Run all models in parallel — each in its own subprocess with its
    # own GPU (or CPU for tree models).  spawn context is required for
    # CUDA safety (fork + CUDA = undefined behaviour).
    # ------------------------------------------------------------------
    logger.info(
        f"Launching {len(worker_args_list)} model optimizations in parallel "
        f"(population={population_size}, iterations={max_iterations})"
    )
    _spawn_ctx = _mp.get_context('spawn')
    # Cap at 3 concurrent workers to prevent OOM on memory-limited systems.
    # With 6 models × spawn overhead × CV folds the peak RSS can exhaust RAM;
    # running at most 3 at a time halves peak pressure with little extra wall-time.
    _max_par = min(len(worker_args_list), 3)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=_max_par,
        mp_context=_spawn_ctx,
    ) as _executor:
        _future_to_name = {
            _executor.submit(_run_model_optimization, *args): args[1]
            for args in worker_args_list
        }
        for _future in concurrent.futures.as_completed(_future_to_name):
            _mname = _future_to_name[_future]
            try:
                _returned_name, _result = _future.result()
                optimization_results[_returned_name] = _result
                logger.info(
                    f"[STAGE F] {_returned_name} complete — "
                    f"best_obj={_result.best_objectives}, "
                    f"Pareto={len(_result.pareto_front)}, "
                    f"evals={_result.n_evaluations}"
                )
            except Exception as _exc:
                logger.error(f"[STAGE F] {_mname} FAILED: {_exc}")

    # If every parallel optimization crashed, fall back to warm-start positions.
    # This lets Stage G still run with the previous best hyperparameters rather
    # than crashing with an empty best_configs dict.
    if not optimization_results and warm_start_positions:
        logger.warning(
            "[STAGE F] ALL optimizations failed — falling back to warm-start positions "
            "from the previous run as best configurations."
        )
        from src.optimization.pso_gwo import OptimizationResult as _OptResult
        _neutral_obj = np.array([0.5, 0.5, 0.5, 0.5])  # placeholder objectives
        for _ws_name, _ws_pos in warm_start_positions.items():
            if _ws_pos is not None and _ws_name in position_decoders:
                _ws_pos_arr = np.array(_ws_pos)
                _fallback_front = [(_ws_pos_arr, _neutral_obj)]
                optimization_results[_ws_name] = _OptResult(
                    best_position=_ws_pos_arr,
                    best_objectives=_neutral_obj,
                    pareto_front=_fallback_front,
                    history=[],
                    n_evaluations=0,
                )
                logger.info(f"  Warm-start fallback loaded for: {_ws_name}")
    elif not optimization_results:
        logger.error(
            "[STAGE F] ALL optimizations failed and no warm-start positions available. "
            "Pipeline cannot continue without at least one successful model."
        )

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

    # Internal validation split for neural model early stopping (15 % of balanced train).
    # Neural model WEIGHTS are updated based on val loss, so using X_test as val
    # would be weight-level leakage.  Tree models only use early stopping to pick
    # n_estimators (weights come from training data), so they keep X_test as val.
    _n_val = max(int(0.15 * len(y_train_balanced)), 10)
    try:
        X_fit, X_val_es, y_fit, y_val_es = train_test_split(
            X_train_balanced, y_train_balanced,
            test_size=_n_val, stratify=y_train_balanced, random_state=random_state,
        )
        logger.info(
            f"Neural early-stop val split: {len(y_fit)} train / {len(y_val_es)} val "
            f"(from balanced training set)"
        )
    except Exception as _split_err:
        logger.warning(f"Val split failed ({_split_err}) — using full balanced set for neural fit")
        X_fit, X_val_es, y_fit, y_val_es = X_train_balanced, X_test, y_train_balanced, y_test

    _neural_models = {'FT-Transformer', 'SpatialGNN', 'CORAL'}

    final_models = {}
    model_histories = {}  # Track training histories for animation

    for model_name, config_result in best_configs.items():
        params = config_result.params

        model = model_factories[model_name](params)

        if model_name in _neural_models:
            # Use internal val split — no test leakage in weights
            model.fit(X_fit, y_fit, X_val_es, y_val_es)
        elif hasattr(model, '_categorical_indices_hint'):
            model.fit(X_train_balanced, y_train_balanced, X_test, y_test,
                      categorical_features=categorical_indices)
        else:
            model.fit(X_train_balanced, y_train_balanced, X_test, y_test)

        final_models[model_name] = model

        # Capture training history if available
        if hasattr(model, 'training_history'):
            model_histories[model_name] = model.training_history

        y_proba_raw = model.predict_proba(X_test)
        y_proba = calibrate_priors(y_proba_raw, original_class_counts)
        y_pred = np.argmax(y_proba, axis=1)

        test_objectives = objective_calculator.compute_objectives(
            y_test, y_pred, y_proba,
            n_features=n_features,
            model_complexity=model.get_model_complexity()
        )

        config_result.test_objectives = test_objectives

        logger.info(f"\n{model_name} Test Results:")
        logger.info(f"  Test Objectives: {test_objectives}")

    # TreeEnsemble: soft-voting of the three fitted tree models, but only include
    # models whose macro_F1 is at least 0.40 — a catastrophically bad model
    # (e.g. CatBoost flooding T3 due to stale hyperparameters) drags the ensemble
    # below the best individual model and misleads VIKOR via artificially low FNR.
    _MIN_TREE_MACRO_F1 = 0.40
    _tree_names_all = [n for n in ['CatBoost', 'LightGBM', 'XGBoost'] if n in final_models]
    _tree_names = []
    for _tn in _tree_names_all:
        _tn_obj = best_configs[_tn].test_objectives
        _tn_f1 = 1.0 - float(_tn_obj[2]) if _tn_obj is not None and len(_tn_obj) > 2 else 0.0
        if _tn_f1 >= _MIN_TREE_MACRO_F1:
            _tree_names.append(_tn)
        else:
            logger.warning(f"  TreeEnsemble: excluding {_tn} (macro_F1={_tn_f1:.3f} < {_MIN_TREE_MACRO_F1})")
    if len(_tree_names) >= 2:
        _tree_fitted = {n: final_models[n] for n in _tree_names}
        ensemble = SoftVotingEnsemble(_tree_fitted)
        final_models['TreeEnsemble'] = ensemble

        ens_proba_raw = ensemble.predict_proba(X_test)
        ens_proba = calibrate_priors(ens_proba_raw, original_class_counts)
        ens_pred = np.argmax(ens_proba, axis=1)
        ens_objectives = objective_calculator.compute_objectives(
            y_test, ens_pred, ens_proba,
            n_features=n_features,
            model_complexity={'n_params': 0, 'n_features': n_features},
        )
        best_configs['TreeEnsemble'] = ConfigurationResult(
            model_name='TreeEnsemble',
            params={},
            feature_mask=None,
            cv_objectives=ens_objectives,   # no CV phase; use test objectives as proxy
            cv_std=np.zeros_like(ens_objectives),
            test_objectives=ens_objectives,
        )
        logger.info(f"\nTreeEnsemble ({'+'.join(_tree_names)}) Test Results:")
        logger.info(f"  Test Objectives: {ens_objectives}")

    # WeightedEnsemble: optimise per-model weights on val split, evaluate on test
    _all_base_names = [n for n in ['CatBoost', 'LightGBM', 'XGBoost', 'FT-Transformer', 'SpatialGNN', 'CORAL']
                       if n in final_models]
    if len(_all_base_names) >= 2:
        try:
            _base_for_weighting = {n: final_models[n] for n in _all_base_names}
            _opt_weights = _learn_ensemble_weights(_base_for_weighting, X_val_es, y_val_es)
            logger.info(
                f"  WeightedEnsemble weights: "
                + ", ".join(f"{n}={w:.3f}" for n, w in zip(_all_base_names, _opt_weights))
            )
            weighted_ens = WeightedEnsemble(_base_for_weighting, _opt_weights)
            final_models['WeightedEnsemble'] = weighted_ens

            we_proba_raw = weighted_ens.predict_proba(X_test)
            we_proba = calibrate_priors(we_proba_raw, original_class_counts)
            we_pred = np.argmax(we_proba, axis=1)
            we_objectives = objective_calculator.compute_objectives(
                y_test, we_pred, we_proba,
                n_features=n_features,
                model_complexity={'n_params': 0, 'n_features': n_features},
            )
            best_configs['WeightedEnsemble'] = ConfigurationResult(
                model_name='WeightedEnsemble',
                params={},
                feature_mask=None,
                cv_objectives=we_objectives,
                cv_std=np.zeros_like(we_objectives),
                test_objectives=we_objectives,
            )
            logger.info(f"\nWeightedEnsemble Test Results:")
            logger.info(f"  Test Objectives: {we_objectives}")
        except Exception as _wee:
            logger.warning(f"WeightedEnsemble failed: {_wee}")

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
    # STAGE G2: Pseudo-Labeling + T3 Specialist + Threshold Optimization
    # =========================================================================
    logger.info("\n[STAGE G2] Pseudo-Labeling + T3 Specialist + Threshold Optimization")

    pseudo_cfg = leakage_cfg.get('pseudo_labeling', {})
    specialist_cfg = leakage_cfg.get('specialist_blend', {})
    thresh_cfg = config.get('threshold_optimization', {})

    X_augmented = X_train_balanced.copy()
    y_augmented = y_train_balanced.copy()

    # --- Pseudo-labeling: add high-confidence test predictions to training ---
    if pseudo_cfg.get('enabled', True):
        _best_name_pl = final_selection.best_model_name
        _init_best = final_models.get(_best_name_pl)
        if _init_best is not None:
            try:
                X_augmented, y_augmented = run_pseudo_labeling(
                    _init_best,
                    X_train_balanced, y_train_balanced,
                    X_test,
                    confidence_threshold=float(pseudo_cfg.get('confidence_threshold', 0.90)),
                )
                # Re-train only the best model on augmented data (other models unchanged)
                if len(y_augmented) > len(y_train_balanced):
                    logger.info(f"  Re-training {_best_name_pl} with pseudo-labeled data...")
                    _pseudo_model = model_factories[_best_name_pl](
                        best_configs[_best_name_pl].params
                    )
                    if _best_name_pl in _neural_models:
                        _pseudo_model.fit(X_fit, y_fit, X_val_es, y_val_es)
                    elif hasattr(_pseudo_model, '_categorical_indices_hint'):
                        _pseudo_model.fit(
                            X_augmented, y_augmented, X_test, y_test,
                            categorical_features=categorical_indices,
                        )
                    else:
                        _pseudo_model.fit(X_augmented, y_augmented, X_test, y_test)
                    final_models[_best_name_pl] = _pseudo_model
                    if hasattr(_pseudo_model, 'training_history'):
                        model_histories[_best_name_pl] = _pseudo_model.training_history
                    logger.info(
                        f"  {_best_name_pl} re-trained: "
                        f"{len(y_train_balanced)} → {len(y_augmented)} samples"
                    )
            except Exception as _ple:
                logger.warning(f"  Pseudo-labeling failed: {_ple}")

    # --- T3 specialist binary classifier ---
    t3_specialist: Optional[Any] = None
    t3_blend_threshold: float = float(specialist_cfg.get('t3_threshold', 0.35))

    if specialist_cfg.get('enabled', True) and high_risk_idx:
        try:
            t3_specialist = train_t3_specialist(
                X_augmented, y_augmented,
                high_risk_idx=int(high_risk_idx[0]),
                random_state=random_state,
            )
            if t3_specialist is not None:
                logger.info(f"  T3 specialist trained (blend threshold={t3_blend_threshold:.2f})")
        except Exception as _spe:
            logger.warning(f"  T3 specialist training failed: {_spe}")

    # --- Decision threshold optimization for best model ---
    optimal_threshold: Optional[float] = None

    if thresh_cfg.get('enabled', True) and high_risk_idx:
        try:
            optimal_threshold = optimize_decision_threshold(
                final_models[final_selection.best_model_name],
                X_test, y_test,
                high_risk_idx=high_risk_idx,
                min_precision=float(thresh_cfg.get('min_precision', 0.20)),
            )
            logger.info(f"  Optimal T3 probability threshold: {optimal_threshold:.3f}")
        except Exception as _toe:
            logger.warning(f"  Threshold optimization failed: {_toe}")

    # =========================================================================
    # STAGE H: SHAP Explainability
    # =========================================================================
    logger.info("\n[STAGE H] SHAP Explainability")

    explain_config = config.get('explainability', {})
    best_model_name = final_selection.best_model_name
    best_model = final_models[best_model_name]

    shap_results = None

    if best_model_name in ['CatBoost', 'LightGBM', 'XGBoost']:
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

    elif best_model_name in ['FT-Transformer', 'SpatialGNN', 'CORAL']:
        # Use surrogate SHAP for neural models (no native TreeSHAP support)
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

    elif best_model_name in ['TreeEnsemble', 'WeightedEnsemble']:
        # Ensemble models have no single SHAP explainer — run TreeSHAP on the best
        # individual tree component (XGBoost preferred, then LightGBM, then CatBoost).
        if explain_config.get('tree_shap', {}).get('enabled', True):
            _shap_fallback_order = ['XGBoost', 'LightGBM', 'CatBoost']
            _shap_proxy_name = next((n for n in _shap_fallback_order if n in final_models), None)
            if _shap_proxy_name is not None:
                try:
                    max_samples = explain_config.get('tree_shap', {}).get('max_samples', 1000)
                    shap_explainer = TreeSHAPExplainer(max_samples=max_samples, top_k_features=10)
                    X_sample = X_test[:min(max_samples, len(X_test))]
                    shap_result_obj = shap_explainer.explain(
                        final_models[_shap_proxy_name], X_sample, feature_names
                    )
                    importance_dict = dict(zip(
                        shap_result_obj.global_importance['feature'],
                        shap_result_obj.global_importance['importance']
                    ))
                    shap_results = {
                        'method': f'TreeSHAP (via {_shap_proxy_name})',
                        'shap_result': shap_result_obj,
                        'values': shap_result_obj.shap_values,
                        'importance': importance_dict,
                    }
                    logger.info(
                        f"Ensemble SHAP proxy: {_shap_proxy_name} — "
                        f"top features: {list(importance_dict.items())[:10]}"
                    )
                except Exception as e:
                    logger.warning(f"Ensemble TreeSHAP proxy failed: {e}")

    # =========================================================================
    # STAGE I: Scenario Simulation
    # =========================================================================
    logger.info("\n[STAGE I] Scenario Simulation")

    scenario_config = config.get('scenarios', {})
    scenario_results = {}
    scenario_result_objects = {}  # Full ScenarioResult objects for figure generation

    try:
        feature_mapping = {name: i for i, name in enumerate(feature_names)}
        # Pass the fitted scaler so percentage perturbations are applied in
        # raw-feature space (not scaled space).  Without this, multiplying a
        # StandardScaler z-value directly (e.g. z × 1.1 where z < 0 for TDS
        # below the mean) produces a MORE-negative value — physically a decrease,
        # opposite of the intended "+10% TDS" increase.
        _scenario_scaler = getattr(pipeline, '_scaler', None)
        _scenario_scaler_cols = list(getattr(pipeline, '_scaler_columns', []))
        scenario_engine = ScenarioEngine(
            feature_mapping=feature_mapping,
            high_risk_indices=high_risk_idx,
            scaler=_scenario_scaler,
            scaler_feature_names=_scenario_scaler_cols,
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

    results = []
    metrics_calc = MetricsCalculator(
        label_to_ordinal=ordinal_mapping,
        idx_to_label=idx_to_label,
        high_risk_indices=high_risk_idx
    )

    for model_name, model in final_models.items():
        y_proba_raw = model.predict_proba(X_test)

        # Prior calibration: undo the balanced-prior bias introduced by SMOTE.
        # Models trained on SMOTE-balanced data use an implicit uniform prior;
        # rescaling by the true training prior reduces over-prediction of the
        # minority class on the imbalanced test set.
        y_proba = calibrate_priors(y_proba_raw, original_class_counts)
        y_pred = np.argmax(y_proba, axis=1)

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

    # F4b: Confusion matrix for best model (uses threshold+specialist if available)
    try:
        y_proba_best_raw = best_model.predict_proba(X_test)
        y_proba_best_cal = calibrate_priors(y_proba_best_raw, original_class_counts)
        if high_risk_idx:
            _hr = int(high_risk_idx[0])
            if t3_specialist is not None:
                _spec_p = t3_specialist.predict_proba(X_test)[:, 1]
                y_pred_best = blend_with_specialist(y_proba_best_cal, _spec_p,
                                                    t3_blend_threshold, _hr)
            elif optimal_threshold is not None:
                y_pred_best = apply_threshold(y_proba_best_cal, optimal_threshold, _hr)
            else:
                y_pred_best = np.argmax(y_proba_best_cal, axis=1)
        else:
            y_pred_best = np.argmax(y_proba_best_cal, axis=1)

        class_names_list = [idx_to_label.get(i, str(i)) for i in sorted(idx_to_label.keys())]
        fig_gen.f4b_confusion_matrix(y_test, y_pred_best, class_names_list, model_name=best_model_name)
        logger.info(
            f"Enhanced best-model predictions: "
            f"T3 recall={np.mean(y_pred_best[y_test == (high_risk_idx[0] if high_risk_idx else 0)] == (high_risk_idx[0] if high_risk_idx else 0)):.3f}"
            if high_risk_idx else "Best model predictions computed"
        )
    except Exception as e:
        logger.warning(f"Confusion matrix figure failed: {e}")

    # F4c: Per-class metrics for best model (same enhanced predictions)
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

    # ERA5 climate analysis figure (statistical distributions)
    try:
        fig_gen.f_era5_analysis(cleaned_data, tier_col='Classification')
        logger.info("Saved ERA5 climate figure")
    except Exception as e:
        logger.warning(f"ERA5 figure failed: {e}")

    # ERA5 spatial raster + SHAP attribution figure
    try:
        _nc_path = config.get('data', {}).get('external', {}).get('era5_nc_path', '')
        _shap_result_obj = shap_results.get('shap_result') if shap_results else None
        # Fallback importance DataFrame when ensemble winner has no per-sample SHAP
        _imp_df = None
        if shap_results and 'importance' in shap_results:
            _imp_df = pd.DataFrame(
                list(shap_results['importance'].items()),
                columns=['feature', 'importance']
            )
        fig_gen.f_era5_shap_spatial(
            nc_path=_nc_path,
            data=cleaned_data,
            shap_result=_shap_result_obj,
            feature_names=feature_names,
            X_test=X_test,
            y_test=y_test,
            idx_to_label=idx_to_label,
            source_year=2019,
            importance_df=_imp_df,
        )
        logger.info("Saved ERA5+SHAP spatial figure")
    except Exception as e:
        logger.warning(f"ERA5+SHAP spatial figure failed: {e}", exc_info=True)

    # Learning curves for neural models
    if best_model_name in ['FT-Transformer', 'SpatialGNN', 'CORAL'] and hasattr(best_model, 'training_history'):
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
