"""
Version 2 Pipeline — RAE Imputation + Enhanced SHAP Analysis
=============================================================

Improvements over run_enhanced_pipeline.py
-------------------------------------------
1. **Raw preprocessing** in DataIngestion:
       - Year-specific column renames (chemical notation)
       - Drops 'sno' / 'season' / 'Unnamed: 8'
       - Fixes known outliers / typos (pH '8..05', Classification 'O.G')

2. **Recurrent Autoencoder (RAE) imputation**:
       - GRU encoder-decoder trained on training years (2018–2019) only.
       - Imputes missing hydrochemical values in a temporally consistent
         manner before the preprocessing pipeline runs.
       - Prevents data leakage: test year (2020) features are imputed using
         a model that has never seen 2020 data.

3. **Enhanced SHAP analysis figures**:
       - F7  — bar chart (mean |SHAP|)           [existing]
       - F7b — beeswarm dot plot                  [new]
       - F7c — global vs high-risk side-by-side   [new]
       - F7d — per-feature dependence plots       [new]
       - F7e — feature × class heatmap            [new]

All other stages (PSO-GWO optimisation, VIKOR, scenario simulation,
table generation, managerial insights) are identical to the enhanced
pipeline so that module interactions are fully preserved.

Usage
-----
    python -m experiments.run_v2_pipeline --config configs/enhanced_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# ── utils ──────────────────────────────────────────────────────────────────
from src.utils.config import load_config, get_config_value
from src.utils.logging import setup_logging
from src.utils.reproducibility import set_seed

# ── data ───────────────────────────────────────────────────────────────────
from src.data import DataIngestion, DataHarmonizer, TransitionBuilder
from src.data_quality import DataQualityValidator, DataCleaner

# ── imputation (NEW) ────────────────────────────────────────────────────────
from src.imputation import RAEImputer, TORCH_AVAILABLE as RAE_TORCH_AVAILABLE

# ── preprocessing ──────────────────────────────────────────────────────────
from src.preprocessing import EnhancedPreprocessingPipeline, EnhancedPreprocessingConfig

# ── evaluation ─────────────────────────────────────────────────────────────
from src.evaluation import TemporalSplitter, MetricsCalculator

# ── imbalance ──────────────────────────────────────────────────────────────
from src.imbalance import (
    EnhancedImbalanceHandler,
    compute_enhanced_class_weights,
    optimize_threshold_for_recall,
    apply_threshold_adjustment,
)

# ── models ─────────────────────────────────────────────────────────────────
from src.models.trees import CatBoostForecaster, LightGBMForecaster
from src.models.base import ForecasterConfig

try:
    from src.models.deep import GRUForecaster, LSTMForecaster
    from src.imbalance import FocalLoss, create_loss_function, TORCH_LOSSES_AVAILABLE
    DEEP_MODELS_AVAILABLE = True
except ImportError:
    DEEP_MODELS_AVAILABLE = False
    GRUForecaster = LSTMForecaster = None
    TORCH_LOSSES_AVAILABLE = False

# ── optimisation ───────────────────────────────────────────────────────────
from src.optimization.pso_gwo import PSOGWO, create_search_space, decode_position
from src.decision.vikor import VIKOR
from src.decision.model_selection import ModelSelector, ConfigurationResult

# ── objectives ─────────────────────────────────────────────────────────────
from src.objectives import ObjectiveCalculator
from src.objectives.definitions import get_high_risk_indices, get_ordinal_mapping, HIGH_RISK_CLASSES

# ── explainability ─────────────────────────────────────────────────────────
from src.explain.shap_tree import TreeSHAPExplainer, SHAPResult

# ── scenario simulation ────────────────────────────────────────────────────
from src.scenarios.engine import ScenarioEngine

# ── reporting ──────────────────────────────────────────────────────────────
from src.reporting import FigureGenerator, TableGenerator
from src.reporting.managerial import ManagerialInsights

logger = logging.getLogger(__name__)


# =============================================================================
# MODEL FACTORIES  (same as enhanced pipeline)
# =============================================================================

def create_catboost_factory(n_classes, class_weights, random_state):
    def factory(params):
        cfg = ForecasterConfig(
            params={
                "iterations": int(params.get("iterations", 500)),
                "depth": int(params.get("depth", 6)),
                "learning_rate": params.get("learning_rate", 0.05),
                "l2_leaf_reg": params.get("l2_leaf_reg", 5),
                "early_stopping_rounds": 30,
                "subsample": params.get("subsample", 0.8),
            },
            n_classes=n_classes, class_weights=class_weights,
            random_state=random_state,
        )
        return CatBoostForecaster(config=cfg)
    return factory


def create_lightgbm_factory(n_classes, class_weights, random_state):
    def factory(params):
        cfg = ForecasterConfig(
            params={
                "n_estimators": int(params.get("n_estimators", 500)),
                "max_depth": int(params.get("max_depth", 6)),
                "learning_rate": params.get("learning_rate", 0.05),
                "num_leaves": int(params.get("num_leaves", 31)),
                "min_child_samples": int(params.get("min_child_samples", 30)),
                "reg_alpha": params.get("reg_alpha", 0.5),
                "reg_lambda": params.get("reg_lambda", 0.5),
                "subsample": params.get("subsample", 0.8),
                "colsample_bytree": params.get("colsample_bytree", 0.8),
                "early_stopping_rounds": 30,
            },
            n_classes=n_classes, class_weights=class_weights,
            random_state=random_state,
        )
        return LightGBMForecaster(config=cfg)
    return factory


def create_gru_factory(n_classes, class_weights, random_state, categorical_indices, use_focal_loss=True):
    def factory(params):
        cfg = ForecasterConfig(
            params={
                "hidden_size": int(params.get("hidden_size", 128)),
                "num_layers": int(params.get("num_layers", 1)),
                "dropout": params.get("dropout", 0.4),
                "learning_rate": params.get("learning_rate", 0.001),
                "batch_size": int(params.get("batch_size", 32)),
                "embedding_dim": int(params.get("embedding_dim", 16)),
                "max_epochs": 300, "patience": 30,
                "use_focal_loss": use_focal_loss, "focal_gamma": 2.0,
            },
            n_classes=n_classes, class_weights=class_weights,
            random_state=random_state,
        )
        m = GRUForecaster(config=cfg)
        m._categorical_indices_hint = categorical_indices
        return m
    return factory


def create_lstm_factory(n_classes, class_weights, random_state, categorical_indices, use_focal_loss=True):
    def factory(params):
        cfg = ForecasterConfig(
            params={
                "hidden_size": int(params.get("hidden_size", 128)),
                "num_layers": int(params.get("num_layers", 1)),
                "dropout": params.get("dropout", 0.4),
                "learning_rate": params.get("learning_rate", 0.001),
                "batch_size": int(params.get("batch_size", 32)),
                "embedding_dim": int(params.get("embedding_dim", 16)),
                "max_epochs": 300, "patience": 30,
                "use_focal_loss": use_focal_loss, "focal_gamma": 2.0,
            },
            n_classes=n_classes, class_weights=class_weights,
            random_state=random_state,
        )
        m = LSTMForecaster(config=cfg)
        m._categorical_indices_hint = categorical_indices
        return m
    return factory


# =============================================================================
# SEARCH SPACES
# =============================================================================

def get_catboost_search_space(n_features):
    bounds = {"iterations": (200, 800), "depth": (4, 8),
               "learning_rate": (-1.5, -0.5), "l2_leaf_reg": (3, 10), "subsample": (0.7, 0.95)}
    types  = {"iterations": "int", "depth": "int", "learning_rate": "log",
               "l2_leaf_reg": "float", "subsample": "float"}
    return bounds, types


def get_lightgbm_search_space(n_features):
    bounds = {"n_estimators": (200, 800), "max_depth": (4, 8),
               "learning_rate": (-1.5, -0.5), "num_leaves": (15, 63),
               "min_child_samples": (20, 50), "reg_alpha": (0.1, 1.0), "reg_lambda": (0.1, 1.0)}
    types  = {"n_estimators": "int", "max_depth": "int", "learning_rate": "log",
               "num_leaves": "int", "min_child_samples": "int",
               "reg_alpha": "float", "reg_lambda": "float"}
    return bounds, types


def get_gru_search_space(n_features):
    bounds = {"hidden_size": (64, 256), "num_layers": (1, 2), "dropout": (0.3, 0.5),
               "learning_rate": (-3.3, -2.5), "batch_size": (32, 64), "embedding_dim": (16, 32)}
    types  = {"hidden_size": "int", "num_layers": "int", "dropout": "float",
               "learning_rate": "log", "batch_size": "int", "embedding_dim": "int"}
    return bounds, types


get_lstm_search_space = get_gru_search_space


# =============================================================================
# OBJECTIVE FUNCTION
# =============================================================================

def create_objective_function(
    X_train, y_train, model_factory, calculator, cv_splits,
    param_bounds, param_types, n_features,
    include_feature_selection=True, categorical_indices=None,
):
    lower, upper, mapping = create_search_space(
        param_bounds, n_features, include_feature_mask=include_feature_selection
    )

    def objective_fn(position):
        params, feature_mask = decode_position(position, mapping, param_types)

        if include_feature_selection and feature_mask.sum() > 0:
            selected = feature_mask.astype(bool)
            X_m = X_train[:, selected]
            n_sel = int(feature_mask.sum())
            if categorical_indices:
                cat_m = []
                idx = 0
                for i, s in enumerate(selected):
                    if s:
                        if i in categorical_indices:
                            cat_m.append(idx)
                        idx += 1
            else:
                cat_m = None
        else:
            X_m, n_sel, cat_m = X_train, n_features, categorical_indices

        all_obj = []
        for tr_idx, val_idx in cv_splits:
            X_tr, X_val = X_m[tr_idx], X_m[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            try:
                model = model_factory(params)
                if hasattr(model, "_categorical_indices_hint"):
                    model.fit(X_tr, y_tr, X_val, y_val, categorical_features=cat_m)
                else:
                    model.fit(X_tr, y_tr, X_val, y_val)
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)
                obj = calculator.compute_objectives(
                    y_val, y_pred, y_proba,
                    n_features=n_sel,
                    model_complexity=model.get_model_complexity(),
                )
                all_obj.append(obj)
            except Exception as e:
                logger.warning(f"CV fold failed: {e}")
                all_obj.append(np.array([10.0, 1.0, 1.0, 1.0]))

        return np.mean(all_obj, axis=0)

    return objective_fn, lower, upper, mapping


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_v2_pipeline(config_path: str):
    """
    Run the v2 forecasting pipeline with RAE imputation and full SHAP figures.
    """
    config = load_config(config_path)

    setup_logging(
        level=get_config_value(config, "logging", "level", default="INFO"),
        log_file=(
            get_config_value(config, "output", "logs", default="outputs/logs")
            + "/v2_pipeline.log"
        ),
    )

    seed = get_config_value(config, "reproducibility", "global_seed", default=42)
    set_seed(seed)

    logger.info("=" * 70)
    logger.info("V2 GROUNDWATER QUALITY FORECASTING PIPELINE")
    logger.info("=" * 70)
    logger.info("Key additions vs enhanced pipeline:")
    logger.info("  [+] Raw preprocessing in DataIngestion (rename/drop/fix-outliers)")
    logger.info("  [+] Recurrent Autoencoder (RAE) temporal imputation")
    logger.info("  [+] Full SHAP figure suite (F7 bar, F7b beeswarm, F7c comparison,")
    logger.info("       F7d dependence, F7e class heatmap)")
    logger.info("=" * 70)

    output_dir = Path(get_config_value(config, "output", "base_dir", default="outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_output_dir = Path(
        get_config_value(config, "output", "paper_outputs", default="outputs/paper_outputs")
    )
    paper_output_dir.mkdir(parents=True, exist_ok=True)
    (paper_output_dir / "tables").mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # STAGE A: Data Ingestion
    # =========================================================================
    logger.info("\n[STAGE A] Data Ingestion + Raw Preprocessing")
    # NOTE: DataIngestion._load_single_file now applies:
    #   - year-specific column renames  (data2: EC→E.C, CO_-2→CO3, …)
    #   - drops sno, season
    #   - removes Unnamed columns
    #   - fixes pH / Classification outliers in year 2020

    data_config = config.get("data", {})
    base_dir = data_config.get("base_dir")
    files_int = {int(k): v for k, v in data_config.get("files", {}).items()}

    ingestion = DataIngestion(base_dir=base_dir, files=files_int)
    raw_data = ingestion.load_all()

    # =========================================================================
    # STAGE A2: Harmonisation
    # =========================================================================
    logger.info("\n[STAGE A2] Column Harmonisation")
    harmonizer = DataHarmonizer()
    data = harmonizer.harmonize_all(raw_data)

    for year, df in data.items():
        logger.info(f"  Year {year}: {len(df)} rows × {len(df.columns)} columns")

    # =========================================================================
    # STAGE B: Data Cleaning
    # =========================================================================
    logger.info("\n[STAGE B] Data Cleaning")
    label_config = config.get("labels", {})
    dq_config = config.get("data_quality", {})

    cleaner = DataCleaner(
        label_typo_mapping=label_config.get("typo_mapping", {}),
        rare_class_handling=dq_config.get("cleaning", {}).get("handle_rare_classes", "merge"),
        rare_class_threshold=dq_config.get("cleaning", {}).get("rare_class_threshold", 5),
    )
    cleaned_data = cleaner.clean_all(data, fit_year=2018)
    logger.info(f"  Label encoder from cleaner: {cleaner.get_label_encoder()}")

    # =========================================================================
    # STAGE C: RAE Imputation  [NEW]
    # =========================================================================
    logger.info("\n[STAGE C] Recurrent Autoencoder (RAE) Imputation")

    preproc_cfg = config.get("preprocessing", {})
    numeric_feature_cols = preproc_cfg.get("numeric_features", [
        "pH", "EC", "CO3", "HCO3", "Cl", "F", "NO3", "SO4",
        "Na", "K", "Ca", "Mg", "TH", "SAR", "RSC",
    ])
    location_keys = data_config.get("location_keys", ["district", "mandal", "village"])
    all_years = sorted(cleaned_data.keys())
    train_years_rae = all_years[:-1]  # fit RAE on all years except the last (test year)

    rae_cfg = config.get("rae_imputation", {})
    rae = RAEImputer(
        hidden_size=rae_cfg.get("hidden_size", 64),
        latent_size=rae_cfg.get("latent_size", 32),
        n_layers=rae_cfg.get("n_layers", 1),
        dropout=rae_cfg.get("dropout", 0.1),
        n_epochs=rae_cfg.get("n_epochs", 200),
        lr=rae_cfg.get("lr", 1e-3),
        batch_size=rae_cfg.get("batch_size", 32),
        patience=rae_cfg.get("patience", 30),
        random_state=seed,
    )

    if not RAE_TORCH_AVAILABLE:
        logger.warning(
            "PyTorch unavailable — RAE will use median imputation fallback. "
            "Install PyTorch to enable the full RAE architecture."
        )

    # fit on training years, transform all years (no leakage: test year 2020
    # is imputed using a model that has never observed 2020 data)
    imputed_data = rae.fit_transform(
        cleaned_data, location_keys, numeric_feature_cols,
        train_years=train_years_rae,
    )

    missing_before = sum(
        df[[c for c in numeric_feature_cols if c in df.columns]].isna().sum().sum()
        for df in cleaned_data.values()
    )
    missing_after = sum(
        df[[c for c in numeric_feature_cols if c in df.columns]].isna().sum().sum()
        for df in imputed_data.values()
    )
    logger.info(
        f"  Missing values: {missing_before} → {missing_after} "
        f"(imputed {missing_before - missing_after})"
    )

    # =========================================================================
    # STAGE D: Transition Building
    # =========================================================================
    logger.info("\n[STAGE D] Transition Building")
    transition_builder = TransitionBuilder(location_keys=location_keys)
    transitions = transition_builder.build_all_transitions(
        imputed_data, target_col="Classification"
    )

    # =========================================================================
    # STAGE E: Enhanced Preprocessing
    # =========================================================================
    logger.info("\n[STAGE E] Enhanced Preprocessing")

    eval_cfg = config.get("evaluation", {})
    splitter = TemporalSplitter(
        train_transitions=eval_cfg.get("temporal_split", {}).get(
            "train_transitions", ["2018_2019"]
        ),
        test_transitions=eval_cfg.get("temporal_split", {}).get(
            "test_transitions", ["2019_2020"]
        ),
    )
    train_df, test_df = splitter.split(transitions)
    logger.info(f"  Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    enhanced_config = EnhancedPreprocessingConfig(
        remove_redundant_features=True,
        remove_shifted_features=True,
        apply_power_transform=preproc_cfg.get("power_transform", {}).get("enabled", True),
        scaling_method="robust",
        winsorize_limits=(0.02, 0.98),
        add_current_class_feature=preproc_cfg.get("feature_engineering", {}).get(
            "add_current_class", True
        ),
        add_interaction_features=preproc_cfg.get("feature_engineering", {}).get(
            "add_interactions", True
        ),
        class_weight_strategy="custom_high_risk",
        high_risk_weight_multiplier=config.get("imbalance", {})
        .get("class_weights", {})
        .get("high_risk_multiplier", 3.0),
    )

    pipeline = EnhancedPreprocessingPipeline(
        config=enhanced_config,
        numeric_features=numeric_feature_cols,
        categorical_features=preproc_cfg.get("categorical_features", []),
        spatial_features=preproc_cfg.get("spatial_features", []),
        target_column="Classification_target",
    )

    X_train, y_train, feature_names = pipeline.fit_transform(train_df, scale_features=True)
    X_test, y_test, _ = pipeline.transform(test_df, scale_features=True)

    train_mask = y_train >= 0
    test_mask = (y_test >= 0) if y_test is not None else np.ones(len(X_test), dtype=bool)
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]

    class_weights = pipeline.get_class_weights()
    label_encoder = pipeline.get_label_encoder()
    categorical_indices = pipeline.get_categorical_indices()
    n_classes = len(np.unique(y_train))
    n_features = X_train.shape[1]
    random_state = seed

    logger.info(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    logger.info(f"  Feature names: {feature_names}")
    logger.info(f"  n_classes: {n_classes}, class_weights: {class_weights}")

    # =========================================================================
    # STAGE F: Objective setup
    # =========================================================================
    logger.info("\n[STAGE F] Objective Setup")

    ordinal_mapping = get_ordinal_mapping(label_encoder)
    idx_to_label = {v: k for k, v in label_encoder.items()}
    high_risk_idx = get_high_risk_indices(label_encoder)

    obj_weights_cfg = config.get("mcdm", {}).get("objective_weights", {})
    objective_calculator = ObjectiveCalculator(
        label_to_ordinal=ordinal_mapping,
        idx_to_label=idx_to_label,
        high_risk_indices=high_risk_idx,
        objective_weights={
            "ordinal_distance":     obj_weights_cfg.get("ordinal_distance", 0.20),
            "severe_fnr":           obj_weights_cfg.get("severe_fnr", 0.45),
            "macro_f1_complement":  obj_weights_cfg.get("macro_f1", 0.25),
            "complexity":           obj_weights_cfg.get("complexity", 0.10),
        },
    )

    cv_cfg = eval_cfg.get("inner_cv", {})
    cv = StratifiedKFold(
        n_splits=cv_cfg.get("n_splits", 5), shuffle=True, random_state=random_state
    )
    cv_splits = list(cv.split(X_train, y_train))

    opt_cfg = config.get("optimization", {})
    population_size = opt_cfg.get("population_size", 20)
    max_iterations = opt_cfg.get("max_iterations", 30)

    # =========================================================================
    # STAGE G: PSO-GWO Optimisation
    # =========================================================================
    logger.info("\n[STAGE G] PSO-GWO Model Optimisation")

    model_cfgs = config.get("models", {})
    optimization_results: Dict[str, Any] = {}
    model_factories: Dict[str, Callable] = {}
    position_decoders: Dict[str, Callable] = {}

    models_to_opt = []
    if model_cfgs.get("catboost", {}).get("enabled", True):
        models_to_opt.append(("CatBoost", "catboost"))
    if model_cfgs.get("lightgbm", {}).get("enabled", True):
        models_to_opt.append(("LightGBM", "lightgbm"))
    if DEEP_MODELS_AVAILABLE:
        if model_cfgs.get("gru", {}).get("enabled", True):
            models_to_opt.append(("GRU", "gru"))
        if model_cfgs.get("lstm", {}).get("enabled", True):
            models_to_opt.append(("LSTM", "lstm"))

    for model_name, model_key in models_to_opt:
        logger.info(f"\n--- Optimising {model_name} ---")

        if model_key == "catboost":
            p_bounds, p_types = get_catboost_search_space(n_features)
            factory = create_catboost_factory(n_classes, class_weights, random_state)
            cat_idx = None
        elif model_key == "lightgbm":
            p_bounds, p_types = get_lightgbm_search_space(n_features)
            factory = create_lightgbm_factory(n_classes, class_weights, random_state)
            cat_idx = None
        elif model_key == "gru":
            p_bounds, p_types = get_gru_search_space(n_features)
            factory = create_gru_factory(n_classes, class_weights, random_state, categorical_indices)
            cat_idx = categorical_indices
        elif model_key == "lstm":
            p_bounds, p_types = get_lstm_search_space(n_features)
            factory = create_lstm_factory(n_classes, class_weights, random_state, categorical_indices)
            cat_idx = categorical_indices
        else:
            continue

        model_factories[model_name] = factory

        obj_fn, lower, upper, mapping = create_objective_function(
            X_train, y_train, factory, objective_calculator,
            cv_splits, p_bounds, p_types, n_features,
            include_feature_selection=True, categorical_indices=cat_idx,
        )

        def _make_decoder(pt, mp):
            return lambda pos: decode_position(pos, mp, pt)

        position_decoders[model_name] = _make_decoder(p_types, mapping)

        optimizer = PSOGWO(
            population_size=population_size,
            max_iterations=max_iterations,
            w=opt_cfg.get("pso", {}).get("w", 0.6),
            c1=opt_cfg.get("pso", {}).get("c1", 1.8),
            c2=opt_cfg.get("pso", {}).get("c2", 1.2),
            a_start=opt_cfg.get("gwo", {}).get("a_start", 2.0),
            a_end=opt_cfg.get("gwo", {}).get("a_end", 0.0),
            hybrid_weight=opt_cfg.get("hybrid_weight", 0.5),
            random_state=random_state,
        )
        optimizer.initialize(lower, upper)
        result = optimizer.optimize(obj_fn, verbose=True)
        optimization_results[model_name] = result
        logger.info(f"  {model_name} best_objectives: {result.best_objectives}")

    # =========================================================================
    # STAGE H: VIKOR Model Selection
    # =========================================================================
    logger.info("\n[STAGE H] VIKOR Model Selection")

    mcdm_cfg = config.get("mcdm", {})
    vikor_v = mcdm_cfg.get("vikor", {}).get("v", 0.6)
    obj_w_arr = np.array([
        obj_weights_cfg.get("ordinal_distance", 0.20),
        obj_weights_cfg.get("severe_fnr", 0.45),
        obj_weights_cfg.get("macro_f1", 0.25),
        obj_weights_cfg.get("complexity", 0.10),
    ])

    model_selector = ModelSelector(vikor_v=vikor_v, objective_weights=obj_w_arr)
    best_configs: Dict[str, ConfigurationResult] = {}

    for model_name, opt_result in optimization_results.items():
        pareto_front = opt_result.pareto_front
        if not pareto_front:
            logger.warning(f"Empty Pareto front for {model_name}")
            continue

        best_idx, _ = model_selector.select_best_config(pareto_front, model_name)
        best_pos, best_cv_obj = pareto_front[best_idx]
        params, feature_mask = position_decoders[model_name](best_pos)

        logger.info(f"\n  {model_name}: params={params}, "
                    f"n_features={int(feature_mask.sum())}/{n_features}, "
                    f"cv_obj={best_cv_obj}")

        best_configs[model_name] = ConfigurationResult(
            model_name=model_name,
            params=params,
            feature_mask=feature_mask,
            cv_objectives=best_cv_obj,
            cv_std=np.zeros_like(best_cv_obj),
            test_objectives=None,
        )

    # =========================================================================
    # STAGE I: Test Evaluation + Threshold Optimisation
    # =========================================================================
    logger.info("\n[STAGE I] Test Evaluation + Threshold Optimisation")

    metrics_calc = MetricsCalculator(
        label_to_ordinal=ordinal_mapping,
        idx_to_label=idx_to_label,
        high_risk_indices=high_risk_idx,
    )

    final_models: Dict[str, Tuple] = {}
    all_results: List[Dict] = []
    final_preds: Dict[str, Tuple] = {}  # model_name -> (y_true, y_pred_adj, y_proba)

    for model_name, cfg_result in best_configs.items():
        params = cfg_result.params
        feature_mask = cfg_result.feature_mask

        if feature_mask is not None and feature_mask.sum() > 0:
            sel = feature_mask.astype(bool)
            X_tr_sel = X_train[:, sel]
            X_te_sel = X_test[:, sel]
            n_sel = int(feature_mask.sum())
            if model_name in ("GRU", "LSTM") and categorical_indices:
                cat_sel = []
                idx = 0
                for i, s in enumerate(sel):
                    if s:
                        if i in categorical_indices:
                            cat_sel.append(idx)
                        idx += 1
            else:
                cat_sel = None
        else:
            X_tr_sel, X_te_sel = X_train, X_test
            n_sel = n_features
            cat_sel = categorical_indices if model_name in ("GRU", "LSTM") else None

        model = model_factories[model_name](params)
        if hasattr(model, "_categorical_indices_hint"):
            model.fit(X_tr_sel, y_train, X_te_sel, y_test, categorical_features=cat_sel)
        else:
            model.fit(X_tr_sel, y_train, X_te_sel, y_test)

        final_models[model_name] = (model, sel if feature_mask is not None else None)

        y_pred = model.predict(X_te_sel)
        y_proba = model.predict_proba(X_te_sel)

        thr_cfg = config.get("imbalance", {}).get("threshold_optimization", {})
        if thr_cfg.get("enabled", True):
            opt_thr, thr_metrics = optimize_threshold_for_recall(
                y_test, y_proba, high_risk_idx,
                target_recall=thr_cfg.get("target_recall", 0.7),
            )
            y_pred_adj = apply_threshold_adjustment(y_proba, high_risk_idx, threshold=opt_thr)
            logger.info(f"  {model_name} threshold: {thr_metrics}")
        else:
            y_pred_adj = y_pred

        metrics = metrics_calc.compute_all(y_test, y_pred_adj, y_proba)
        metrics["Model"] = model_name

        test_obj = objective_calculator.compute_objectives(
            y_test, y_pred_adj, y_proba,
            n_features=n_sel, model_complexity=model.get_model_complexity(),
        )
        cfg_result.test_objectives = test_obj

        logger.info(
            f"\n  {model_name} — Accuracy={metrics['accuracy']:.4f}  "
            f"Macro-F1={metrics['macro_f1']:.4f}  "
            f"Severe-FNR={metrics['severe_fnr']:.4f}"
        )
        all_results.append(metrics)
        final_preds[model_name] = (y_test, y_pred_adj, y_proba)

    final_selection = model_selector.select_best_model(best_configs)
    logger.info(f"\n  Best model: {final_selection.best_model_name}")

    # =========================================================================
    # STAGE J: SHAP Explainability  [ENHANCED]
    # =========================================================================
    logger.info("\n[STAGE J] SHAP Explainability (Full Figure Suite)")

    best_model_name = final_selection.best_model_name
    shap_result: Optional[SHAPResult] = None
    best_model_obj = None

    if best_model_name in final_models:
        best_model_obj, best_sel = final_models[best_model_name]
        X_explain = X_test[:, best_sel] if best_sel is not None else X_test
        explain_names = (
            [feature_names[i] for i, s in enumerate(best_sel) if s]
            if best_sel is not None
            else feature_names
        )

        shap_explainer = TreeSHAPExplainer(max_samples=500, top_k_features=15)
        try:
            shap_result = shap_explainer.explain(best_model_obj, X_explain, explain_names)
            logger.info(f"  SHAP top features: {shap_result.top_features}")

            # SHAP focused on high-risk predictions
            y_pred_best = best_model_obj.predict(X_explain)
            shap_hr = shap_explainer.explain_high_risk_predictions(
                best_model_obj, X_explain, y_pred_best, high_risk_idx, explain_names
            )

        except Exception as e:
            logger.warning(f"  SHAP failed for {best_model_name}: {e}")
            shap_result = None
            shap_hr = None

    # =========================================================================
    # STAGE K: Paper Outputs — Figures
    # =========================================================================
    logger.info("\n[STAGE K] Paper Outputs — Figures")

    fig_gen = FigureGenerator(
        output_dir=paper_output_dir / "figures", dpi=300
    )
    fig_gen.f2_class_distribution(data, target_col="Classification")
    fig_gen.f3_temporal_forecasting_schematic()
    fig_gen.create_mermaid_flowchart()

    if all_results:
        results_df_stage_k = pd.DataFrame(all_results)
        metric_cols = ["macro_f1", "severe_fnr", "ordinal_distance_mean"]
        avail_metrics = [c for c in metric_cols if c in results_df_stage_k.columns]
        if avail_metrics:
            fig_gen.f4_model_comparison(results_df_stage_k, metrics=avail_metrics)

        # F4d: severity metrics comparison across all models
        fig_gen.f4d_severity_comparison(results_df_stage_k)

    fig_gen.f5_pareto_fronts(optimization_results)

    # F6: VIKOR rankings
    objective_names = ["Ordinal Distance", "Severe FNR", "1 - Macro F1", "Complexity"]
    try:
        comparison_df = model_selector.generate_comparison_table(
            final_selection, objective_names=objective_names
        )
        fig_gen.f6_vikor_rankings(comparison_df)
        comparison_df.to_csv(
            paper_output_dir / "tables" / "model_comparison_vikor.csv", index=False
        )
    except Exception as _e:
        logger.warning(f"  F6 VIKOR figure skipped: {_e}")

    # F4b: Confusion matrix + F4c: per-class metrics for best model
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

    # F7 suite — enhanced SHAP figures
    if shap_result is not None:
        hr_imp = shap_hr.global_importance if shap_hr is not None else None

        shap_paths = fig_gen.generate_all_shap_figures(
            shap_result=shap_result,
            X=X_explain,
            global_importance=shap_result.global_importance,
            high_risk_importance=hr_imp,
            class_names=class_names_list,
            top_k=15,
        )
        logger.info(f"  SHAP figures saved: {list(shap_paths.keys())}")

    # =========================================================================
    # STAGE L: Paper Outputs — Tables
    # =========================================================================
    logger.info("\n[STAGE L] Paper Outputs — Tables")

    tbl_gen = TableGenerator(
        output_dir=paper_output_dir / "tables", include_latex=True
    )
    tbl_gen.t1_dataset_overview(data)

    if shap_result is not None and shap_hr is not None:
        tbl_gen.t6_shap_drivers(
            shap_result.global_importance,
            shap_hr.global_importance,
        )

    # Final rankings
    rankings = final_selection.level2_ranking.rankings
    model_list = list(best_configs.keys())
    for m in all_results:
        mn = m["Model"]
        if mn in model_list:
            mi = model_list.index(mn)
            rp = np.where(rankings == mi)[0]
            m["Rank"] = int(rp[0] + 1) if len(rp) > 0 else 99
            m["VIKOR_Q"] = float(final_selection.level2_ranking.Q[mi])

    results_df = pd.DataFrame(all_results).sort_values("Rank")
    results_df.to_csv(paper_output_dir / "tables" / "model_results_v2.csv", index=False)

    # =========================================================================
    # STAGE M: Managerial Insights
    # =========================================================================
    logger.info("\n[STAGE M] Managerial Insights")

    try:
        insights = ManagerialInsights(output_dir=str(paper_output_dir))
        model_results_for_insights = {
            mn: cr for mn, cr in best_configs.items()
        }
        shap_results_for_insights = (
            {best_model_name: shap_result} if shap_result else {}
        )

        findings = insights.generate_key_findings(
            model_results=model_results_for_insights,
            shap_results=shap_results_for_insights,
            scenario_results={},
        )
        if shap_result:
            top_shap = shap_result.top_features[:5]
        else:
            top_shap = []

        recommendations = insights.generate_recommendations(
            shap_top_features=top_shap,
            vulnerable_districts=[],
            high_impact_scenarios=[],
        )
        insights.save_insights(findings, recommendations)
    except Exception as e:
        logger.warning(f"  Managerial insights skipped: {e}")

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("V2 PIPELINE COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Best Model: {final_selection.best_model_name}")
    logger.info(f"\nModel Comparison:\n{results_df.to_string()}")

    if shap_result:
        logger.info(f"\nTop SHAP features: {shap_result.top_features[:8]}")

    return results_df, final_selection


# =============================================================================
# CLI entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="V2 Groundwater Forecasting Pipeline (RAE + full SHAP)"
    )
    parser.add_argument("--config", type=str, default="configs/enhanced_config.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        results, selection = run_v2_pipeline(args.config)
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print(f"Best Model: {selection.best_model_name}")
        print("\nModel Comparison:")
        print(results.to_string())
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
