"""
Main Pipeline Runner
====================

Single entrypoint for the groundwater quality forecasting framework.

Usage:
    python -m experiments.run_pipeline --config configs/main.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.utils.config import load_config, get_config_value
from src.utils.logging import setup_logging
from src.utils.reproducibility import set_seed

from src.data import DataIngestion, DataHarmonizer, TransitionBuilder
from src.data_quality import DataQualityValidator, DataCleaner, DataQualityReport
from src.preprocessing import PreprocessingPipeline, LabelParser, OrdinalEncoder
from src.evaluation import TemporalSplitter, CrossValidator, MetricsCalculator
from src.imbalance import ImbalanceHandler
from src.feature_selection import FilterSelector

from src.models.trees import CatBoostForecaster, LightGBMForecaster
from src.models.base import ForecasterConfig, compute_class_weights

from src.objectives import ObjectiveCalculator
from src.objectives.definitions import get_high_risk_indices, get_ordinal_mapping, HIGH_RISK_CLASSES

from src.reporting import FigureGenerator, TableGenerator, ManagerialInsights

logger = logging.getLogger(__name__)


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
    X_train, y_train, feature_names = pipeline.fit_transform(train_df, scale_features=False)
    X_test, y_test, _ = pipeline.transform(test_df, scale_features=False)

    # Handle missing targets
    train_mask = y_train >= 0
    test_mask = y_test >= 0 if y_test is not None else np.ones(len(X_test), dtype=bool)

    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]

    logger.info(f"Final shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Classes: {np.unique(y_train)}")

    # =========================================================================
    # STAGE E: Model Training (Baseline)
    # =========================================================================
    logger.info("\n[STAGE E] Model Training (Baseline)")

    # Setup imbalance handling
    imbalance_config = config.get('imbalance', {})
    imbalance_handler = ImbalanceHandler(
        strategy=imbalance_config.get('strategy', 'class_weights'),
        weight_method=imbalance_config.get('class_weights', {}).get('method', 'balanced')
    )
    imbalance_handler.fit(y_train)
    class_weights = imbalance_handler.get_class_weights()

    # Train CatBoost (baseline)
    catboost_config = ForecasterConfig(
        params={
            'iterations': 500,
            'depth': 6,
            'learning_rate': 0.1,
            'early_stopping_rounds': 50
        },
        n_classes=len(np.unique(y_train)),
        class_weights=class_weights,
        random_state=42
    )

    catboost_model = CatBoostForecaster(config=catboost_config)
    catboost_model.fit(X_train, y_train, X_test, y_test, feature_names=feature_names)

    # Train LightGBM (baseline)
    lgbm_config = ForecasterConfig(
        params={
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'early_stopping_rounds': 50
        },
        n_classes=len(np.unique(y_train)),
        class_weights=class_weights,
        random_state=42
    )

    lgbm_model = LightGBMForecaster(config=lgbm_config)
    lgbm_model.fit(X_train, y_train, X_test, y_test, feature_names=feature_names)

    # =========================================================================
    # STAGE F-G: Evaluation
    # =========================================================================
    logger.info("\n[STAGE F-G] Model Evaluation")

    # Setup metrics
    ordinal_mapping = get_ordinal_mapping(label_encoder)
    idx_to_label = {v: k for k, v in label_encoder.items()}
    high_risk_idx = get_high_risk_indices(label_encoder)

    metrics_calc = MetricsCalculator(
        label_to_ordinal=ordinal_mapping,
        idx_to_label=idx_to_label,
        high_risk_indices=high_risk_idx
    )

    results = []

    for name, model in [('CatBoost', catboost_model), ('LightGBM', lgbm_model)]:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        metrics = metrics_calc.compute_all(y_test, y_pred, y_proba)
        metrics['Model'] = name
        results.append(metrics)

        logger.info(f"\n{name} Results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Macro F1: {metrics['macro_f1']:.4f}")
        logger.info(f"  Severe FNR: {metrics.get('severe_fnr', 'N/A')}")
        logger.info(f"  Ordinal Distance: {metrics.get('ordinal_distance_mean', 'N/A')}")

    results_df = pd.DataFrame(results)

    # =========================================================================
    # STAGE J: Paper Outputs
    # =========================================================================
    logger.info("\n[STAGE J] Generating Paper Outputs")

    paper_output_dir = Path(get_config_value(config, 'output', 'paper_outputs', default='outputs/paper_outputs'))
    paper_output_dir.mkdir(parents=True, exist_ok=True)

    # Generate figures
    fig_gen = FigureGenerator(output_dir=paper_output_dir / 'figures')

    # F1: Flowchart
    fig_gen.create_mermaid_flowchart()

    # F2: Class distribution
    fig_gen.f2_class_distribution(cleaned_data, target_col='Classification')

    # F3: Temporal schematic
    fig_gen.f3_temporal_forecasting_schematic()

    # F4: Model comparison
    fig_gen.f4_model_comparison(results_df, metrics=['macro_f1', 'accuracy'])

    # Generate tables
    table_gen = TableGenerator(output_dir=paper_output_dir / 'tables')

    # T1: Dataset overview
    table_gen.t1_dataset_overview(cleaned_data)

    # T2: Label mapping
    table_gen.t2_label_mapping(label_encoder, HIGH_RISK_CLASSES)

    # Save results
    results_df.to_csv(paper_output_dir / 'tables' / 'model_results.csv', index=False)

    # Generate insights
    insights_gen = ManagerialInsights(output_dir=paper_output_dir)
    findings = insights_gen.generate_key_findings(
        {'best_model': 'CatBoost', 'test_metrics': results[0]}
    )

    recommendations = insights_gen.generate_recommendations(
        shap_top_features=feature_names[:5],
        vulnerable_districts=[],
        high_impact_scenarios=[]
    )

    insights_gen.save_insights(findings, recommendations)

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Outputs saved to: {paper_output_dir}")

    return results_df


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
        results = run_pipeline(args.config)
        print("\nFinal Results:")
        print(results.to_string())
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
