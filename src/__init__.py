"""
Groundwater Quality Forecasting Framework
==========================================

A modular research framework for spatio-temporal groundwater quality forecasting,
designed for publication in the Journal of Environmental Management (JEM).

Modules:
--------
- data: Data ingestion and harmonization
- data_quality: Data validation and cleaning
- preprocessing: Leakage-safe preprocessing pipeline
- models: Forecasting models (tree-based and deep learning)
- optimization: PSO-GWO optimization and Pareto archive
- objectives: Multi-objective evaluation metrics
- decision: MCDM methods (VIKOR) for model selection
- explain: SHAP explainability modules
- scenarios: Scenario simulation engine
- reporting: Paper outputs (figures and tables)
- evaluation: Cross-validation and evaluation utilities
- feature_selection: Feature selection strategies
- imbalance: Class imbalance handling
- utils: Common utilities

Author: Research Framework
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Research Framework"

from pathlib import Path

# Package root
PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent
