# Groundwater Quality Forecasting Framework

A modular, decision-ready spatio-temporal forecasting framework for groundwater usability risk classification, designed for publication in the **Journal of Environmental Management (JEM)**.

## Overview

This framework implements next-year (t → t+1) forecasting of groundwater quality classification (C#S# format) using machine learning models, with comprehensive support for:

- **4 Forecasting Models**: CatBoost, LightGBM, GRU, LSTM
- **Multi-objective Optimization**: PSO-GWO hybrid metaheuristic
- **Two-level Model Selection**: VIKOR MCDM for config and model selection
- **Explainability**: TreeSHAP and Surrogate SHAP
- **Scenario Simulation**: Policy analysis for TDS, SAR, RSC perturbations
- **Publication-ready Outputs**: Figures and tables for journal submission

## Framework Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE STAGES                               │
├─────────────────────────────────────────────────────────────────┤
│ A. Data Ingestion      → B. Transition Building                  │
│ C. Data Quality        → D. Preprocessing                        │
│ E. Setup               → F. PSO-GWO Optimization (4 Models)      │
│ G. VIKOR Selection     → H. SHAP Explainability                  │
│ I. Scenarios           → J. Paper Outputs                        │
└─────────────────────────────────────────────────────────────────┘
```

## Features

### Data Processing
- Column harmonization across years (handles naming variations)
- Temporal transition building with location matching
- Comprehensive data quality validation
- Leakage-safe preprocessing (fit on train only)

### Models (4 Forecasters)

| Model | Type | Features |
|-------|------|----------|
| CatBoost | Tree | Native categorical handling, early stopping |
| LightGBM | Tree | Fast gradient boosting, early stopping |
| GRU | Deep | Categorical embeddings, PyTorch-based |
| LSTM | Deep | Categorical embeddings, PyTorch-based |

### PSO-GWO Hybrid Optimization

For each model, the PSO-GWO optimizer tunes hyperparameters using multi-objective optimization:

**Objectives (all minimized):**
1. **Ordinal Distance Error**: Penalizes predictions far from true ordinal class
2. **Severe FNR**: False negative rate for high-risk classes
3. **1 - Macro F1**: Classification performance
4. **Complexity**: Number of features + model size

**Algorithm Features:**
- Particle Swarm Optimization (exploration)
- Grey Wolf Optimizer (exploitation)
- Pareto archive for non-dominated solutions
- Integrated feature selection

### Two-Level VIKOR Model Selection

**Level 1: Best Configuration per Model**
- From each model's Pareto front, VIKOR selects the best compromise configuration
- Balances all four objectives based on configurable weights

**Level 2: Best Model Selection**
- Evaluates best-configured models on test set
- VIKOR ranks models and selects final winner

### Explainability
- **TreeSHAP**: For CatBoost and LightGBM models
- **Surrogate SHAP**: For GRU/LSTM models (with fidelity checking)
- Focus analysis on high-risk predictions

### Scenario Simulation
- TDS perturbation (+10%, +20%, +30%)
- SAR perturbation (+10%, +20%)
- RSC threshold crossing (1.25, 2.5)
- Combined scenarios
- District vulnerability ranking

## Installation

```bash
# Clone or navigate to the repository
cd groundwater-forecasting

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run the full pipeline
python -m experiments.run_pipeline --config configs/main.yaml

# Run with debug logging
python -m experiments.run_pipeline --config configs/main.yaml --debug
```

## Repository Structure

```
groundwater-forecasting/
├── configs/
│   └── main.yaml              # Main configuration
├── src/
│   ├── data/                  # Data ingestion & harmonization
│   │   ├── ingestion.py
│   │   ├── harmonization.py
│   │   └── transitions.py
│   ├── data_quality/          # Validation & cleaning
│   │   ├── validator.py
│   │   ├── cleaner.py
│   │   └── report.py
│   ├── preprocessing/         # Leakage-safe pipeline
│   │   ├── pipeline.py
│   │   ├── encoders.py
│   │   └── scalers.py
│   ├── models/
│   │   ├── base.py            # Abstract interface
│   │   ├── trees/             # CatBoost, LightGBM
│   │   │   ├── catboost_model.py
│   │   │   └── lightgbm_model.py
│   │   └── deep/              # GRU, LSTM
│   │       ├── gru.py
│   │       └── lstm.py
│   ├── optimization/
│   │   ├── pso_gwo.py         # Hybrid optimizer
│   │   └── pareto.py          # Pareto archive
│   ├── decision/
│   │   ├── vikor.py           # VIKOR MCDM
│   │   └── model_selection.py # Two-level selection
│   ├── objectives/            # Multi-objective definitions
│   │   ├── calculator.py
│   │   └── definitions.py
│   ├── explain/               # SHAP modules
│   │   ├── shap_tree.py
│   │   ├── surrogate.py
│   │   └── fidelity.py
│   ├── scenarios/             # Scenario simulation
│   │   └── engine.py
│   ├── reporting/             # Paper outputs
│   │   ├── figures.py
│   │   ├── tables.py
│   │   └── managerial.py
│   ├── evaluation/            # CV & metrics
│   ├── feature_selection/
│   ├── imbalance/
│   └── utils/
├── experiments/
│   └── run_pipeline.py        # Main entrypoint
├── tests/
├── outputs/
│   └── paper_outputs/
│       ├── figures/
│       └── tables/
├── requirements.txt
└── README.md
```

## Configuration

The main configuration file (`configs/main.yaml`) controls all pipeline settings:

```yaml
# Data paths
data:
  base_dir: "path/to/data"
  files:
    2018: "ground_water_quality_2018_post.csv"
    2019: "ground_water_quality_2019_post.csv"
    2020: "ground_water_quality_2020_post.csv"

# All 4 models
models:
  catboost:
    enabled: true
    params:
      iterations: [100, 500, 1000]
      depth: [4, 6, 8, 10]
      ...
  lightgbm:
    enabled: true
    ...
  gru:
    enabled: true
    params:
      hidden_size: [32, 64, 128]
      ...
  lstm:
    enabled: true
    ...

# PSO-GWO Optimization
optimization:
  algorithm: "pso_gwo"
  population_size: 30
  max_iterations: 50
  pso:
    w: 0.7
    c1: 1.5
    c2: 1.5
  gwo:
    a_start: 2.0
    a_end: 0.0
  hybrid_weight: 0.5

# Multi-objective weights
objectives:
  ordinal_distance:
    enabled: true
    weight: 1.0
  severe_fnr:
    enabled: true
    weight: 1.5
  macro_f1:
    enabled: true
    weight: 1.0
  complexity:
    enabled: true
    weight: 0.5

# VIKOR MCDM
mcdm:
  method: "vikor"
  vikor:
    v: 0.5
  objective_weights:
    ordinal_distance: 0.25
    severe_fnr: 0.35
    macro_f1: 0.25
    complexity: 0.15
```

## Classification System

The framework predicts groundwater quality classes in C#S# format:

| Component | Range | Interpretation |
|-----------|-------|----------------|
| C (Salinity) | 1-4 | C1=Excellent, C4=Very High |
| S (Sodium) | 1-4 | S1=Excellent, S4=Very High |

**High-Risk Classes**: C4S1, C4S2, C4S3, C4S4, C3S3, C3S4

## Pipeline Outputs

### Figures
- **F1**: Framework flowchart (Mermaid)
- **F2**: Class distribution per year
- **F3**: Temporal forecasting schematic
- **F4**: Model comparison (all 4 models)
- **F5**: Pareto fronts per model
- **F6**: VIKOR rankings
- **F7**: SHAP feature importance
- **F8**: Scenario impact analysis
- **F9**: Spatial risk map

### Tables
- **T1**: Dataset overview + missingness
- **T2**: Label mapping + high-risk definition
- **T3**: Hyperparameter search spaces
- **T4**: Best configurations per model
- **T5**: Test performance with metrics
- **T6**: SHAP feature drivers
- **T7**: Scenario outcomes

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- catboost, lightgbm
- torch (for GRU/LSTM)
- shap
- matplotlib, seaborn
- pyyaml

## Citation

If you use this framework, please cite:

```bibtex
@article{groundwater2024,
  title={Decision-Ready Spatio-Temporal Forecasting Framework for Groundwater Usability Risk},
  journal={Journal of Environmental Management},
  year={2024}
}
```

## License

MIT License

## Acknowledgments

- Groundwater quality data from public monitoring programs
- Open-source ML libraries (CatBoost, LightGBM, PyTorch, SHAP)
