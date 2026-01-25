# Groundwater Quality Forecasting Framework

A modular, decision-ready spatio-temporal forecasting framework for groundwater usability risk classification, designed for publication in the **Journal of Environmental Management (JEM)**.

## Overview

This framework implements next-year (t → t+1) forecasting of groundwater quality classification (C#S# format) using machine learning models, with comprehensive support for:

- **Multi-objective optimization** (PSO-GWO hybrid)
- **Pareto-optimal model selection** (VIKOR MCDM)
- **Explainability** (TreeSHAP and surrogate SHAP)
- **Scenario simulation** for policy analysis
- **Publication-ready outputs**

## Framework Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE STAGES                               │
├─────────────────────────────────────────────────────────────────┤
│ A. Data Ingestion    → B. Transition Building                    │
│ C. Data Quality      → D. Preprocessing                          │
│ E. Model Training    → F. PSO-GWO Optimization                   │
│ G. VIKOR Selection   → H. SHAP Explainability                    │
│ I. Scenarios         → J. Paper Outputs                          │
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
| CatBoost | Tree | Native categorical handling |
| LightGBM | Tree | Fast gradient boosting |
| GRU | Deep | Categorical embeddings |
| LSTM | Deep | Categorical embeddings |

### Optimization
- **PSO-GWO Hybrid**: Combines exploration (PSO) with exploitation (GWO)
- **Multi-objective**: Ordinal distance, Severe FNR, MacroF1, Complexity
- **Pareto Archive**: Maintains non-dominated solutions
- **Feature Selection**: Integrated wrapper-based selection

### Decision Making
- **VIKOR MCDM**: Compromise solution ranking
- **Two-level Selection**:
  1. Best configuration per model
  2. Best model across types

### Explainability
- TreeSHAP for tree models
- Surrogate SHAP for deep models (with fidelity checking)
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
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run the full pipeline
python -m experiments.run_pipeline --config configs/main.yaml
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
│   │   └── deep/              # GRU, LSTM
│   ├── optimization/
│   │   ├── pso_gwo.py         # Hybrid optimizer
│   │   └── pareto.py          # Pareto archive
│   ├── objectives/            # Multi-objective definitions
│   ├── decision/
│   │   ├── vikor.py           # VIKOR MCDM
│   │   └── model_selection.py
│   ├── explain/               # SHAP modules
│   │   ├── shap_tree.py
│   │   ├── surrogate.py
│   │   └── fidelity.py
│   ├── scenarios/             # Scenario simulation
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
│   ├── test_data.py
│   ├── test_models.py
│   └── test_optimization.py
├── outputs/
│   └── paper_outputs/
│       ├── figures/
│       └── tables/
├── requirements.txt
└── README.md
```

## Configuration

The main configuration file (`configs/main.yaml`) controls:

```yaml
# Data paths
data:
  base_dir: "path/to/data"
  files:
    2018: "ground_water_quality_2018_post.csv"
    2019: "ground_water_quality_2019_post.csv"
    2020: "ground_water_quality_2020_post.csv"

# Model settings
models:
  catboost:
    enabled: true
    params:
      iterations: [100, 500, 1000]
      depth: [4, 6, 8, 10]
      ...

# Optimization
optimization:
  algorithm: "pso_gwo"
  population_size: 30
  max_iterations: 50

# Objectives
objectives:
  ordinal_distance:
    enabled: true
    weight: 1.0
  severe_fnr:
    enabled: true
    weight: 1.5
  ...
```

## Classification System

The framework predicts groundwater quality classes in C#S# format:

| Component | Range | Interpretation |
|-----------|-------|----------------|
| C (Salinity) | 1-4 | C1=Excellent, C4=Very High |
| S (Sodium) | 1-4 | S1=Excellent, S4=Very High |

**High-Risk Classes**: C4S1, C4S2, C4S3, C4S4, C3S3, C3S4

## Outputs

### Figures
- F1: Framework flowchart (Mermaid)
- F2: Class distribution per year
- F3: Temporal forecasting schematic
- F4: Model comparison
- F5: Pareto fronts
- F6: VIKOR rankings
- F7: SHAP summary
- F8: Scenario impact
- F9: Spatial risk map

### Tables
- T1: Dataset overview + missingness
- T2: Label mapping + high-risk definition
- T3: Hyperparameter bounds
- T4: Best configurations (VIKOR)
- T5: Test performance + CIs
- T6: SHAP drivers
- T7: Scenario outcomes

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
