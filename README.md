# Decision-Oriented Forecasting of Groundwater Usability for Irrigation and Livestock Management

A modular, decision-ready tabular forecasting framework for groundwater usability risk classification, designed for publication in the **Journal of Environmental Management (JEM)**.

## Overview

This framework implements next-year (t → t+1) forecasting of groundwater quality using a **3-tier semantic risk classification** based on the USDA salinity-sodium hazard chart. It compares **6 forecasting models** spanning tree-based gradient boosting and modern neural architectures, with multi-objective optimization, ordinal-aware MCDM selection, and full explainability support.

- **6 Forecasting Models**: CatBoost, LightGBM, XGBoost, FT-Transformer, SpatialGNN, CORAL
- **Two Data Sources**: groundwater hydrochemistry (primary) + ERA5-Land climate reanalysis (precipitation and volumetric soil water layer-1, leakage-safe year-matched join)
- **Multi-objective Optimization**: PSO-GWO hybrid metaheuristic
- **Two-level Model Selection**: VIKOR MCDM for configuration and model selection
- **Advanced Imbalance Handling**: Class weights + SMOTE / BorderlineSMOTE / ADASYN
- **Explainability**: TreeSHAP (tree models) and Surrogate SHAP (neural models)
- **Scale-Correct Scenario Simulation**: Policy analysis for TDS, SAR, RSC perturbations
- **Publication-ready Outputs**: Figures, tables, and animations for journal submission

## Problem Statement

**Task**: 1-year-ahead tabular classification of groundwater usability for 353 wells in Telangana, India.

**Data**: Annual post-monsoon snapshots (2018 / 2019 / 2020), USDA C#S# classification.

**Two data sources are merged**:
1. **Groundwater hydrochemistry** (primary) — district/mandal/village-level well measurements (pH, EC, TDS, ions, etc.)
2. **ERA5-Land climate reanalysis** (external enrichment, Stage A1) — total precipitation and volumetric soil water layer-1, aggregated to annual / monsoon (Jun–Sep) / pre-monsoon (Feb–May) statistics and joined to each well by nearest grid cell (~9 km) and matching year (`src/data/era5_loader.py`, `src/data/external_joiner.py`)

**Setup**:
- Training: 2018 → 2019 transition pairs (~361 samples)
- Test: 2019 → 2020 transition pairs (~356 samples)
- Features: 15 hydrochemical measurements + 5 derived + 6 ERA5 climate features (precipitation/soil moisture, annual + seasonal) + spatial lat/lon + categorical location
- Class imbalance: T3_Restricted is the minority class (~48 training samples → SMOTE with k=3)

**Why tabular (not recurrent)?** Each well has only one input year per sample. LSTM/GRU require multi-timestep sequences, making them architecturally mismatched for this dataset. The 6 models here are all designed for tabular, single-snapshot input.

## Classification System

All C#S# labels are mapped to **3 semantic risk tiers** (USDA salinity-sodium hazard chart).

| Tier | C#S# Classes | Risk Level | Agricultural Interpretation |
|------|-------------|-----------|----------------------------|
| **T1_Safe** | C1S1, C1S2, C1S3, C2S1, OG | None | Unrestricted irrigation and livestock use |
| **T2_Marginal** | C1S4, C2S2, C2S3, C3S1, C3S2 | Low | Use with caution; some crop/stock restrictions |
| **T3_Restricted** | C2S4, C3S3, C3S4, C4S1–C4S4 | High | Restricted/unsuitable; includes former T4 |

**High-risk class (for FNR metric)**: T3_Restricted only.

## Six Forecasting Models

### 1. CatBoost
Gradient boosting with native categorical feature handling.
- Passes `district`, `mandal`, `village` directly as `cat_features` for ordered target encoding
- High-iteration, slow-learning search space (up to 14 000 iterations, LR down to 0.0003)
- VIKOR objective weights account for complexity and FNR tradeoff

### 2. LightGBM
Efficient gradient boosting with leaf-wise (rather than depth-wise) tree growth.
- Fast training, lower memory footprint
- Optimized search space: extended `n_estimators` range, slow LR ceiling

### 3. XGBoost
Regularized gradient boosting with depth-wise tree growth.
- `objective='multi:softprob'`, `eval_metric='mlogloss'`
- L1 (`reg_alpha`) + L2 (`reg_lambda`) regularization tuned by PSO-GWO
- Class imbalance via `sample_weight` computed from `class_weights` config
- Native SHAP support via `XGBClassifier.get_booster()`

### 4. FT-Transformer
**Feature Tokenization Transformer** (Gorishniy et al., NeurIPS 2021).
- Each scalar feature is embedded as a `d_token`-dimensional token via a dedicated Linear layer
- A learnable CLS token is prepended; N × TransformerEncoderLayer (Pre-LN) processes all tokens
- CLS token output is passed through an MLP head to produce class logits
- Focal loss (γ=2.0) for class imbalance; cosine LR schedule; early stopping on validation loss
- No external library beyond PyTorch — pure `nn.TransformerEncoderLayer` with `batch_first=True`
- `n_heads` automatically rounded down to the nearest valid divisor of `d_token`

### 5. Spatial GNN
**K-nearest-neighbor Spatial Graph Neural Network** (GraphSAGE-style mean aggregation).
- Builds a spatial k-NN graph from lat/lon coordinates (no torch_geometric required — uses scipy.spatial.cKDTree)
- For each well, aggregates features of its k nearest training neighbors
- Concatenates `[X_self, mean(X_neighbors)]` → MLP classifier with focal loss
- Inductive: test nodes find their nearest training nodes at prediction time
- Captures hydrogeological spatial autocorrelation between neighboring wells
- Falls back to a plain MLP if `lat_gis` / `long_gis` columns are not found in features

### 6. CORAL
**COnsistent RAnk Logits** ordinal neural network (Cao et al., Pattern Recognition Letters, 2020).
- Shared MLP feature extractor → single weight vector → K-1 binary output neurons with individual biases
- Enforces P(rank≥1) ≥ P(rank≥2) ≥ ... by construction, directly encoding the T1 < T2 < T3 ordinal structure
- CORAL loss: sum of focal-binary cross-entropy over all K-1 rank thresholds
- Class probabilities recovered as: P(y=0) = 1 - P(rank≥1); P(y=k) = P(rank≥k) - P(rank≥k+1)
- Intermediate probabilities clipped to (ε, 1-ε) to prevent degenerate values

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PIPELINE STAGES                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ A. Data Ingestion         → A1. External Enrichment (ERA5 climate)          │
│    + Harmonization              precipitation + soil moisture, year-matched │
│ B. Transition Building    → C.  Data Quality + 3-Tier Classification         │
│    (t→t+1 pairs)                (T1_Safe | T2_Marginal | T3_Restricted)      │
│ B2. RAE Temporal                                                              │
│     Imputation                                                                │
│ D. Preprocessing          → D2. Imbalance Handling                           │
│    (leakage-safe)               Class weights on original dist.               │
│                                 SMOTE deferred to per-CV-fold                │
│ E. Model Candidates (6)  → F.  PSO-GWO Optimisation                          │
│    CatBoost, LightGBM,          SMOTE applied inside each fold only          │
│    XGBoost, FT-Transformer,                                                   │
│    SpatialGNN, CORAL                                                          │
│ G. VIKOR Two-Level        → H.  SHAP Explainability                          │
│    Selection                    Prior-calibrated probabilities at inference  │
│ I. Scale-Correct          → J.  Paper Outputs + Animations                   │
│    Scenario Simulation                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## PSO-GWO Optimization

For each model, the PSO-GWO hybrid optimizer searches the hyperparameter space using multi-objective optimization:

**Objectives (all minimized):**
1. **Ordinal Tier-Distance** — penalizes predictions far from the true tier (max 2 steps, 3-tier system)
2. **Severe FNR** — false negative rate for T3_Restricted (the single high-risk tier)
3. **1 − Macro F1** — classification performance across all 3 tiers
4. **Complexity** — model size (parameters/trees)

**SMOTE is applied inside each CV fold** so synthetic validation samples never leak into fold metrics.

## VIKOR MCDM Selection

**Objective weights (VIKOR):**

| Objective | Weight |
|-----------|--------|
| Severe FNR | 0.50 |
| Ordinal Distance | 0.20 |
| Macro F1 | 0.20 |
| Complexity | 0.10 |

**Level 1**: For each model's Pareto front, VIKOR selects the best compromise configuration.

**Level 2**: Best-configured models are evaluated on the temporal test set (2019→2020); VIKOR ranks models by Q-score (lowest wins).

## Explainability

- **TreeSHAP**: For CatBoost, LightGBM, and XGBoost (native SHAP support)
- **Surrogate SHAP**: For FT-Transformer, SpatialGNN, and CORAL (LightGBM surrogate with fidelity checking)
- **Prior calibration**: SMOTE-induced balanced-prior bias corrected at inference via `calibrate_priors()`

## Scenario Simulation

All perturbations applied in raw-feature space (inverse-scale → perturb → re-scale):

- TDS perturbation: +10%, +20%, +30%
- SAR perturbation: +10%, +20%
- RSC threshold crossing: 1.25, 2.5
- Combined high-salinity scenario

## Data Sources

The pipeline merges **two datasets** (Stage A1 — `src/data/external_joiner.py`):

1. **Groundwater hydrochemistry** (primary, required) — annual CSV snapshots configured under `data.files` in `configs/main.yaml` (paths resolved relative to `data.base_dir`).
2. **ERA5-Land climate reanalysis** (external enrichment, optional but recommended) — a NetCDF file (`telangana_era5_land_<years>.nc`) providing hourly `tp` (total precipitation) and `swvl1` (volumetric soil water layer-1). The loader (`src/data/era5_loader.py`) aggregates these into 6 features per well-year:
   - `era5_precip_annual_mm`, `era5_precip_monsoon_mm`, `era5_precip_premonsoon_mm`
   - `era5_soil_moisture_annual`, `era5_soil_moisture_monsoon`, `era5_soil_moisture_pre`

   Each well is matched to its nearest ERA5 grid cell (~9 km, 0.1°) and the **same year** (leakage-safe — no cross-year contamination). Set the path in `configs/main.yaml`:
   ```yaml
   data:
     external:
       era5_nc_path: "/path/to/telangana_era5_land_2018_2020.nc"
   ```
   If the file is missing, the pipeline logs a warning and continues with NaN-filled ERA5 columns (handled by downstream imputation) — the run still completes using the hydrochemistry data alone.

   > NDVI (Sentinel-2 via Google Earth Engine) integration exists in `src/data/ndvi_loader.py` but is currently disabled pending GEE authentication.

## Installation

```bash
# Clone or navigate to the repository
cd Decision-Oriented-Forecasting-of-Groundwater-Usability-for-Irrigation-and-Livestock-Management

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# PyTorch (for FT-Transformer, SpatialGNN, CORAL) — install separately if not pulled by requirements
pip install torch
```

## Usage

```bash
# Run the full pipeline
python -m experiments.run_pipeline --config configs/main.yaml

# Run with debug logging
python -m experiments.run_pipeline --config configs/main.yaml --debug
```

## Configuration

```yaml
# configs/main.yaml — key sections

# All 6 models
models:
  catboost:
    enabled: true
  lightgbm:
    enabled: true
  xgboost:
    enabled: true
  ft_transformer:
    enabled: true
  spatial_gnn:
    enabled: true
  coral:
    enabled: true

# PSO-GWO Optimization
optimization:
  algorithm: "pso_gwo"
  population_size: 20
  max_iterations: 50        # tree models (CatBoost/LightGBM/XGBoost) — fast, seconds/fit
  deep_max_iterations: 15   # neural models (FT-Transformer/SpatialGNN/CORAL) — ~1 min/fit,
                            # smaller budget keeps total runtime manageable
  pso:
    w: 0.7
    c1: 1.5
    c2: 1.5
  gwo:
    a_start: 2.0
    a_end: 0.0
  hybrid_weight: 0.5

# Imbalance handling
imbalance:
  strategy: "smote"      # SMOTE with k=3 for ~48-sample minority class
  class_weights:
    method: "balanced"

# Multi-objective weights
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

## Project Structure

```
Decision-Oriented-Forecasting-of-Groundwater-Usability-for-Irrigation-and-Livestock-Management/
├── configs/
│   └── main.yaml                  # Main configuration
├── src/
│   ├── data/                      # Data ingestion & harmonization
│   │   ├── ingestion.py
│   │   ├── harmonization.py
│   │   └── transitions.py
│   ├── data_quality/              # Validation & cleaning
│   │   ├── validator.py
│   │   ├── cleaner.py             # 3-tier semantic label mapping
│   │   └── report.py
│   ├── preprocessing/             # Leakage-safe pipeline
│   │   ├── pipeline.py
│   │   ├── enhanced_pipeline.py   # + district/mandal risk prior features
│   │   ├── encoders.py
│   │   └── scalers.py
│   ├── models/
│   │   ├── base.py                # Abstract BaseForecaster + ForecasterConfig
│   │   ├── trees/                 # Tree-based models
│   │   │   ├── catboost_model.py  # CatBoost with native categorical handling
│   │   │   ├── lightgbm_model.py  # LightGBM
│   │   │   └── xgboost_model.py   # XGBoost (NEW)
│   │   └── advanced/              # Neural models (require PyTorch)
│   │       ├── ft_transformer.py  # FT-Transformer (Gorishniy et al., NeurIPS 2021)
│   │       ├── gnn_model.py       # Spatial GNN (k-NN + mean aggregation + MLP)
│   │       └── coral_model.py     # CORAL ordinal network (Cao et al., 2020)
│   ├── optimization/
│   │   ├── pso_gwo.py             # Hybrid PSO-GWO optimizer
│   │   └── pareto.py              # Pareto archive
│   ├── decision/
│   │   ├── vikor.py               # VIKOR MCDM
│   │   └── model_selection.py     # Two-level selection
│   ├── objectives/                # Multi-objective function definitions
│   │   ├── calculator.py
│   │   └── definitions.py         # RISK_TIER_MAPPING, HIGH_RISK_CLASSES
│   ├── explain/                   # SHAP modules
│   │   ├── shap_tree.py           # TreeSHAP (tree models + XGBoost)
│   │   ├── surrogate.py           # Surrogate SHAP (neural models)
│   │   └── fidelity.py
│   ├── scenarios/                 # Scale-correct scenario simulation
│   │   └── engine.py
│   ├── reporting/                 # Paper outputs
│   │   ├── figures.py             # Static figures (F1–F9) — MODEL_COLORS palette
│   │   ├── animation.py           # Training animations
│   │   ├── tables.py
│   │   └── managerial.py
│   ├── imbalance/                 # Imbalance handling
│   │   └── handlers.py
│   ├── imputation/                # Temporal imputation (Stage B2)
│   │   └── recurrent_autoencoder.py  # RAE: GRU encoder-decoder, leakage-safe
│   ├── evaluation/                # CV & metrics
│   └── utils/
├── experiments/
│   └── run_pipeline.py            # Main pipeline entrypoint
├── outputs/
│   └── paper_outputs/
│       ├── figures/
│       └── tables/
├── requirements.txt
└── README.md
```

## Pipeline Outputs

### Figures
- **F1**: Framework flowchart (Mermaid)
- **F2**: Class distribution per tier per year
- **F3**: Temporal forecasting schematic
- **F4**: Model comparison — Macro F1, Severe FNR, Ordinal Distance (6 models, MODEL_COLORS palette)
- **F4b**: Confusion matrix for best model (3 tiers)
- **F4c**: Per-tier precision/recall/F1
- **F4d**: Severity-focused metrics across all 6 models
- **F5**: Pareto fronts per model
- **F6**: VIKOR Q-score rankings (6 models)
- **F7**: SHAP feature importance (summary, beeswarm, dependence)
- **F8**: Scenario impact analysis (risk change heatmap, class transition matrix)
- **F9**: Imbalance handling — before/after tier distribution
- **F_ROC**: ROC curves per tier
- **F_PR**: Precision-Recall curves per tier
- **F_learning_curves**: Training curves for best neural model (if selected)

### Tables
- **T1**: Dataset overview + missingness
- **T2**: 3-tier label mapping + high-risk definition
- **T3**: Hyperparameter search spaces (all 6 models)
- **T4**: Best configurations per model
- **T5**: Test performance with metrics
- **T6**: SHAP feature drivers
- **T7**: Scenario outcomes

## Requirements

- Python 3.8+
- pandas, numpy, scipy, scikit-learn
- catboost, lightgbm, xgboost
- torch (for FT-Transformer, SpatialGNN, CORAL)
- shap
- matplotlib, seaborn
- pyyaml
- imbalanced-learn (for SMOTE strategies)

## Key Design Decisions

### Why 6 models spanning trees and neural architectures?

Tree-based models (CatBoost, LightGBM, XGBoost) excel on small tabular datasets with few samples and mixed feature types. The three neural models cover complementary inductive biases:

- **FT-Transformer**: Captures cross-feature interactions via self-attention — beneficial for correlated hydrochemical parameters (SAR, TDS, EC, Na all interact)
- **SpatialGNN**: Encodes spatial autocorrelation — neighboring wells in the same aquifer system tend to have similar water quality trajectories
- **CORAL**: Exploits the natural ordinal structure T1 < T2 < T3 — physically, "more restrictive" means higher mineral concentration, not a qualitatively different category

### Why not LSTM/GRU?

Each sample in this dataset consists of exactly one input year (the source year's hydrochemical measurements) and one target year (the next year's risk tier). LSTM and GRU require multi-timestep sequences; feeding them a single time-step is equivalent to a plain MLP with unnecessary architectural overhead and no sequential modeling benefit.

### Why 3 tiers instead of 4?

The transition dataset (2018→2019 pairs) contains fewer than 3 T4_Unsafe samples — far too few for any model to learn a meaningful boundary. Merging T4 into T3 is scientifically sound: both tiers share the "do not use for irrigation/livestock" operational consequence.

### Why SMOTE inside CV folds?

Global SMOTE before splitting creates synthetic validation samples that share nearest neighbors with training rows — inflating CV metrics. By applying SMOTE only to the training fold inside each CV split, the validation slice is always original, unaugmented data.

### Why prior calibration at inference?

SMOTE trains models on a near-balanced distribution, creating an implicit uniform class prior. At inference on the imbalanced test set, this over-predicts the minority class. `calibrate_priors()` rescales softmax probabilities by the ratio of true prior to SMOTE-induced prior, then renormalizes.

## References

1. Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting Deep Learning Models for Tabular Data. *Advances in Neural Information Processing Systems*, 34.
2. Cao, W., Mirjalili, V., & Raschka, S. (2020). Rank Consistent Ordinal Regression for Neural Networks with Application to Age Estimation. *Pattern Recognition Letters*, 140, 325–331.
3. Hamilton, W., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. *Advances in Neural Information Processing Systems*, 30.

## Citation

```bibtex
@article{groundwater2025,
  title={Decision-Oriented Forecasting of Groundwater Usability for Irrigation and Livestock Management},
  journal={Journal of Environmental Management},
  year={2025}
}
```

## License

MIT License

## Acknowledgments

- Groundwater quality data from public monitoring programs in Telangana, India
- Open-source ML libraries: CatBoost, LightGBM, XGBoost, PyTorch, SHAP
- imbalanced-learn for SMOTE strategies
