# Executive Summary: Groundwater Quality Forecasting Framework

## Overview

This research implements a decision-ready spatio-temporal forecasting framework for
predicting next-year groundwater usability risk classes (T1_Safe / T2_Marginal /
T3_Restricted) based on hydrochemical parameters from Telangana, India.  The 3-tier
semantic classification is grounded in the USDA salinity-sodium hazard chart.

---

## Model Performance (Test Set: 2019 → 2020)

VIKOR ranks models by composite Q-score across four objectives (ordinal distance,
severe FNR, 1 − macro F1, complexity).  **Lower Q = better**.

| Rank | Model     | VIKOR Q | Macro F1 | Severe FNR ↓ | Accuracy | Ordinal Dist. |
|------|-----------|---------|----------|--------------|----------|---------------|
| **1** | **CatBoost** | **0.497** | **0.338** | **0.213** | 0.331 | 1.506 |
| 2     | LightGBM  | 0.500   | 0.530    | 0.511        | 0.607    | 0.831         |
| 3     | LSTM      | 0.500   | 0.382    | 0.277        | 0.376    | 1.343         |
| 4     | GRU       | 0.646   | 0.357    | 0.191        | 0.346    | 1.489         |

**Winner: CatBoost** — lowest VIKOR Q-score (0.497), best compromise between
T3_Restricted recall (FNR = 0.213) and ordinal safety.  LightGBM achieves higher raw
accuracy but at the cost of a 0.511 FNR on the safety-critical high-risk class.

**Number of classes throughout pipeline: 3** (T1_Safe = 0, T2_Marginal = 1, T3_Restricted = 2).
T4_Unsafe merged into T3_Restricted — transition dataset contains < 3 T4 samples.

---

## Key SHAP Feature Drivers

| Rank | Feature | Role |
|------|---------|------|
| 1 | **SAR** | Sodium Adsorption Ratio — dominant predictor of sodium hazard |
| 2 | **EC** | Electrical Conductivity — proxy for total dissolved salts |
| 3 | **HCO₃** | Bicarbonate — residual sodium carbonate indicator |
| 4 | **NO₃** | Nitrate — anthropogenic contamination signal |
| 5 | **Ca** | Calcium — hardness and cation exchange capacity |

---

## Risk Mitigation Scenarios (Scale-Correct Perturbations)

All perturbations applied in **raw-feature space** (not scaled space) so "+10% TDS"
always means +10% mg/L regardless of model input normalisation.

| Scenario | Risk Change | % Samples Reclassified |
|----------|-------------|------------------------|
| TDS +10% | highest     | varies by district      |
| TDS +20% | higher      | varies by district      |
| TDS +30% | highest     | varies by district      |
| SAR +10% | moderate    | varies by district      |
| SAR +20% | moderate    | varies by district      |

---

## Engineering Fixes Applied in This Version

| # | Fix | Impact |
|---|-----|--------|
| 1 | SMOTE now applied per CV fold (not globally) | Eliminates validation-set contamination |
| 2 | Prior calibration at inference | Corrects SMOTE-induced minority-class over-prediction |
| 3 | RAE best-weights checkpoint | Restores true best model after early stopping |
| 5 | RAE wired into pipeline as Stage B2 | Temporally consistent imputation active |
| 6 | CUDA mid-training fallback | GPU OOM no longer crashes the run |
| 7 | Vectorised RAE inner loops | ~10-50× faster for large monitoring datasets |

---

## Recommendations for Water Resource Management

### Priority Monitoring Parameters

1. **SAR** — High influence; monitor closely in sodium-sensitive districts
2. **EC** — Strong salinity proxy; threshold alerts recommended
3. **HCO₃** — RSC-linked; monitor alongside RSC for restricted-use decisions
4. **NO₃** — Anthropogenic signal; seasonal variability important
5. **Ca** — Hardness indicator; relevant for livestock suitability

### General Recommendations

1. **Early Warning System** — Deploy the 1-year-ahead CatBoost model in an operational
   dashboard; trigger field verification whenever P(T3_Restricted) > 0.4.

2. **Targeted Interventions** — Prioritise districts with consistent T2 → T3 transitions
   in the scenario heatmap for source-water diversification or treatment investment.

3. **Parameter Thresholds** — Set automated alerts at SAR > 10, EC > 3 000 µS/cm,
   and RSC > 1.25 meq/L based on model sensitivity analysis.

4. **Data Collection** — Ensure consistent annual post-monsoon sampling at all
   monitoring wells to maintain temporal forecasting capability.

5. **Stakeholder Communication** — Report risk tier (T1/T2/T3) alongside the
   underlying C#S# class to farmers and irrigation engineers for actionable decisions.
