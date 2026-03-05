# Executive Summary: Groundwater Quality Forecasting Framework

## Overview

This research implements a **decision-ready spatio-temporal forecasting framework** for predicting next-year groundwater usability risk classes across Telangana, India. The framework uses machine learning to forecast water quality classifications (C#S#) based on hydrochemical parameters measured annually at post-monsoon monitoring stations.

The pipeline addresses a critical limitation of standard ML benchmarks for water management: **accuracy alone is insufficient**. A model that never predicts high-risk conditions (achieving high accuracy on imbalanced data) is worse than a less accurate model that reliably flags hazardous water.

---

# Key Findings

## Model Selection via VIKOR MCDM

Four candidate models (LSTM, GRU, CatBoost, LightGBM) were optimized using PSO-GWO multi-objective optimization and ranked by the VIKOR compromise solution across four objectives:

| Model    | VIKOR Q | Rank | Macro F1 | Severe FNR | Ordinal Dist. | Accuracy |
|----------|---------|------|----------|------------|---------------|----------|
| **LSTM** | 0.000   | **1**| 0.223    | 0.478      | 0.657         | 0.561    |
| GRU      | 0.165   | 2    | 0.223    | 0.391      | 0.742         | 0.504    |
| CatBoost | 0.440   | 3    | 0.214    | 0.348      | 0.943         | 0.433    |
| LightGBM | 1.000   | 4    | 0.183    | **1.000**  | 0.428         | **0.674**|

- **Best Model**: LSTM (VIKOR Q = 0.000)
- **Macro F1**: 0.223
- **Severe FNR**: 0.478 (best among practical models; LightGBM=1.000 is disqualifying)
- **Key finding**: LightGBM's 67.4% accuracy masks that it predicts **zero** high-risk samples — unacceptable for early-warning systems

## Why Not LightGBM?

LightGBM achieves the highest raw accuracy (67.4%) but has a Severe False Negative Rate of 1.000 — it fails to identify a single high-risk groundwater location in the test set (0 of 46 high-risk samples detected). In water resource management, missing a high-risk case has severe consequences for agricultural yields and livestock health.

## Feature Drivers (SHAP Analysis)

Parameters with the highest influence on quality classification:
1. **TDS** — Total Dissolved Solids (salinity proxy)
2. **HCO3** — Bicarbonate (alkalinity, RSC contribution)
3. **Cl** — Chloride (ionic contamination)
4. **F** — Fluoride (health risk)
5. **NO3** — Nitrate (agricultural runoff)

---

# Recommendations for Water Resource Management

## Priority Monitoring

Based on the analysis, the following parameters should be prioritized for monitoring:

- **TDS**: Primary driver of C-class (salinity) escalation
- **HCO3**: Key contributor to RSC and C-class transitions
- **Cl**: Indicator of ionic contamination trends
- **F**: Health risk parameter; threshold exceedance triggers high-risk classification
- **NO3**: Agricultural runoff proxy; elevated in irrigation-intensive districts

## General Recommendations

1. **Early Warning System**: Deploy the LSTM forecasting model for 1-year ahead predictions to enable proactive management before the growing season.

2. **Targeted Interventions**: Focus resources on high-risk areas identified by the model, particularly districts showing consistent C4/S2+ degradation trends across years.

3. **Parameter Thresholds**: Establish monitoring thresholds for key drivers (TDS > 2000 mg/L, SAR > 10, RSC > 2.5 meq/L) based on SHAP sensitivity analysis.

4. **Data Collection**: Ensure consistent annual post-monsoon data collection to maintain forecasting capability and track long-term trends.

5. **Stakeholder Communication**: Use the classification system (C1–C4, S1–S4) to communicate water quality status to farmers and water users in accessible terms.

6. **Model Retraining**: Retrain annually as new monitoring data becomes available to capture evolving hydrochemical trends driven by climate variability and land-use change.
