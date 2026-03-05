# Key Findings

## Model Performance

| Model    | VIKOR Rank | Macro F1 | Severe FNR | Ordinal Distance | Accuracy |
|----------|-----------|----------|------------|-----------------|----------|
| **LSTM** | **1 ★**   | 0.223    | 0.478      | 0.657           | 0.561    |
| GRU      | 2         | 0.223    | 0.391      | 0.742           | 0.504    |
| CatBoost | 3         | 0.214    | 0.348      | 0.943           | 0.433    |
| LightGBM | 4         | 0.183    | **1.000**  | 0.428           | 0.674    |

## Why LSTM Wins (Multi-Criteria Decision Making)

LSTM was selected by the VIKOR MCDM framework (Q=0.000) as the best overall model because it achieves the best **balance** across four decision-critical objectives:

1. **Ordinal Distance** (weight 0.25): LSTM correctly predicts classes close to the true class, limiting misclassification severity.
2. **Severe FNR** (weight 0.35): LSTM misses fewer high-risk cases than LightGBM (0.48 vs 1.00). LightGBM's perfect accuracy hides that it predicts **zero** high-risk samples, making it dangerous for early-warning use.
3. **Macro F1 Complement** (weight 0.25): LSTM achieves the joint-highest Macro F1 (0.223) across all models.
4. **Complexity** (weight 0.15): LSTM uses only 6 features after PSO-GWO feature selection.

> **Critical insight**: LightGBM has the highest accuracy (67.4%) but a Severe FNR of 1.0 — it misses every single high-risk groundwater sample. This makes it unacceptable for decision-oriented water resource management.

## Feature Importance (SHAP)

Top parameters driving groundwater quality classification:
- **TDS** (Total Dissolved Solids) — primary salinity driver
- **HCO3** (Bicarbonate) — alkalinity indicator
- **Cl** (Chloride) — ionic contamination marker
- **F** (Fluoride) — health risk indicator
- **NO3** (Nitrate) — agricultural runoff proxy

## Dataset Characteristics

- **Years covered**: 2018, 2019, 2020 (post-monsoon)
- **Train set**: 2018→2019 transitions
- **Test set**: 2019→2020 transitions
- **Severe class imbalance**: Some classes (C3S4, C4S4) have ≤1 sample — grouped via PSO-GWO feature selection and class-weighted training
- **High-risk classes**: C3S2, C4S1, C4S2 (46 test samples)
