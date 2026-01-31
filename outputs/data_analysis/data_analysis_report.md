# Data Distribution Analysis Report
## Key Findings
### 1. Class Imbalance
- Year 2018.0: 11.0 classes, imbalance ratio 234.0, high-risk 12.0%
- Year 2019.0: 8.0 classes, imbalance ratio 234.0, high-risk 12.6%
- Year 2020.0: 9.0 classes, imbalance ratio 228.0, high-risk 13.9%

### 2. Covariate Shift
- Features with significant shift: 5/15
- Shifted features:
  - gwl: Cohen's d = -0.978
  - hco3: Cohen's d = 0.254
  - f: Cohen's d = -0.270
  - so4: Cohen's d = -0.301
  - ca: Cohen's d = 0.244

### 3. Feature Importance for Classification
- Most discriminative features (ANOVA F-stat):
  - ec: F = 340.01
  - tds: F = 340.01
  - na: F = 294.37
  - sar: F = 279.81
  - cl: F = 120.96

### 4. Outlier Analysis
- Features with >5% outliers:
  - co3: 15.9%
  - so4: 12.1%
  - k: 11.8%
  - no3: 6.9%
  - sar: 6.3%
  - na: 6.1%

## Preprocessing Recommendations

1. **Address Class Imbalance**:
   - Current high-risk samples are minority class (~13-15%)
   - Consider: SMOTE/ADASYN, class weights, or focal loss
   - Ensure stratified splits maintain class distribution

2. **Handle Covariate Shift**:
   - Train and test distributions differ for some features
   - Consider: Domain adaptation, importance weighting
   - Alternatively: Use more robust features

3. **Feature Engineering**:
   - Focus on most discriminative features (EC, TDS, SAR, Na, Cl)
   - Consider interaction terms and ratio features
   - Dimensionality reduction may help

4. **Outlier Treatment**:
   - Use RobustScaler instead of StandardScaler
   - Or apply winsorization/clipping before scaling

5. **Temporal Considerations**:
   - Class transitions between years show significant shifts
   - Model should account for temporal dynamics
   - Consider using current-year class as additional feature

6. **Model-Specific Suggestions**:
   - For tree models: native handling of categoricals, focus on class weights
   - For deep models: ensure proper scaling, use embeddings for categoricals
   - LightGBM predicting zero high-risk -> adjust decision threshold or use calibration
