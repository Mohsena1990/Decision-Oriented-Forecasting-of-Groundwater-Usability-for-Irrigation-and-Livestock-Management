"""
Data Distribution Analysis Script
=================================

This script provides comprehensive analysis of the groundwater quality data
to diagnose preprocessing issues and understand why model performance is low.

Key analyses:
1. Class distribution and imbalance analysis
2. Feature distributions (histograms, statistics)
3. Train vs Test distribution comparison (covariate shift detection)
4. Feature correlations with target
5. Temporal patterns and distribution shifts
6. Outlier analysis
7. Class separability analysis

Run this script to generate diagnostic outputs for preprocessing improvements.
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
DATA_DIR = Path("/Users/mohsenasghariilani/Downloads/archive")  # Update if needed
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "data_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature definitions
NUMERIC_FEATURES = ['gwl', 'pH', 'EC', 'TDS', 'CO3', 'HCO3', 'Cl', 'F', 'NO3',
                    'SO4', 'Na', 'K', 'Ca', 'Mg', 'TH', 'SAR', 'RSC']
CATEGORICAL_FEATURES = ['district', 'mandal', 'village']
SPATIAL_FEATURES = ['lat_gis', 'long_gis']
TARGET_COL = 'Classification'

# High-risk classes
HIGH_RISK_CLASSES = ['C4S1', 'C4S2', 'C4S3', 'C4S4', 'C3S3', 'C3S4']


def load_data():
    """Load all year data."""
    data = {}
    files = {
        2018: "ground_water_quality_2018_post.csv",
        2019: "ground_water_quality_2019_post.csv",
        2020: "ground_water_quality_2020_post.csv"
    }

    for year, filename in files.items():
        filepath = DATA_DIR / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['year'] = year
            data[year] = df
            print(f"Loaded {year}: {len(df)} records")
        else:
            print(f"WARNING: File not found: {filepath}")

    return data


def harmonize_columns(df):
    """Standardize column names."""
    # Lowercase all columns
    df.columns = df.columns.str.lower().str.strip()

    # Common mappings
    mapping = {
        'e.c': 'ec', 'e.c.': 'ec',
        'ph ': 'ph',
        'na+': 'na', 'sodium': 'na',
        'k+': 'k', 'potassium': 'k',
        'ca+2': 'ca', 'ca++': 'ca', 'calcium': 'ca',
        'mg+2': 'mg', 'mg++': 'mg', 'magnesium': 'mg',
        'co3--': 'co3', 'co3-2': 'co3',
        'hco3-': 'hco3',
        'cl-': 'cl', 'chloride': 'cl',
        'so4--': 'so4', 'so4-2': 'so4',
        'no3-': 'no3',
        'rsc meq / l': 'rsc', 'rsc meq/l': 'rsc',
        'classification ': 'classification'
    }

    df.columns = [mapping.get(c, c) for c in df.columns]
    return df


def clean_classification(df, col='classification'):
    """Clean classification labels."""
    if col not in df.columns:
        return df

    df[col] = df[col].astype(str).str.strip().str.upper()

    # Fix common typos
    df[col] = df[col].replace({
        'O.G': 'OG',
        'O.G.': 'OG',
        'NAN': np.nan
    })

    return df


def analyze_class_distribution(data, output_dir):
    """Analyze class distribution across years."""
    print("\n" + "="*80)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*80)

    results = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (year, df) in enumerate(sorted(data.items())):
        df = harmonize_columns(df.copy())
        df = clean_classification(df)

        if 'classification' not in df.columns:
            print(f"Year {year}: 'classification' column not found")
            continue

        class_counts = df['classification'].value_counts()

        print(f"\n--- Year {year} ---")
        print(f"Total samples: {len(df)}")
        print(f"Unique classes: {len(class_counts)}")
        print("\nClass distribution:")
        for cls, count in class_counts.items():
            pct = 100 * count / len(df)
            is_high_risk = "HIGH-RISK" if cls in HIGH_RISK_CLASSES else ""
            print(f"  {cls}: {count:5d} ({pct:5.1f}%) {is_high_risk}")

        # High-risk analysis
        high_risk_count = df['classification'].isin(HIGH_RISK_CLASSES).sum()
        print(f"\nHigh-risk samples: {high_risk_count} ({100*high_risk_count/len(df):.1f}%)")

        # Imbalance ratio
        max_class = class_counts.max()
        min_class = class_counts.min()
        imbalance_ratio = max_class / min_class if min_class > 0 else np.inf
        print(f"Imbalance ratio (max/min): {imbalance_ratio:.1f}")

        results.append({
            'year': year,
            'n_samples': len(df),
            'n_classes': len(class_counts),
            'high_risk_count': high_risk_count,
            'high_risk_pct': 100*high_risk_count/len(df),
            'imbalance_ratio': imbalance_ratio
        })

        # Plot
        ax = axes[i]
        colors = ['red' if cls in HIGH_RISK_CLASSES else 'steelblue'
                  for cls in class_counts.index]
        class_counts.plot(kind='bar', ax=ax, color=colors)
        ax.set_title(f'Year {year}\n(n={len(df)})')
        ax.set_xlabel('Classification')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution_by_year.png', dpi=150, bbox_inches='tight')
    plt.close()

    return pd.DataFrame(results)


def analyze_transition_class_distribution(data, output_dir):
    """Analyze class transitions between years."""
    print("\n" + "="*80)
    print("TRANSITION CLASS ANALYSIS (Train vs Test)")
    print("="*80)

    # Create location IDs and match
    results = {}

    for (from_year, to_year) in [(2018, 2019), (2019, 2020)]:
        if from_year not in data or to_year not in data:
            continue

        df_from = harmonize_columns(data[from_year].copy())
        df_to = harmonize_columns(data[to_year].copy())

        df_from = clean_classification(df_from)
        df_to = clean_classification(df_to)

        # Create location ID
        for df in [df_from, df_to]:
            cat_parts = []
            for col in ['district', 'mandal', 'village']:
                if col in df.columns:
                    cat_parts.append(df[col].astype(str).str.upper().str.strip())

            if 'lat_gis' in df.columns and 'long_gis' in df.columns:
                lat_rounded = np.round(df['lat_gis'], 3).astype(str)
                lon_rounded = np.round(df['long_gis'], 3).astype(str)
                cat_parts.extend([lat_rounded, lon_rounded])

            df['_loc_id'] = cat_parts[0]
            for part in cat_parts[1:]:
                df['_loc_id'] = df['_loc_id'] + '|' + part

        # Match locations
        common_locs = set(df_from['_loc_id'].unique()) & set(df_to['_loc_id'].unique())

        print(f"\n--- Transition {from_year} -> {to_year} ---")
        print(f"Unique locations in {from_year}: {df_from['_loc_id'].nunique()}")
        print(f"Unique locations in {to_year}: {df_to['_loc_id'].nunique()}")
        print(f"Matched locations: {len(common_locs)}")

        # Get matched data
        df_matched_from = df_from[df_from['_loc_id'].isin(common_locs)].drop_duplicates('_loc_id')
        df_matched_to = df_to[df_to['_loc_id'].isin(common_locs)].drop_duplicates('_loc_id')

        # Sort by location ID for alignment
        df_matched_from = df_matched_from.sort_values('_loc_id').reset_index(drop=True)
        df_matched_to = df_matched_to.sort_values('_loc_id').reset_index(drop=True)

        print(f"\nTarget class distribution (y = {to_year} classification):")
        target_counts = df_matched_to['classification'].value_counts()
        for cls, count in target_counts.items():
            pct = 100 * count / len(df_matched_to)
            is_high_risk = "HIGH-RISK" if cls in HIGH_RISK_CLASSES else ""
            print(f"  {cls}: {count:4d} ({pct:5.1f}%) {is_high_risk}")

        high_risk_count = df_matched_to['classification'].isin(HIGH_RISK_CLASSES).sum()
        print(f"\nHigh-risk in target: {high_risk_count} ({100*high_risk_count/len(df_matched_to):.1f}%)")

        # Class transition matrix
        if len(df_matched_from) == len(df_matched_to):
            transition_name = f"{from_year}_{to_year}"
            results[transition_name] = {
                'n_samples': len(df_matched_from),
                'classes_from': df_matched_from['classification'].value_counts().to_dict(),
                'classes_to': df_matched_to['classification'].value_counts().to_dict(),
                'high_risk_target': high_risk_count,
                'high_risk_target_pct': 100*high_risk_count/len(df_matched_to)
            }

            # Transition matrix
            transition_matrix = pd.crosstab(
                df_matched_from['classification'],
                df_matched_to['classification'],
                margins=True
            )
            print(f"\nClass Transition Matrix ({from_year} -> {to_year}):")
            print(transition_matrix.to_string())

            # Save transition matrix
            transition_matrix.to_csv(output_dir / f'transition_matrix_{from_year}_{to_year}.csv')

    # Compare train vs test target distributions
    print("\n--- TRAIN vs TEST Comparison ---")
    if '2018_2019' in results and '2019_2020' in results:
        train_classes = results['2018_2019']['classes_to']
        test_classes = results['2019_2020']['classes_to']

        all_classes = sorted(set(train_classes.keys()) | set(test_classes.keys()))

        print(f"\n{'Class':<10} {'Train (2019)':<15} {'Test (2020)':<15} {'Shift':<10}")
        print("-" * 50)

        train_total = sum(train_classes.values())
        test_total = sum(test_classes.values())

        for cls in all_classes:
            train_pct = 100 * train_classes.get(cls, 0) / train_total
            test_pct = 100 * test_classes.get(cls, 0) / test_total
            shift = test_pct - train_pct
            print(f"{cls:<10} {train_pct:>12.1f}%   {test_pct:>12.1f}%   {shift:>+7.1f}%")

    return results


def analyze_feature_distributions(data, output_dir):
    """Analyze numeric feature distributions."""
    print("\n" + "="*80)
    print("FEATURE DISTRIBUTION ANALYSIS")
    print("="*80)

    # Combine all data
    all_dfs = []
    for year, df in data.items():
        df = harmonize_columns(df.copy())
        df['year'] = year
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    # Identify available numeric features
    available_numeric = [f.lower() for f in NUMERIC_FEATURES if f.lower() in combined.columns]

    print(f"\nAvailable numeric features: {len(available_numeric)}")
    print(f"Features: {available_numeric}")

    # Statistics per feature
    stats_list = []
    for feature in available_numeric:
        values = pd.to_numeric(combined[feature], errors='coerce').dropna()

        stat = {
            'feature': feature,
            'count': len(values),
            'missing': len(combined) - len(values),
            'missing_pct': 100 * (len(combined) - len(values)) / len(combined),
            'mean': values.mean(),
            'std': values.std(),
            'min': values.min(),
            'q25': values.quantile(0.25),
            'median': values.median(),
            'q75': values.quantile(0.75),
            'max': values.max(),
            'skewness': stats.skew(values),
            'kurtosis': stats.kurtosis(values)
        }
        stats_list.append(stat)

        print(f"\n{feature.upper()}")
        print(f"  Range: [{stat['min']:.2f}, {stat['max']:.2f}]")
        print(f"  Mean: {stat['mean']:.2f}, Std: {stat['std']:.2f}")
        print(f"  Median: {stat['median']:.2f}")
        print(f"  Skewness: {stat['skewness']:.2f}, Kurtosis: {stat['kurtosis']:.2f}")

    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(output_dir / 'feature_statistics.csv', index=False)

    # Plot distributions
    n_features = len(available_numeric)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axes = axes.flatten()

    for i, feature in enumerate(available_numeric):
        ax = axes[i]
        for year in sorted(data.keys()):
            df = harmonize_columns(data[year].copy())
            if feature in df.columns:
                values = pd.to_numeric(df[feature], errors='coerce').dropna()
                ax.hist(values, bins=30, alpha=0.5, label=str(year), density=True)

        ax.set_title(feature.upper())
        ax.legend()

    # Hide unused axes
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_distributions_by_year.png', dpi=150, bbox_inches='tight')
    plt.close()

    return stats_df


def analyze_covariate_shift(data, output_dir):
    """Detect covariate shift between train and test sets."""
    print("\n" + "="*80)
    print("COVARIATE SHIFT ANALYSIS (Train vs Test Features)")
    print("="*80)

    # Use 2018 as proxy for train features, 2020 for test features
    if 2018 not in data or 2020 not in data:
        print("Need 2018 and 2020 data for covariate shift analysis")
        return None

    df_train = harmonize_columns(data[2018].copy())
    df_test = harmonize_columns(data[2020].copy())

    available_numeric = [f.lower() for f in NUMERIC_FEATURES
                        if f.lower() in df_train.columns and f.lower() in df_test.columns]

    shift_results = []

    print(f"\n{'Feature':<12} {'Train Mean':<12} {'Test Mean':<12} {'KS Stat':<10} {'P-value':<12} {'Shift?':<8}")
    print("-" * 70)

    for feature in available_numeric:
        train_vals = pd.to_numeric(df_train[feature], errors='coerce').dropna()
        test_vals = pd.to_numeric(df_test[feature], errors='coerce').dropna()

        # Kolmogorov-Smirnov test for distribution shift
        ks_stat, ks_pval = stats.ks_2samp(train_vals, test_vals)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((train_vals.std()**2 + test_vals.std()**2) / 2)
        cohens_d = (test_vals.mean() - train_vals.mean()) / pooled_std if pooled_std > 0 else 0

        significant_shift = ks_pval < 0.05 and abs(cohens_d) > 0.2

        shift_results.append({
            'feature': feature,
            'train_mean': train_vals.mean(),
            'test_mean': test_vals.mean(),
            'train_std': train_vals.std(),
            'test_std': test_vals.std(),
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval,
            'cohens_d': cohens_d,
            'significant_shift': significant_shift
        })

        shift_marker = "YES" if significant_shift else ""
        print(f"{feature:<12} {train_vals.mean():<12.2f} {test_vals.mean():<12.2f} "
              f"{ks_stat:<10.3f} {ks_pval:<12.4f} {shift_marker:<8}")

    shift_df = pd.DataFrame(shift_results)
    shift_df.to_csv(output_dir / 'covariate_shift_analysis.csv', index=False)

    n_shifted = shift_df['significant_shift'].sum()
    print(f"\nFeatures with significant covariate shift: {n_shifted}/{len(shift_df)}")

    if n_shifted > 0:
        print("\nFeatures with shift:")
        for _, row in shift_df[shift_df['significant_shift']].iterrows():
            print(f"  {row['feature']}: Cohen's d = {row['cohens_d']:.3f}")

    return shift_df


def analyze_feature_target_correlation(data, output_dir):
    """Analyze correlation between features and target."""
    print("\n" + "="*80)
    print("FEATURE-TARGET CORRELATION ANALYSIS")
    print("="*80)

    # Combine all data
    all_dfs = []
    for year, df in data.items():
        df = harmonize_columns(df.copy())
        df = clean_classification(df)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    if 'classification' not in combined.columns:
        print("Classification column not found")
        return None

    # Encode target
    target_encoder = {label: i for i, label in enumerate(sorted(combined['classification'].dropna().unique()))}
    combined['target_encoded'] = combined['classification'].map(target_encoder)

    available_numeric = [f.lower() for f in NUMERIC_FEATURES if f.lower() in combined.columns]

    correlations = []

    print(f"\n{'Feature':<12} {'Pearson r':<12} {'P-value':<12} {'Strength':<15}")
    print("-" * 55)

    for feature in available_numeric:
        values = pd.to_numeric(combined[feature], errors='coerce')

        # Remove NaN
        mask = ~(values.isna() | combined['target_encoded'].isna())
        x = values[mask]
        y = combined.loc[mask, 'target_encoded']

        if len(x) < 10:
            continue

        # Pearson correlation
        r, p = stats.pearsonr(x, y)

        # Determine strength
        if abs(r) < 0.1:
            strength = "Negligible"
        elif abs(r) < 0.3:
            strength = "Weak"
        elif abs(r) < 0.5:
            strength = "Moderate"
        elif abs(r) < 0.7:
            strength = "Strong"
        else:
            strength = "Very Strong"

        correlations.append({
            'feature': feature,
            'pearson_r': r,
            'pvalue': p,
            'strength': strength
        })

        print(f"{feature:<12} {r:<12.3f} {p:<12.4f} {strength:<15}")

    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('pearson_r', key=abs, ascending=False)
    corr_df.to_csv(output_dir / 'feature_target_correlations.csv', index=False)

    print(f"\nTop 5 correlated features:")
    for _, row in corr_df.head(5).iterrows():
        print(f"  {row['feature']}: r = {row['pearson_r']:.3f}")

    return corr_df


def analyze_outliers(data, output_dir):
    """Analyze outliers in the data."""
    print("\n" + "="*80)
    print("OUTLIER ANALYSIS")
    print("="*80)

    # Combine all data
    all_dfs = []
    for year, df in data.items():
        df = harmonize_columns(df.copy())
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    available_numeric = [f.lower() for f in NUMERIC_FEATURES if f.lower() in combined.columns]

    outlier_results = []

    print(f"\n{'Feature':<12} {'Q1':<10} {'Q3':<10} {'IQR':<10} {'Lower':<10} {'Upper':<10} {'N_Outliers':<12} {'%':<8}")
    print("-" * 85)

    for feature in available_numeric:
        values = pd.to_numeric(combined[feature], errors='coerce').dropna()

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        n_outliers = ((values < lower_bound) | (values > upper_bound)).sum()
        pct_outliers = 100 * n_outliers / len(values)

        outlier_results.append({
            'feature': feature,
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'n_outliers': n_outliers,
            'pct_outliers': pct_outliers
        })

        print(f"{feature:<12} {q1:<10.2f} {q3:<10.2f} {iqr:<10.2f} "
              f"{lower_bound:<10.2f} {upper_bound:<10.2f} {n_outliers:<12d} {pct_outliers:<8.1f}")

    outlier_df = pd.DataFrame(outlier_results)
    outlier_df = outlier_df.sort_values('pct_outliers', ascending=False)
    outlier_df.to_csv(output_dir / 'outlier_analysis.csv', index=False)

    high_outlier_features = outlier_df[outlier_df['pct_outliers'] > 5]
    if len(high_outlier_features) > 0:
        print(f"\nFeatures with >5% outliers:")
        for _, row in high_outlier_features.iterrows():
            print(f"  {row['feature']}: {row['pct_outliers']:.1f}% outliers")

    return outlier_df


def analyze_class_separability(data, output_dir):
    """Analyze how well features separate classes."""
    print("\n" + "="*80)
    print("CLASS SEPARABILITY ANALYSIS (ANOVA F-statistic)")
    print("="*80)

    # Combine all data
    all_dfs = []
    for year, df in data.items():
        df = harmonize_columns(df.copy())
        df = clean_classification(df)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    if 'classification' not in combined.columns:
        print("Classification column not found")
        return None

    available_numeric = [f.lower() for f in NUMERIC_FEATURES if f.lower() in combined.columns]

    # Group by classification
    groups = combined.groupby('classification')

    separability_results = []

    print(f"\n{'Feature':<12} {'F-statistic':<15} {'P-value':<12} {'Discriminative?':<15}")
    print("-" * 55)

    for feature in available_numeric:
        feature_groups = []
        for name, group in groups:
            values = pd.to_numeric(group[feature], errors='coerce').dropna()
            if len(values) >= 2:
                feature_groups.append(values.values)

        if len(feature_groups) < 2:
            continue

        # One-way ANOVA
        try:
            f_stat, p_val = stats.f_oneway(*feature_groups)
        except:
            continue

        is_discriminative = p_val < 0.05 and f_stat > 10

        separability_results.append({
            'feature': feature,
            'f_statistic': f_stat,
            'pvalue': p_val,
            'is_discriminative': is_discriminative
        })

        disc_marker = "YES" if is_discriminative else ""
        print(f"{feature:<12} {f_stat:<15.2f} {p_val:<12.6f} {disc_marker:<15}")

    sep_df = pd.DataFrame(separability_results)
    sep_df = sep_df.sort_values('f_statistic', ascending=False)
    sep_df.to_csv(output_dir / 'class_separability.csv', index=False)

    print(f"\nTop 5 most discriminative features:")
    for _, row in sep_df.head(5).iterrows():
        print(f"  {row['feature']}: F = {row['f_statistic']:.2f}")

    return sep_df


def analyze_feature_correlations(data, output_dir):
    """Analyze inter-feature correlations (multicollinearity)."""
    print("\n" + "="*80)
    print("FEATURE CORRELATION MATRIX (Multicollinearity)")
    print("="*80)

    # Combine all data
    all_dfs = []
    for year, df in data.items():
        df = harmonize_columns(df.copy())
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    available_numeric = [f.lower() for f in NUMERIC_FEATURES if f.lower() in combined.columns]

    # Create numeric dataframe
    numeric_df = combined[available_numeric].apply(pd.to_numeric, errors='coerce')

    # Correlation matrix
    corr_matrix = numeric_df.corr()

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    corr_matrix.to_csv(output_dir / 'feature_correlation_matrix.csv')

    # Find highly correlated pairs
    print("\nHighly correlated feature pairs (|r| > 0.8):")
    for i in range(len(available_numeric)):
        for j in range(i+1, len(available_numeric)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.8:
                print(f"  {available_numeric[i]} <-> {available_numeric[j]}: r = {r:.3f}")

    return corr_matrix


def analyze_high_risk_patterns(data, output_dir):
    """Analyze patterns specific to high-risk classes."""
    print("\n" + "="*80)
    print("HIGH-RISK CLASS PATTERN ANALYSIS")
    print("="*80)

    # Combine all data
    all_dfs = []
    for year, df in data.items():
        df = harmonize_columns(df.copy())
        df = clean_classification(df)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    if 'classification' not in combined.columns:
        print("Classification column not found")
        return None

    # Split into high-risk and low-risk
    combined['is_high_risk'] = combined['classification'].isin(HIGH_RISK_CLASSES)

    high_risk = combined[combined['is_high_risk']]
    low_risk = combined[~combined['is_high_risk']]

    print(f"\nTotal samples: {len(combined)}")
    print(f"High-risk samples: {len(high_risk)} ({100*len(high_risk)/len(combined):.1f}%)")
    print(f"Low-risk samples: {len(low_risk)} ({100*len(low_risk)/len(combined):.1f}%)")

    available_numeric = [f.lower() for f in NUMERIC_FEATURES if f.lower() in combined.columns]

    print(f"\n{'Feature':<12} {'LowRisk Mean':<15} {'HighRisk Mean':<15} {'Diff':<12} {'T-stat':<10} {'P-value':<12}")
    print("-" * 80)

    pattern_results = []

    for feature in available_numeric:
        lr_vals = pd.to_numeric(low_risk[feature], errors='coerce').dropna()
        hr_vals = pd.to_numeric(high_risk[feature], errors='coerce').dropna()

        if len(lr_vals) < 5 or len(hr_vals) < 5:
            continue

        # T-test
        t_stat, p_val = stats.ttest_ind(lr_vals, hr_vals)
        diff = hr_vals.mean() - lr_vals.mean()

        pattern_results.append({
            'feature': feature,
            'low_risk_mean': lr_vals.mean(),
            'high_risk_mean': hr_vals.mean(),
            'difference': diff,
            't_statistic': t_stat,
            'pvalue': p_val,
            'significant': p_val < 0.05
        })

        sig_marker = "*" if p_val < 0.05 else ""
        print(f"{feature:<12} {lr_vals.mean():<15.2f} {hr_vals.mean():<15.2f} "
              f"{diff:<+12.2f} {t_stat:<10.2f} {p_val:<12.4f}{sig_marker}")

    pattern_df = pd.DataFrame(pattern_results)
    pattern_df = pattern_df.sort_values('pvalue')
    pattern_df.to_csv(output_dir / 'high_risk_feature_patterns.csv', index=False)

    print(f"\nMost discriminative features for high-risk detection:")
    for _, row in pattern_df[pattern_df['significant']].head(5).iterrows():
        direction = "higher" if row['difference'] > 0 else "lower"
        print(f"  {row['feature']}: High-risk has {direction} values (diff={row['difference']:.2f})")

    return pattern_df


def generate_summary_report(output_dir, results):
    """Generate comprehensive summary report."""
    print("\n" + "="*80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*80)

    report = []
    report.append("# Data Distribution Analysis Report\n")
    report.append("## Key Findings\n")

    # Class imbalance
    report.append("### 1. Class Imbalance\n")
    if 'class_distribution' in results:
        for _, row in results['class_distribution'].iterrows():
            report.append(f"- Year {row['year']}: {row['n_classes']} classes, "
                         f"imbalance ratio {row['imbalance_ratio']:.1f}, "
                         f"high-risk {row['high_risk_pct']:.1f}%\n")

    # Covariate shift
    report.append("\n### 2. Covariate Shift\n")
    if 'covariate_shift' in results and results['covariate_shift'] is not None:
        n_shifted = results['covariate_shift']['significant_shift'].sum()
        total = len(results['covariate_shift'])
        report.append(f"- Features with significant shift: {n_shifted}/{total}\n")

        if n_shifted > 0:
            report.append("- Shifted features:\n")
            for _, row in results['covariate_shift'][results['covariate_shift']['significant_shift']].iterrows():
                report.append(f"  - {row['feature']}: Cohen's d = {row['cohens_d']:.3f}\n")

    # Feature importance
    report.append("\n### 3. Feature Importance for Classification\n")
    if 'class_separability' in results and results['class_separability'] is not None:
        report.append("- Most discriminative features (ANOVA F-stat):\n")
        for _, row in results['class_separability'].head(5).iterrows():
            report.append(f"  - {row['feature']}: F = {row['f_statistic']:.2f}\n")

    # Outliers
    report.append("\n### 4. Outlier Analysis\n")
    if 'outliers' in results and results['outliers'] is not None:
        high_outlier = results['outliers'][results['outliers']['pct_outliers'] > 5]
        if len(high_outlier) > 0:
            report.append("- Features with >5% outliers:\n")
            for _, row in high_outlier.iterrows():
                report.append(f"  - {row['feature']}: {row['pct_outliers']:.1f}%\n")
        else:
            report.append("- No features have >5% outliers\n")

    # Recommendations
    report.append("\n## Preprocessing Recommendations\n")
    report.append("""
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
""")

    # Write report
    report_text = ''.join(report)
    with open(output_dir / 'data_analysis_report.md', 'w') as f:
        f.write(report_text)

    print(report_text)

    return report_text


def main():
    """Run all analyses."""
    print("="*80)
    print("GROUNDWATER QUALITY DATA DISTRIBUTION ANALYSIS")
    print("="*80)

    # Load data
    print("\nLoading data...")
    data = load_data()

    if not data:
        print("ERROR: No data loaded. Please check DATA_DIR path.")
        print(f"Current DATA_DIR: {DATA_DIR}")
        return

    # Results dictionary
    results = {}

    # Run analyses
    results['class_distribution'] = analyze_class_distribution(data, OUTPUT_DIR)
    results['transitions'] = analyze_transition_class_distribution(data, OUTPUT_DIR)
    results['feature_stats'] = analyze_feature_distributions(data, OUTPUT_DIR)
    results['covariate_shift'] = analyze_covariate_shift(data, OUTPUT_DIR)
    results['feature_target_corr'] = analyze_feature_target_correlation(data, OUTPUT_DIR)
    results['outliers'] = analyze_outliers(data, OUTPUT_DIR)
    results['class_separability'] = analyze_class_separability(data, OUTPUT_DIR)
    results['feature_correlations'] = analyze_feature_correlations(data, OUTPUT_DIR)
    results['high_risk_patterns'] = analyze_high_risk_patterns(data, OUTPUT_DIR)

    # Generate summary
    generate_summary_report(OUTPUT_DIR, results)

    print("\n" + "="*80)
    print(f"Analysis complete! Results saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
