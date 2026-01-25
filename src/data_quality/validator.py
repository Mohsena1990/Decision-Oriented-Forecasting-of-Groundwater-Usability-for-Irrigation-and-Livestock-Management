"""
Data Quality Validator Module
=============================

Comprehensive data quality validation including schema checks, duplicates,
missingness, outliers, range checks, and class distribution analysis.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container for validation check results."""
    check_name: str
    passed: bool
    details: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class DataQualityValidator:
    """
    Comprehensive data quality validator.

    Performs schema checks, duplicate detection, missingness analysis,
    outlier detection, range validation, and class distribution analysis.
    """

    expected_columns: List[str] = field(default_factory=list)
    high_missing_threshold: float = 0.30
    drop_missing_threshold: float = 0.80
    outlier_method: str = "iqr"
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0
    range_checks: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    location_keys: List[str] = field(default_factory=lambda: [
        'district', 'mandal', 'village', 'lat_gis', 'long_gis'
    ])

    def validate_all(
        self,
        data: Dict[int, pd.DataFrame]
    ) -> Dict[str, ValidationResult]:
        """
        Run all validation checks on data.

        Args:
            data: Dictionary mapping year to DataFrame

        Returns:
            Dictionary of validation results
        """
        results = {}

        # Schema checks
        results['schema'] = self._check_schema(data)

        # Duplicate checks
        results['duplicates'] = self._check_duplicates(data)

        # Missingness analysis
        results['missingness'] = self._check_missingness(data)

        # Outlier detection
        results['outliers'] = self._check_outliers(data)

        # Range validation
        results['range_checks'] = self._check_ranges(data)

        # Class distribution
        results['class_distribution'] = self._check_class_distribution(data)

        # Class imbalance
        results['imbalance'] = self._check_imbalance(data)

        # Location coverage
        results['location_coverage'] = self._check_location_coverage(data)

        # Log summary
        self._log_summary(results)

        return results

    def _check_schema(self, data: Dict[int, pd.DataFrame]) -> ValidationResult:
        """Check schema consistency and expected columns."""
        details = {
            'years': {},
            'common_columns': [],
            'extra_columns': {},
            'missing_columns': {}
        }
        warnings = []
        errors = []

        all_columns_per_year = {}
        for year, df in data.items():
            cols = set(df.columns)
            all_columns_per_year[year] = cols

            # Check for empty/unnamed columns
            empty_cols = [c for c in df.columns if df[c].isna().all()]
            unnamed_cols = [c for c in df.columns if 'unnamed' in c.lower()]

            details['years'][year] = {
                'n_columns': len(df.columns),
                'columns': list(df.columns),
                'empty_columns': empty_cols,
                'unnamed_columns': unnamed_cols
            }

            if empty_cols:
                warnings.append(f"Year {year}: Empty columns found: {empty_cols}")
            if unnamed_cols:
                warnings.append(f"Year {year}: Unnamed columns found: {unnamed_cols}")

        # Find common columns
        if all_columns_per_year:
            common = set.intersection(*all_columns_per_year.values())
            details['common_columns'] = sorted(common)

        # Check expected columns
        if self.expected_columns:
            for year, cols in all_columns_per_year.items():
                expected_set = set(self.expected_columns)
                missing = expected_set - cols
                extra = cols - expected_set - {'year'}

                if missing:
                    details['missing_columns'][year] = list(missing)
                    warnings.append(f"Year {year}: Missing expected columns: {missing}")

                if extra:
                    details['extra_columns'][year] = list(extra)

        passed = len(errors) == 0
        return ValidationResult('schema', passed, details, warnings, errors)

    def _check_duplicates(self, data: Dict[int, pd.DataFrame]) -> ValidationResult:
        """Check for duplicate records by location keys."""
        details = {'years': {}}
        warnings = []
        errors = []

        total_duplicates = 0

        for year, df in data.items():
            available_keys = [k for k in self.location_keys if k in df.columns]

            if not available_keys:
                warnings.append(f"Year {year}: No location keys available for duplicate check")
                continue

            # Check duplicates
            dup_mask = df.duplicated(subset=available_keys, keep=False)
            n_duplicates = dup_mask.sum()
            total_duplicates += n_duplicates

            details['years'][year] = {
                'n_records': len(df),
                'n_duplicates': n_duplicates,
                'duplicate_rate': n_duplicates / len(df) if len(df) > 0 else 0,
                'keys_used': available_keys
            }

            if n_duplicates > 0:
                # Get duplicate examples
                dup_examples = df[dup_mask][available_keys].head(5).to_dict('records')
                details['years'][year]['examples'] = dup_examples
                warnings.append(f"Year {year}: {n_duplicates} duplicate records found")

        details['total_duplicates'] = total_duplicates
        passed = total_duplicates == 0

        return ValidationResult('duplicates', passed, details, warnings, errors)

    def _check_missingness(self, data: Dict[int, pd.DataFrame]) -> ValidationResult:
        """Analyze missing values per column and year."""
        details = {'years': {}, 'high_missing_columns': []}
        warnings = []
        errors = []

        for year, df in data.items():
            missing_rates = df.isna().mean().to_dict()
            total_missing = df.isna().sum().sum()
            total_cells = df.size

            details['years'][year] = {
                'missing_rates': missing_rates,
                'total_missing': total_missing,
                'total_cells': total_cells,
                'overall_missing_rate': total_missing / total_cells if total_cells > 0 else 0
            }

            # Flag high missing columns
            high_missing = [col for col, rate in missing_rates.items()
                          if rate > self.high_missing_threshold]

            if high_missing:
                warnings.append(
                    f"Year {year}: High missing columns (>{self.high_missing_threshold:.0%}): {high_missing}"
                )
                details['high_missing_columns'].extend(
                    [(year, col, missing_rates[col]) for col in high_missing]
                )

            # Flag columns for potential dropping
            drop_candidates = [col for col, rate in missing_rates.items()
                             if rate > self.drop_missing_threshold]
            if drop_candidates:
                warnings.append(
                    f"Year {year}: Columns with >{self.drop_missing_threshold:.0%} missing: {drop_candidates}"
                )

        passed = len(errors) == 0
        return ValidationResult('missingness', passed, details, warnings, errors)

    def _check_outliers(self, data: Dict[int, pd.DataFrame]) -> ValidationResult:
        """Detect outliers in numeric columns."""
        details = {'years': {}, 'method': self.outlier_method}
        warnings = []
        errors = []

        numeric_cols = ['gwl', 'pH', 'EC', 'TDS', 'CO3', 'HCO3', 'Cl', 'F',
                       'NO3', 'SO4', 'Na', 'K', 'Ca', 'Mg', 'TH', 'SAR', 'RSC']

        for year, df in data.items():
            year_details = {}

            for col in numeric_cols:
                if col not in df.columns:
                    continue

                values = df[col].dropna()
                if len(values) < 10:  # Need minimum samples
                    continue

                if self.outlier_method == "iqr":
                    outliers = self._detect_iqr_outliers(values)
                else:
                    outliers = self._detect_zscore_outliers(values)

                n_outliers = outliers.sum()
                if n_outliers > 0:
                    outlier_rate = n_outliers / len(values)
                    year_details[col] = {
                        'n_outliers': n_outliers,
                        'outlier_rate': outlier_rate,
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'median': float(values.median())
                    }

                    if outlier_rate > 0.1:
                        warnings.append(
                            f"Year {year}, {col}: High outlier rate ({outlier_rate:.1%})"
                        )

            details['years'][year] = year_details

        passed = len(errors) == 0
        return ValidationResult('outliers', passed, details, warnings, errors)

    def _detect_iqr_outliers(self, values: pd.Series) -> pd.Series:
        """Detect outliers using IQR method."""
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.iqr_multiplier * iqr
        upper = q3 + self.iqr_multiplier * iqr
        return (values < lower) | (values > upper)

    def _detect_zscore_outliers(self, values: pd.Series) -> pd.Series:
        """Detect outliers using robust Z-score method."""
        median = values.median()
        mad = np.median(np.abs(values - median))
        if mad == 0:
            return pd.Series(False, index=values.index)
        modified_zscore = 0.6745 * (values - median) / mad
        return np.abs(modified_zscore) > self.zscore_threshold

    def _check_ranges(self, data: Dict[int, pd.DataFrame]) -> ValidationResult:
        """Check if values fall within expected ranges."""
        details = {'years': {}, 'violations': []}
        warnings = []
        errors = []

        for year, df in data.items():
            year_violations = {}

            for col, (min_val, max_val) in self.range_checks.items():
                if col not in df.columns:
                    continue

                values = df[col].dropna()
                below = (values < min_val).sum()
                above = (values > max_val).sum()

                if below > 0 or above > 0:
                    year_violations[col] = {
                        'below_min': below,
                        'above_max': above,
                        'range': (min_val, max_val),
                        'actual_range': (float(values.min()), float(values.max()))
                    }
                    details['violations'].append((year, col, below, above))
                    warnings.append(
                        f"Year {year}, {col}: {below} below min, {above} above max"
                    )

            details['years'][year] = year_violations

        passed = len(details['violations']) == 0
        return ValidationResult('range_checks', passed, details, warnings, errors)

    def _check_class_distribution(
        self,
        data: Dict[int, pd.DataFrame],
        target_col: str = 'Classification'
    ) -> ValidationResult:
        """Analyze class distribution per year."""
        details = {'years': {}, 'all_classes': set()}
        warnings = []
        errors = []

        for year, df in data.items():
            if target_col not in df.columns:
                errors.append(f"Year {year}: Target column '{target_col}' not found")
                continue

            class_counts = df[target_col].value_counts().to_dict()
            class_probs = df[target_col].value_counts(normalize=True).to_dict()

            details['years'][year] = {
                'counts': class_counts,
                'proportions': class_probs,
                'n_classes': len(class_counts),
                'total_samples': len(df)
            }

            details['all_classes'].update(class_counts.keys())

            # Check for rare classes
            rare_classes = [c for c, count in class_counts.items() if count < 5]
            if rare_classes:
                warnings.append(f"Year {year}: Rare classes (<5 samples): {rare_classes}")

        details['all_classes'] = sorted(details['all_classes'])

        passed = len(errors) == 0
        return ValidationResult('class_distribution', passed, details, warnings, errors)

    def _check_imbalance(
        self,
        data: Dict[int, pd.DataFrame],
        target_col: str = 'Classification'
    ) -> ValidationResult:
        """Analyze class imbalance."""
        details = {'years': {}}
        warnings = []
        errors = []

        for year, df in data.items():
            if target_col not in df.columns:
                continue

            class_counts = df[target_col].value_counts()
            majority = class_counts.max()
            minority = class_counts.min()

            imbalance_ratio = majority / minority if minority > 0 else float('inf')

            details['years'][year] = {
                'majority_class': class_counts.idxmax(),
                'majority_count': int(majority),
                'minority_class': class_counts.idxmin(),
                'minority_count': int(minority),
                'imbalance_ratio': imbalance_ratio,
                'n_minority_classes': (class_counts < class_counts.median()).sum()
            }

            if imbalance_ratio > 10:
                warnings.append(
                    f"Year {year}: High class imbalance (ratio: {imbalance_ratio:.1f})"
                )

        passed = len(errors) == 0
        return ValidationResult('imbalance', passed, details, warnings, errors)

    def _check_location_coverage(self, data: Dict[int, pd.DataFrame]) -> ValidationResult:
        """Check location coverage across years."""
        details = {}
        warnings = []
        errors = []

        # Create location identifiers
        location_sets = {}
        for year, df in data.items():
            available_keys = [k for k in self.location_keys if k in df.columns]
            if available_keys:
                location_ids = df[available_keys].apply(
                    lambda row: '|'.join(str(v).upper().strip() for v in row),
                    axis=1
                )
                location_sets[year] = set(location_ids.unique())
            else:
                location_sets[year] = set()

        years = sorted(location_sets.keys())
        all_locations = set.union(*location_sets.values())
        common_locations = set.intersection(*location_sets.values())

        details['total_unique_locations'] = len(all_locations)
        details['common_across_all_years'] = len(common_locations)
        details['locations_per_year'] = {y: len(locs) for y, locs in location_sets.items()}

        # Check transition feasibility
        for i in range(len(years) - 1):
            y1, y2 = years[i], years[i + 1]
            overlap = len(location_sets[y1] & location_sets[y2])
            details[f'overlap_{y1}_{y2}'] = overlap

            if overlap < 50:
                warnings.append(
                    f"Low location overlap between {y1} and {y2}: {overlap} locations"
                )

        passed = len(common_locations) > 0
        if not passed:
            errors.append("No common locations across all years!")

        return ValidationResult('location_coverage', passed, details, warnings, errors)

    def _log_summary(self, results: Dict[str, ValidationResult]):
        """Log summary of validation results."""
        logger.info("=" * 60)
        logger.info("DATA QUALITY VALIDATION SUMMARY")
        logger.info("=" * 60)

        for name, result in results.items():
            status = "PASS" if result.passed else "FAIL"
            n_warnings = len(result.warnings)
            n_errors = len(result.errors)
            logger.info(f"  {name}: {status} ({n_warnings} warnings, {n_errors} errors)")

        total_warnings = sum(len(r.warnings) for r in results.values())
        total_errors = sum(len(r.errors) for r in results.values())
        logger.info("=" * 60)
        logger.info(f"Total: {total_warnings} warnings, {total_errors} errors")
