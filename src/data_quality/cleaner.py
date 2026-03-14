"""
Data Cleaner Module
===================

Implements configurable cleaning actions based on validation results,
including label resolution, tier classification, missing value handling,
and outlier treatment.

Label handling uses a 4-tier semantic risk system (USDA salinity-sodium
hazard chart) instead of a frequency-based "Other" catch-all:

    T1_Safe        — unrestricted use
    T2_Marginal    — use with caution
    T3_Restricted  — restricted use
    T4_Unsafe      — unsuitable
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.objectives.definitions import RISK_TIER_MAPPING, map_raw_label_to_tier

logger = logging.getLogger(__name__)


@dataclass
class CleaningAction:
    """Record of a cleaning action taken."""
    action_type: str
    column: Optional[str]
    year: Optional[int]
    details: Dict[str, Any]
    n_affected: int


@dataclass
class DataCleaner:
    """
    Configurable data cleaner with explicit logging of all actions.

    All cleaning decisions are logged for reproducibility and audit trails.

    Label classification uses a 4-tier semantic risk system based on the
    USDA salinity-sodium hazard chart.  Every C#S# label maps to exactly
    one tier — there is no frequency-based "Other" catch-all.
    """

    # Label cleaning
    label_typo_mapping: Dict[str, str] = field(default_factory=dict)

    # Missing value handling
    numeric_impute_strategy: str = "median"
    categorical_impute_strategy: str = "most_frequent"
    high_missing_drop_threshold: float = 0.80

    # Outlier handling
    outlier_method: str = "winsorize"  # "winsorize", "clip", "none"
    winsorize_limits: Tuple[float, float] = (0.01, 0.99)

    # Tracking
    cleaning_log: List[CleaningAction] = field(default_factory=list, init=False)
    fitted_imputers: Dict[str, SimpleImputer] = field(default_factory=dict, init=False)
    fitted_winsorize_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict, init=False)
    label_encoder: Optional[Dict[str, int]] = field(default=None, init=False)

    def clean_all(
        self,
        data: Dict[int, pd.DataFrame],
        fit_year: Optional[int] = None
    ) -> Dict[int, pd.DataFrame]:
        """
        Apply all cleaning steps to data.

        Args:
            data: Dictionary mapping year to DataFrame
            fit_year: Year to fit imputers on (others are transformed)

        Returns:
            Cleaned data dictionary
        """
        logger.info("Starting data cleaning pipeline")
        self.cleaning_log = []

        cleaned = {}
        for year, df in data.items():
            cleaned[year] = df.copy()

        # Step 1: Drop empty/unnamed columns
        cleaned = self._drop_empty_columns(cleaned)

        # Step 2: Resolve label typos
        cleaned = self._resolve_label_typos(cleaned)

        # Step 3: Map C#S# labels to 4-tier semantic risk system
        cleaned = self._apply_tier_classification(cleaned)

        # Step 4: Create unified label encoder
        self._create_label_encoder(cleaned)

        # Step 5: Fit imputers on training year
        if fit_year is not None:
            self._fit_imputers(cleaned[fit_year])
            self._fit_winsorize_bounds(cleaned[fit_year])

        # Step 6: Apply imputation
        cleaned = self._apply_imputation(cleaned)

        # Step 7: Apply outlier handling
        if self.outlier_method != "none":
            cleaned = self._apply_outlier_handling(cleaned)

        logger.info(f"Cleaning complete. {len(self.cleaning_log)} actions taken.")
        return cleaned

    def _drop_empty_columns(self, data: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
        """Drop completely empty and unnamed columns."""
        for year, df in data.items():
            # Empty columns
            empty_cols = [c for c in df.columns if df[c].isna().all()]
            if empty_cols:
                df = df.drop(columns=empty_cols)
                self.cleaning_log.append(CleaningAction(
                    action_type='drop_empty_columns',
                    column=None, year=year,
                    details={'columns': empty_cols},
                    n_affected=len(empty_cols)
                ))

            # Unnamed columns
            unnamed_cols = [c for c in df.columns if 'unnamed' in c.lower()]
            if unnamed_cols:
                df = df.drop(columns=unnamed_cols)
                self.cleaning_log.append(CleaningAction(
                    action_type='drop_unnamed_columns',
                    column=None, year=year,
                    details={'columns': unnamed_cols},
                    n_affected=len(unnamed_cols)
                ))

            data[year] = df

        return data

    def _resolve_label_typos(
        self,
        data: Dict[int, pd.DataFrame],
        target_col: str = 'Classification'
    ) -> Dict[int, pd.DataFrame]:
        """Resolve label typos using mapping."""
        # Default typo mapping
        default_mapping = {
            'O.G': 'OG',
            'O.G.': 'OG',
            ' C2S1': 'C2S1',
            'C2S1 ': 'C2S1',
        }
        mapping = {**default_mapping, **self.label_typo_mapping}

        for year, df in data.items():
            if target_col not in df.columns:
                continue

            original = df[target_col].copy()

            # Strip whitespace
            df[target_col] = df[target_col].astype(str).str.strip()

            # Apply mapping
            for old, new in mapping.items():
                mask = df[target_col] == old
                if mask.any():
                    df[target_col] = df[target_col].replace(old, new)
                    self.cleaning_log.append(CleaningAction(
                        action_type='resolve_label_typo',
                        column=target_col, year=year,
                        details={'from': old, 'to': new},
                        n_affected=int(mask.sum())
                    ))

            # Log overall changes
            n_changed = (original != df[target_col]).sum()
            if n_changed > 0:
                logger.info(f"Year {year}: Resolved {n_changed} label typos")

            data[year] = df

        return data

    def _apply_tier_classification(
        self,
        data: Dict[int, pd.DataFrame],
        target_col: str = 'Classification'
    ) -> Dict[int, pd.DataFrame]:
        """
        Map every C#S# label to one of four semantic risk tiers.

        Tier mapping (USDA salinity-sodium hazard chart):
            T1_Safe        — C1S1, C1S2, C1S3, C2S1, OG
            T2_Marginal    — C1S4, C2S2, C2S3, C3S1, C3S2
            T3_Restricted  — C2S4, C3S3, C3S4, C4S1, C4S2
            T4_Unsafe      — C4S3, C4S4

        Unrecognised labels default to T2_Marginal (conservative).
        This replaces the old frequency-based "Other" catch-all which
        incorrectly mixed safe and dangerous samples in one bucket.
        """
        known_labels = set(RISK_TIER_MAPPING.keys())

        for year, df in data.items():
            if target_col not in df.columns:
                continue

            original = df[target_col].copy()
            df[target_col] = df[target_col].apply(map_raw_label_to_tier)

            n_changed = (original != df[target_col]).sum()
            unrecognised = set(original.unique()) - known_labels - {'nan', 'NaN', 'None'}
            if unrecognised:
                logger.warning(
                    f"Year {year}: unrecognised labels defaulted to T2_Marginal: {unrecognised}"
                )

            self.cleaning_log.append(CleaningAction(
                action_type='apply_tier_classification',
                column=target_col, year=year,
                details={
                    'tier_mapping': 'USDA 4-tier salinity-sodium hazard',
                    'n_mapped': int(n_changed),
                    'unrecognised_labels': list(unrecognised),
                },
                n_affected=int(n_changed)
            ))
            logger.info(
                f"Year {year}: mapped {n_changed} labels to risk tiers. "
                f"Distribution: {df[target_col].value_counts().to_dict()}"
            )

            data[year] = df

        return data

    def _create_label_encoder(
        self,
        data: Dict[int, pd.DataFrame],
        target_col: str = 'Classification'
    ):
        """Create unified label encoder across all years.

        Tiers are sorted in ordinal order T1 < T2 < T3 < T4 so that
        the integer indices preserve the risk ordering.
        """
        all_classes = set()
        for df in data.values():
            if target_col in df.columns:
                all_classes.update(df[target_col].dropna().unique())

        # Sort by tier number (T1_Safe → 0, T2_Marginal → 1, …)
        _tier_order = {'T1_Safe': 0, 'T2_Marginal': 1, 'T3_Restricted': 2, 'T4_Unsafe': 3}

        def sort_key(label):
            return _tier_order.get(str(label), 99)

        sorted_classes = sorted(all_classes, key=sort_key)
        self.label_encoder = {cls: idx for idx, cls in enumerate(sorted_classes)}

        logger.info(f"Label encoder created with {len(self.label_encoder)} tiers: {self.label_encoder}")
        logger.debug(f"Label mapping: {self.label_encoder}")

    def _fit_imputers(self, df: pd.DataFrame):
        """Fit imputers on training data."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Exclude target and ID columns
        exclude = ['Classification', 'Classification_RSC', 'sno', 'year', 'season']
        numeric_cols = [c for c in numeric_cols if c not in exclude]
        categorical_cols = [c for c in categorical_cols if c not in exclude]

        # Fit numeric imputer
        if numeric_cols:
            imputer = SimpleImputer(strategy=self.numeric_impute_strategy)
            imputer.fit(df[numeric_cols])
            self.fitted_imputers['numeric'] = {
                'imputer': imputer,
                'columns': numeric_cols
            }
            logger.info(f"Fitted numeric imputer on {len(numeric_cols)} columns")

        # Fit categorical imputer
        if categorical_cols:
            imputer = SimpleImputer(strategy=self.categorical_impute_strategy)
            imputer.fit(df[categorical_cols].astype(str))
            self.fitted_imputers['categorical'] = {
                'imputer': imputer,
                'columns': categorical_cols
            }
            logger.info(f"Fitted categorical imputer on {len(categorical_cols)} columns")

    def _fit_winsorize_bounds(self, df: pd.DataFrame):
        """Fit winsorization bounds on training data."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ['lat_gis', 'long_gis', 'sno', 'year']
        numeric_cols = [c for c in numeric_cols if c not in exclude]

        for col in numeric_cols:
            values = df[col].dropna()
            if len(values) > 0:
                lower = values.quantile(self.winsorize_limits[0])
                upper = values.quantile(self.winsorize_limits[1])
                self.fitted_winsorize_bounds[col] = (lower, upper)

        logger.info(f"Fitted winsorization bounds for {len(self.fitted_winsorize_bounds)} columns")

    def _apply_imputation(self, data: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
        """Apply fitted imputers to all data."""
        for year, df in data.items():
            # Numeric imputation
            if 'numeric' in self.fitted_imputers:
                info = self.fitted_imputers['numeric']
                cols = [c for c in info['columns'] if c in df.columns]
                if cols:
                    n_missing_before = df[cols].isna().sum().sum()
                    df[cols] = info['imputer'].transform(df[cols])
                    n_missing_after = df[cols].isna().sum().sum()
                    n_imputed = n_missing_before - n_missing_after

                    if n_imputed > 0:
                        self.cleaning_log.append(CleaningAction(
                            action_type='impute_numeric',
                            column=None, year=year,
                            details={'strategy': self.numeric_impute_strategy, 'columns': cols},
                            n_affected=n_imputed
                        ))

            # Categorical imputation
            if 'categorical' in self.fitted_imputers:
                info = self.fitted_imputers['categorical']
                cols = [c for c in info['columns'] if c in df.columns]
                if cols:
                    n_missing_before = df[cols].isna().sum().sum()
                    df[cols] = info['imputer'].transform(df[cols].astype(str))
                    n_missing_after = df[cols].isna().sum().sum()
                    n_imputed = n_missing_before - n_missing_after

                    if n_imputed > 0:
                        self.cleaning_log.append(CleaningAction(
                            action_type='impute_categorical',
                            column=None, year=year,
                            details={'strategy': self.categorical_impute_strategy, 'columns': cols},
                            n_affected=n_imputed
                        ))

            data[year] = df

        return data

    def _apply_outlier_handling(self, data: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
        """Apply outlier handling to all data."""
        if self.outlier_method != "winsorize":
            return data

        for year, df in data.items():
            for col, (lower, upper) in self.fitted_winsorize_bounds.items():
                if col not in df.columns:
                    continue

                original = df[col].copy()
                df[col] = df[col].clip(lower=lower, upper=upper)
                n_clipped = (original != df[col]).sum()

                if n_clipped > 0:
                    self.cleaning_log.append(CleaningAction(
                        action_type='winsorize',
                        column=col, year=year,
                        details={'lower': lower, 'upper': upper},
                        n_affected=n_clipped
                    ))

            data[year] = df

        return data

    def get_cleaning_report(self) -> pd.DataFrame:
        """Generate report of all cleaning actions."""
        records = []
        for action in self.cleaning_log:
            records.append({
                'action_type': action.action_type,
                'column': action.column,
                'year': action.year,
                'n_affected': action.n_affected,
                'details': str(action.details)
            })
        return pd.DataFrame(records)

    def encode_labels(
        self,
        df: pd.DataFrame,
        target_col: str = 'Classification'
    ) -> pd.Series:
        """Encode labels using the unified encoder."""
        if self.label_encoder is None:
            raise ValueError("Label encoder not fitted. Call clean_all() first.")

        return df[target_col].map(self.label_encoder)

    def get_label_encoder(self) -> Dict[str, int]:
        """Get the fitted label encoder."""
        return self.label_encoder.copy() if self.label_encoder else {}
