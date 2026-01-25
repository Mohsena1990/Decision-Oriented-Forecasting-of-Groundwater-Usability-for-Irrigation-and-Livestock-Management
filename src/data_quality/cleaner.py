"""
Data Cleaner Module
===================

Implements configurable cleaning actions based on validation results,
including label resolution, missing value handling, and outlier treatment.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

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
    """

    # Label cleaning
    label_typo_mapping: Dict[str, str] = field(default_factory=dict)
    rare_class_handling: str = "merge"  # "merge" or "drop"
    rare_class_threshold: int = 5
    merge_target: str = "Other"

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

        # Step 3: Handle rare classes
        cleaned = self._handle_rare_classes(cleaned)

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

    def _handle_rare_classes(
        self,
        data: Dict[int, pd.DataFrame],
        target_col: str = 'Classification'
    ) -> Dict[int, pd.DataFrame]:
        """Handle rare classes by merging or dropping."""
        # Count classes across all years
        all_counts = pd.Series(dtype=int)
        for df in data.values():
            if target_col in df.columns:
                counts = df[target_col].value_counts()
                all_counts = all_counts.add(counts, fill_value=0)

        rare_classes = all_counts[all_counts < self.rare_class_threshold].index.tolist()

        if not rare_classes:
            return data

        logger.info(f"Found {len(rare_classes)} rare classes: {rare_classes}")

        for year, df in data.items():
            if target_col not in df.columns:
                continue

            if self.rare_class_handling == "merge":
                mask = df[target_col].isin(rare_classes)
                n_affected = mask.sum()
                if n_affected > 0:
                    df.loc[mask, target_col] = self.merge_target
                    self.cleaning_log.append(CleaningAction(
                        action_type='merge_rare_classes',
                        column=target_col, year=year,
                        details={'merged_classes': rare_classes, 'to': self.merge_target},
                        n_affected=n_affected
                    ))

            elif self.rare_class_handling == "drop":
                mask = df[target_col].isin(rare_classes)
                n_affected = mask.sum()
                if n_affected > 0:
                    df = df[~mask]
                    self.cleaning_log.append(CleaningAction(
                        action_type='drop_rare_classes',
                        column=target_col, year=year,
                        details={'dropped_classes': rare_classes},
                        n_affected=n_affected
                    ))

            data[year] = df

        return data

    def _create_label_encoder(
        self,
        data: Dict[int, pd.DataFrame],
        target_col: str = 'Classification'
    ):
        """Create unified label encoder across all years."""
        all_classes = set()
        for df in data.values():
            if target_col in df.columns:
                all_classes.update(df[target_col].dropna().unique())

        # Sort classes by C and S components for ordinal ordering
        def sort_key(label):
            match = re.match(r'C(\d)S(\d)', str(label))
            if match:
                return (int(match.group(1)), int(match.group(2)))
            return (99, 99)  # Put non-standard labels at end

        sorted_classes = sorted(all_classes, key=sort_key)
        self.label_encoder = {cls: idx for idx, cls in enumerate(sorted_classes)}

        logger.info(f"Label encoder created with {len(self.label_encoder)} classes")
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
