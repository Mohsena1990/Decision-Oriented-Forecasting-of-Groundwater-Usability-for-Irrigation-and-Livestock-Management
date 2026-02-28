"""
Data Ingestion Module
=====================

Handles loading groundwater quality data from CSV files with proper
year identification and initial column standardization.

Raw preprocessing applied per the original data cleaning script:
  - Year-specific column renaming (chemical notation standardisation)
  - Dropping redundant columns (sno, season)
  - Year column assignment
  - Specific outlier / typo fixes identified in the raw CSVs
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Year-specific raw column renames
# (raw CSV name → intermediate name consumed by DataHarmonizer)
# ---------------------------------------------------------------------------
_RAW_COLUMN_RENAMES: Dict[int, Dict[str, str]] = {
    2019: {
        'EC': 'E.C',       # 2019 CSV uses plain "EC"; rename so harmoniser
                           # can map "e.c" → canonical "EC" consistently
        'CO_-2 ':  'CO3',
        'HCO_ - ': 'HCO3',
        'Cl -':    'Cl',
        'F -':     'F',
        'NO3- ':   'NO3',
        'SO4-2':   'SO4',
        'Na+':     'Na',
        'K+':      'K',
        'Ca+2':    'Ca',
        'Mg+2':    'Mg',
    },
}

# ---------------------------------------------------------------------------
# Columns to drop after loading (present in raw CSVs, not needed downstream)
# ---------------------------------------------------------------------------
_COLS_TO_DROP: Dict[int, List[str]] = {
    2018: ['sno', 'season'],
    2019: ['sno', 'season'],
    2020: ['sno', 'season'],   # 'Unnamed: 8' is already removed by the
                                # generic unnamed-column handler
}

# ---------------------------------------------------------------------------
# Specific cell-level outlier / typo corrections for year 2020
# These were found by manual inspection of the raw CSV.
# ---------------------------------------------------------------------------
_YEAR_2020_FIXES = [
    # (column, row_index, old_value_fragment, correct_value)
    ('pH',            261, '8..05',  '8.05'),
    ('Classification', 178, 'O.G',  'OG'),
    ('Classification', 208, 'O.G',  'OG'),
]


@dataclass
class DataIngestion:
    """
    Load and manage groundwater quality data from multiple years.

    Attributes:
        base_dir: Base directory containing data files
        files: Mapping of year to filename
        data: Dictionary storing loaded DataFrames by year
    """

    base_dir: Union[str, Path]
    files: Dict[int, str]
    data: Dict[int, pd.DataFrame] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.base_dir = Path(self.base_dir)
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Base directory not found: {self.base_dir}")

    def load_all(self) -> Dict[int, pd.DataFrame]:
        """
        Load all data files specified in configuration.

        Returns:
            Dictionary mapping year to DataFrame
        """
        logger.info(f"Loading data from {self.base_dir}")

        for year, filename in self.files.items():
            filepath = self.base_dir / filename
            if filepath.exists():
                df = self._load_single_file(filepath, year)
                self.data[year] = df
                logger.info(f"Loaded {year}: {len(df)} records, {len(df.columns)} columns")
            else:
                logger.warning(f"File not found: {filepath}")

        return self.data

    def _load_single_file(self, filepath: Path, year: int) -> pd.DataFrame:
        """
        Load a single CSV file with proper handling.

        Args:
            filepath: Path to CSV file
            year: Year identifier

        Returns:
            DataFrame with year column added
        """
        # Read CSV with flexible parsing
        df = pd.read_csv(
            filepath,
            encoding='utf-8',
            low_memory=False,
            na_values=['', ' ', 'NA', 'N/A', 'nan', 'NaN', '-', '--']
        )

        # Add year column
        df['year'] = year

        # Initial column name cleaning (spaces, special chars)
        df.columns = df.columns.str.strip()

        # Apply year-specific column renames BEFORE harmonisation
        if year in _RAW_COLUMN_RENAMES:
            df = df.rename(columns=_RAW_COLUMN_RENAMES[year])
            logger.debug(f"Year {year}: applied raw column renames")

        # Remove completely empty columns
        empty_cols = df.columns[df.isna().all()]
        if len(empty_cols) > 0:
            logger.debug(f"Year {year}: Removing {len(empty_cols)} empty columns: {list(empty_cols)}")
            df = df.drop(columns=empty_cols)

        # Remove unnamed columns (artifacts from CSV parsing, e.g. 'Unnamed: 8')
        unnamed_cols = [c for c in df.columns if c.startswith('Unnamed')]
        if unnamed_cols:
            logger.debug(f"Year {year}: Removing {len(unnamed_cols)} unnamed columns")
            df = df.drop(columns=unnamed_cols)

        # Drop redundant administrative columns (sno, season)
        if year in _COLS_TO_DROP:
            drop_present = [c for c in _COLS_TO_DROP[year] if c in df.columns]
            if drop_present:
                df = df.drop(columns=drop_present)
                logger.debug(f"Year {year}: Dropped columns {drop_present}")

        # Apply cell-level outlier / typo fixes for year 2020
        if year == 2020:
            df = self._fix_year2020_outliers(df)

        return df

    def _fix_year2020_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply known cell-level corrections for the 2020 raw CSV.

        Fixes identified by manual inspection:
          - pH at row 261:  '8..05' → '8.05'   (double-dot typo)
          - Classification at rows 178, 208: 'O.G' → 'OG'
        """
        for col, idx, bad_val, good_val in _YEAR_2020_FIXES:
            if col not in df.columns:
                continue
            if idx >= len(df):
                logger.debug(f"Year 2020 fix: row {idx} out of range for column '{col}'")
                continue

            cell = df[col].iloc[idx]
            if pd.isna(cell):
                continue

            cell_str = str(cell)
            if bad_val in cell_str:
                fixed = cell_str.replace(bad_val, good_val)
                df.iloc[idx, df.columns.get_loc(col)] = fixed
                logger.debug(f"Year 2020: fixed '{col}'[{idx}]: '{cell_str}' → '{fixed}'")

        # Ensure pH is numeric after potential string fixes
        if 'pH' in df.columns:
            df['pH'] = pd.to_numeric(df['pH'], errors='coerce')

        return df

    def get_year(self, year: int) -> Optional[pd.DataFrame]:
        """Get DataFrame for a specific year."""
        return self.data.get(year)

    def get_all_years(self) -> List[int]:
        """Get list of loaded years."""
        return sorted(self.data.keys())

    def get_combined(self) -> pd.DataFrame:
        """
        Get combined DataFrame with all years.

        Returns:
            Combined DataFrame with year column
        """
        if not self.data:
            raise ValueError("No data loaded. Call load_all() first.")

        return pd.concat(self.data.values(), ignore_index=True)

    def summary(self) -> pd.DataFrame:
        """
        Generate summary statistics for loaded data.

        Returns:
            DataFrame with summary per year
        """
        summaries = []
        for year, df in self.data.items():
            summary = {
                'year': year,
                'n_records': len(df),
                'n_columns': len(df.columns),
                'n_districts': df['district'].nunique() if 'district' in df else 0,
                'missing_rate': df.isna().mean().mean()
            }
            summaries.append(summary)

        return pd.DataFrame(summaries)
