"""
Data Ingestion Module
=====================

Handles loading groundwater quality data from CSV files with proper
year identification and initial column standardization.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


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

        # Remove completely empty columns
        empty_cols = df.columns[df.isna().all()]
        if len(empty_cols) > 0:
            logger.debug(f"Year {year}: Removing {len(empty_cols)} empty columns: {list(empty_cols)}")
            df = df.drop(columns=empty_cols)

        # Remove unnamed columns (artifacts from CSV parsing)
        unnamed_cols = [c for c in df.columns if c.startswith('Unnamed')]
        if unnamed_cols:
            logger.debug(f"Year {year}: Removing {len(unnamed_cols)} unnamed columns")
            df = df.drop(columns=unnamed_cols)

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
