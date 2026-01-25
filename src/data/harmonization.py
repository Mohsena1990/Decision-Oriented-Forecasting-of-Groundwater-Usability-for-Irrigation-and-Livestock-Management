"""
Data Harmonization Module
=========================

Standardizes column names and data types across different years of
groundwater quality data to ensure consistency.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# Column name mapping: maps various column names to standardized names
COLUMN_MAPPING = {
    # ID columns
    'sno': 'sno',
    'serial': 'sno',
    's.no': 'sno',

    # Location columns
    'district': 'district',
    'mandal': 'mandal',
    'village': 'village',
    'lat_gis': 'lat_gis',
    'long_gis': 'long_gis',
    'latitude': 'lat_gis',
    'longitude': 'long_gis',

    # Water level
    'gwl': 'gwl',
    'ground_water_level': 'gwl',

    # Season
    'season': 'season',

    # Chemical parameters (handle various naming conventions)
    'ph': 'pH',
    'e.c': 'EC',
    'ec': 'EC',
    'e.c.': 'EC',
    'electrical_conductivity': 'EC',
    'tds': 'TDS',
    'total_dissolved_solids': 'TDS',

    # Anions - handle ionic notation
    'co3': 'CO3',
    'co_-2': 'CO3',
    'co3--': 'CO3',
    'co_-2 ': 'CO3',

    'hco3': 'HCO3',
    'hco_ -': 'HCO3',
    'hco_ - ': 'HCO3',
    'hco3-': 'HCO3',

    'cl': 'Cl',
    'cl -': 'Cl',
    'cl-': 'Cl',
    'chloride': 'Cl',

    'f': 'F',
    'f -': 'F',
    'f-': 'F',
    'fluoride': 'F',

    'no3': 'NO3',
    'no3 ': 'NO3',
    'no3-': 'NO3',
    'no3- ': 'NO3',
    'nitrate': 'NO3',

    'so4': 'SO4',
    'so4-2': 'SO4',
    'so4--': 'SO4',
    'sulphate': 'SO4',
    'sulfate': 'SO4',

    # Cations
    'na': 'Na',
    'na+': 'Na',
    'sodium': 'Na',

    'k': 'K',
    'k+': 'K',
    'potassium': 'K',

    'ca': 'Ca',
    'ca+2': 'Ca',
    'ca++': 'Ca',
    'calcium': 'Ca',

    'mg': 'Mg',
    'mg+2': 'Mg',
    'mg++': 'Mg',
    'magnesium': 'Mg',

    # Calculated parameters
    't.h': 'TH',
    'th': 'TH',
    'total_hardness': 'TH',
    'total hardness': 'TH',

    'sar': 'SAR',
    'sodium_adsorption_ratio': 'SAR',

    # Classification
    'classification': 'Classification',
    'class': 'Classification',

    # RSC classification
    'rsc  meq  / l': 'RSC',
    'rsc meq/l': 'RSC',
    'rsc': 'RSC',

    'classification.1': 'Classification_RSC',
}


@dataclass
class DataHarmonizer:
    """
    Harmonize column names and data types across years.

    Ensures consistent naming conventions and data types for
    downstream processing.
    """

    column_mapping: Dict[str, str] = field(default_factory=lambda: COLUMN_MAPPING.copy())
    harmonized_data: Dict[int, pd.DataFrame] = field(default_factory=dict, init=False)

    def harmonize_all(self, data: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
        """
        Harmonize all years of data.

        Args:
            data: Dictionary mapping year to DataFrame

        Returns:
            Dictionary mapping year to harmonized DataFrame
        """
        logger.info("Harmonizing data across years")

        for year, df in data.items():
            harmonized = self._harmonize_single(df, year)
            self.harmonized_data[year] = harmonized
            logger.info(f"Harmonized {year}: {len(harmonized.columns)} standardized columns")

        self._validate_consistency()
        return self.harmonized_data

    def _harmonize_single(self, df: pd.DataFrame, year: int) -> pd.DataFrame:
        """Harmonize a single year's data."""
        df = df.copy()

        # Step 1: Clean column names (lowercase, strip whitespace)
        clean_names = {}
        for col in df.columns:
            clean = col.lower().strip()
            # Remove extra whitespace
            clean = re.sub(r'\s+', ' ', clean)
            clean_names[col] = clean

        df = df.rename(columns=clean_names)

        # Step 2: Apply column mapping
        mapped_names = {}
        for col in df.columns:
            if col in self.column_mapping:
                mapped_names[col] = self.column_mapping[col]
            else:
                # Keep original if not in mapping (but log warning)
                if col not in ['year']:  # Don't warn about year column
                    logger.debug(f"Year {year}: Unmapped column '{col}'")

        df = df.rename(columns=mapped_names)

        # Step 3: Convert data types
        df = self._convert_types(df)

        # Step 4: Standardize location names (uppercase for consistency)
        for col in ['district', 'mandal', 'village']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.upper().str.strip()

        return df

    def _convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to appropriate data types."""
        # Numeric columns
        numeric_cols = [
            'gwl', 'pH', 'EC', 'TDS', 'CO3', 'HCO3', 'Cl', 'F', 'NO3',
            'SO4', 'Na', 'K', 'Ca', 'Mg', 'TH', 'SAR', 'RSC',
            'lat_gis', 'long_gis'
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Integer columns
        if 'sno' in df.columns:
            df['sno'] = pd.to_numeric(df['sno'], errors='coerce').astype('Int64')
        if 'year' in df.columns:
            df['year'] = df['year'].astype(int)

        return df

    def _validate_consistency(self):
        """Validate that harmonized data is consistent across years."""
        if len(self.harmonized_data) < 2:
            return

        # Get common columns across all years
        all_columns = [set(df.columns) for df in self.harmonized_data.values()]
        common_columns = set.intersection(*all_columns)

        # Get columns that vary across years
        all_present = set.union(*all_columns)
        varying = all_present - common_columns

        if varying:
            logger.warning(f"Columns not present in all years: {varying}")

        logger.info(f"Common columns across all years: {len(common_columns)}")

    def get_common_columns(self) -> List[str]:
        """Get columns present in all years."""
        if not self.harmonized_data:
            return []

        all_columns = [set(df.columns) for df in self.harmonized_data.values()]
        return sorted(set.intersection(*all_columns))

    def get_schema_report(self) -> pd.DataFrame:
        """
        Generate schema report showing columns by year.

        Returns:
            DataFrame with column presence by year
        """
        all_columns = set()
        for df in self.harmonized_data.values():
            all_columns.update(df.columns)

        report = []
        for col in sorted(all_columns):
            row = {'column': col}
            for year, df in self.harmonized_data.items():
                row[f'year_{year}'] = col in df.columns
                if col in df.columns:
                    row[f'dtype_{year}'] = str(df[col].dtype)
            report.append(row)

        return pd.DataFrame(report)
