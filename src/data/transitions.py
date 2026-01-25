"""
Transition Builder Module
=========================

Constructs temporal transition pairs (t -> t+1) for forecasting
by matching locations across years.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TransitionBuilder:
    """
    Build transition pairs for temporal forecasting.

    Creates matched location pairs between consecutive years for
    predicting next-year groundwater quality class.

    Attributes:
        location_keys: Columns used for location matching
        tolerance: Coordinate tolerance for fuzzy matching (degrees)
    """

    location_keys: List[str] = field(default_factory=lambda: [
        'district', 'mandal', 'village', 'lat_gis', 'long_gis'
    ])
    tolerance: float = 0.001  # ~100m for lat/long fuzzy matching

    def build_transitions(
        self,
        data: Dict[int, pd.DataFrame],
        from_year: int,
        to_year: int,
        target_col: str = 'Classification'
    ) -> pd.DataFrame:
        """
        Build transition dataset from year t to year t+1.

        Args:
            data: Dictionary mapping year to DataFrame
            from_year: Source year (t)
            to_year: Target year (t+1)
            target_col: Target classification column

        Returns:
            DataFrame with features from year t and target from year t+1
        """
        if from_year not in data or to_year not in data:
            raise ValueError(f"Both years {from_year} and {to_year} must be in data")

        df_from = data[from_year].copy()
        df_to = data[to_year].copy()

        logger.info(f"Building transitions: {from_year} -> {to_year}")
        logger.info(f"Source records: {len(df_from)}, Target records: {len(df_to)}")

        # Create location ID for matching
        df_from['_loc_id'] = self._create_location_id(df_from)
        df_to['_loc_id'] = self._create_location_id(df_to)

        # Find exact matches
        matched = self._match_locations(df_from, df_to)

        logger.info(f"Matched locations: {len(matched)}")

        if len(matched) == 0:
            logger.warning("No matching locations found!")
            return pd.DataFrame()

        # Build transition dataset
        transitions = self._build_transition_df(
            df_from, df_to, matched, target_col, from_year, to_year
        )

        return transitions

    def _create_location_id(self, df: pd.DataFrame) -> pd.Series:
        """Create unique location identifier from key columns."""
        # Use categorical columns directly
        cat_cols = ['district', 'mandal', 'village']
        cat_parts = []
        for col in cat_cols:
            if col in df.columns:
                cat_parts.append(df[col].astype(str).str.upper().str.strip())

        # Use rounded coordinates for fuzzy matching
        coord_parts = []
        for col in ['lat_gis', 'long_gis']:
            if col in df.columns:
                rounded = np.round(df[col] / self.tolerance) * self.tolerance
                coord_parts.append(rounded.astype(str))

        all_parts = cat_parts + coord_parts
        if not all_parts:
            raise ValueError("No location key columns found")

        return pd.Series(['|'.join(parts) for parts in zip(*all_parts)])

    def _match_locations(
        self,
        df_from: pd.DataFrame,
        df_to: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Find matching locations between two years.

        Returns DataFrame with from_idx, to_idx columns.
        """
        # Get unique location IDs
        from_locs = df_from[['_loc_id']].drop_duplicates()
        from_locs['_from_idx'] = df_from.index[from_locs.index]

        to_locs = df_to[['_loc_id']].drop_duplicates()
        to_locs['_to_idx'] = df_to.index[to_locs.index]

        # Inner join on location ID
        matched = pd.merge(
            from_locs, to_locs,
            on='_loc_id',
            how='inner'
        )

        return matched

    def _build_transition_df(
        self,
        df_from: pd.DataFrame,
        df_to: pd.DataFrame,
        matched: pd.DataFrame,
        target_col: str,
        from_year: int,
        to_year: int
    ) -> pd.DataFrame:
        """Build final transition DataFrame."""
        # Get matched indices
        from_indices = []
        to_indices = []

        for _, row in matched.iterrows():
            loc_id = row['_loc_id']
            # Find all rows with this location in each year
            from_rows = df_from[df_from['_loc_id'] == loc_id].index.tolist()
            to_rows = df_to[df_to['_loc_id'] == loc_id].index.tolist()

            # Take first match (assuming unique locations)
            if from_rows and to_rows:
                from_indices.append(from_rows[0])
                to_indices.append(to_rows[0])

        # Extract features from source year
        feature_cols = [c for c in df_from.columns
                       if c not in ['_loc_id', 'sno', 'year', 'season']]

        transitions = df_from.loc[from_indices, feature_cols].reset_index(drop=True)

        # Add target from next year
        transitions[f'{target_col}_target'] = df_to.loc[to_indices, target_col].values

        # Add metadata
        transitions['from_year'] = from_year
        transitions['to_year'] = to_year

        # Also preserve the current year's classification for analysis
        if target_col in transitions.columns:
            transitions[f'{target_col}_current'] = transitions[target_col]

        return transitions

    def build_all_transitions(
        self,
        data: Dict[int, pd.DataFrame],
        target_col: str = 'Classification'
    ) -> Dict[str, pd.DataFrame]:
        """
        Build all consecutive year transitions.

        Args:
            data: Dictionary mapping year to DataFrame
            target_col: Target classification column

        Returns:
            Dictionary mapping transition name (e.g., "2018_2019") to DataFrame
        """
        years = sorted(data.keys())
        transitions = {}

        for i in range(len(years) - 1):
            from_year = years[i]
            to_year = years[i + 1]
            name = f"{from_year}_{to_year}"

            trans_df = self.build_transitions(
                data, from_year, to_year, target_col
            )
            transitions[name] = trans_df

            logger.info(f"Transition {name}: {len(trans_df)} samples")

        return transitions

    def get_location_coverage(
        self,
        data: Dict[int, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Analyze location coverage across years.

        Returns:
            DataFrame showing which locations appear in which years
        """
        location_years = {}

        for year, df in data.items():
            df = df.copy()
            df['_loc_id'] = self._create_location_id(df)

            for loc_id in df['_loc_id'].unique():
                if loc_id not in location_years:
                    location_years[loc_id] = set()
                location_years[loc_id].add(year)

        # Build coverage report
        years = sorted(data.keys())
        coverage = []

        for loc_id, present_years in location_years.items():
            row = {'location_id': loc_id}
            for year in years:
                row[f'year_{year}'] = year in present_years
            row['n_years'] = len(present_years)
            row['all_years'] = len(present_years) == len(years)
            coverage.append(row)

        df_coverage = pd.DataFrame(coverage)

        # Summary statistics
        logger.info(f"Total unique locations: {len(df_coverage)}")
        logger.info(f"Locations in all years: {df_coverage['all_years'].sum()}")

        return df_coverage
