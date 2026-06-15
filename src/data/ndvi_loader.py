"""
Sentinel-2 NDVI Feature Loader
================================

Reads the GEE-exported NDVI CSV and matches it to groundwater monitoring
wells using nearest-neighbour spatial join (haversine distance).

Expected input file
-------------------
``telangana_ndvi_2018_2020.csv`` produced by ``scripts/download_ndvi_gee.py``

Columns required: lat, lon, year, ndvi_oct_nov_mean, ndvi_oct_nov_max

Features provided (2 total)
-----------------------------
ndvi_oct_nov_mean : Mean NDVI in Oct-Nov (post-monsoon crop cover)
ndvi_oct_nov_max  : Max NDVI in Oct-Nov (peak vegetation)

Physical rationale
------------------
* High Oct-Nov NDVI → dense Rabi crop cover → intensive fertilizer use
  → elevated NO3, K, SO4 leach into groundwater
* Low NDVI → fallow land / sparse cover → different leaching patterns
* Temporal change in NDVI across years detects land-use intensification

Spatial matching
----------------
Each well (lat, lon) is matched to the nearest NDVI sample point using
haversine distance.  Wells > 5 km from any sample point use the mandal/
district mean NDVI as fallback.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NDVI_FEATURES = ['ndvi_oct_nov_mean', 'ndvi_oct_nov_max']

# Threshold: if nearest NDVI sample is farther than this, use group fallback
MAX_MATCH_DISTANCE_KM = 5.0


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray,
                  lat2: float, lon2: float) -> np.ndarray:
    """Haversine distance in km between arrays of points and a single point."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a.clip(0, 1)))


class NDVILoader:
    """
    Loads NDVI CSV and extracts features for groundwater well DataFrames.

    Parameters
    ----------
    csv_path : str or Path
        Path to ``telangana_ndvi_2018_2020.csv``.
    years : list of int, optional
        Years to include (default: [2018, 2019, 2020]).
    """

    def __init__(self, csv_path: str, years: Optional[List[int]] = None):
        self.csv_path = Path(csv_path)
        self.years = years or [2018, 2019, 2020]
        self._ndvi: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> 'NDVILoader':
        """Read and validate the NDVI CSV."""
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"NDVI CSV not found: {self.csv_path}\n"
                f"Run: python scripts/download_ndvi_gee.py to export it."
            )

        df = pd.read_csv(self.csv_path)

        required = {'lat', 'lon', 'year'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"NDVI CSV missing columns: {missing}")

        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
        df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')

        for feat in NDVI_FEATURES:
            if feat not in df.columns:
                logger.warning(f"NDVI CSV missing column '{feat}' — will be NaN")
                df[feat] = np.nan
            else:
                df[feat] = pd.to_numeric(df[feat], errors='coerce')

        df = df.dropna(subset=['lat', 'lon', 'year'])
        self._ndvi = df
        logger.info(
            f"Loaded {len(df)} NDVI records "
            f"({df['year'].nunique()} years, {len(df[['lat','lon']].drop_duplicates())} locations)"
        )
        return self

    def extract_features(
        self,
        well_df: pd.DataFrame,
        lat_col: str = 'lat_gis',
        lon_col: str = 'long_gis',
    ) -> pd.DataFrame:
        """
        Match NDVI records to wells via nearest-neighbour join.

        For each well (lat, lon, year), finds the closest NDVI sample point
        for the same year using haversine distance.  If the nearest point is
        farther than ``MAX_MATCH_DISTANCE_KM``, a district/mandal-level
        median is used as fallback.

        Returns
        -------
        DataFrame with NDVI_FEATURES columns, same index as ``well_df``.
        """
        if self._ndvi is None:
            self.load()

        ndvi = self._ndvi
        lat_vals = pd.to_numeric(well_df[lat_col], errors='coerce').values
        lon_vals = pd.to_numeric(well_df[lon_col], errors='coerce').values
        year_vals = well_df['year'].values if 'year' in well_df.columns else np.zeros(len(well_df))

        rows = []
        for lat_w, lon_w, year in zip(lat_vals, lon_vals, year_vals):
            ndvi_yr = ndvi[ndvi['year'] == int(year)]
            if len(ndvi_yr) == 0 or np.isnan(lat_w) or np.isnan(lon_w):
                rows.append(self._empty_row())
                continue

            dists = _haversine_km(
                ndvi_yr['lat'].values, ndvi_yr['lon'].values, lat_w, lon_w
            )
            best_idx = int(np.argmin(dists))
            best_dist = dists[best_idx]

            if best_dist <= MAX_MATCH_DISTANCE_KM:
                best_row = ndvi_yr.iloc[best_idx]
                rows.append({
                    'ndvi_oct_nov_mean': float(best_row.get('ndvi_oct_nov_mean', np.nan)),
                    'ndvi_oct_nov_max': float(best_row.get('ndvi_oct_nov_max', np.nan)),
                })
            else:
                # Fallback: year-level median
                rows.append({
                    'ndvi_oct_nov_mean': float(ndvi_yr['ndvi_oct_nov_mean'].median()),
                    'ndvi_oct_nov_max': float(ndvi_yr['ndvi_oct_nov_max'].median()),
                })

        out = pd.DataFrame(rows, index=well_df.index)
        n_valid = out.notna().all(axis=1).sum()
        logger.info(
            f"NDVI features matched for {n_valid}/{len(out)} wells "
            f"(threshold={MAX_MATCH_DISTANCE_KM} km)"
        )
        return out

    def _empty_row(self) -> dict:
        return {f: np.nan for f in NDVI_FEATURES}


def load_ndvi_features(
    csv_path: str,
    well_df: pd.DataFrame,
    lat_col: str = 'lat_gis',
    lon_col: str = 'long_gis',
    years: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper: load NDVI CSV and extract features for a well DataFrame.

    Returns empty DataFrame (all NaN) if ``csv_path`` does not exist.
    """
    if not Path(csv_path).exists():
        logger.warning(
            f"NDVI CSV not found at '{csv_path}'. "
            "NDVI features will be skipped.\n"
            "Export with: python scripts/download_ndvi_gee.py"
        )
        return pd.DataFrame(
            np.nan,
            index=well_df.index,
            columns=NDVI_FEATURES,
        )

    loader = NDVILoader(csv_path, years=years)
    loader.load()
    return loader.extract_features(well_df, lat_col=lat_col, lon_col=lon_col)
