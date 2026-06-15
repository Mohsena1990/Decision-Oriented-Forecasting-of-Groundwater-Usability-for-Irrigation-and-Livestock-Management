"""
External Feature Joiner
========================

Attaches ERA5-Land climate features to the groundwater quality DataFrames
on a per-year, per-well basis.

Note: Sentinel-2 NDVI integration via Google Earth Engine is currently
disabled (GEE authentication not available). NDVI features are set to NaN
and excluded from enrichment. Re-enable by running scripts/download_ndvi_gee.py
once GEE access is configured.

Design principles
-----------------
* Leakage-safe: all external data is year-matched (ERA5 features for year Y
  are joined to groundwater records from year Y, not Y+1).
* Graceful: if the ERA5 dataset is absent, the pipeline continues with
  NaN-filled columns (which the preprocessing pipeline handles via imputation).
* No information leakage between wells: each well is matched independently
  using its GPS coordinates.

All external features (ERA5-Land, 6 features)
----------------------------------------------
    era5_precip_annual_mm
    era5_precip_monsoon_mm
    era5_precip_premonsoon_mm
    era5_soil_moisture_annual
    era5_soil_moisture_monsoon
    era5_soil_moisture_pre
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .era5_loader import ERA5_FEATURES, load_era5_features
# from .ndvi_loader import NDVI_FEATURES, load_ndvi_features  # GEE disabled

logger = logging.getLogger(__name__)

NDVI_FEATURES: List[str] = []  # GEE disabled — no NDVI features
ALL_EXTERNAL_FEATURES = ERA5_FEATURES  # ERA5 only (NDVI disabled)


class ExternalFeatureJoiner:
    """
    Joins ERA5 and NDVI features to annual groundwater well DataFrames.

    Parameters
    ----------
    era5_nc_path : str
        Path to ``telangana_era5_land_2018_2020.nc``.
    ndvi_csv_path : str
        Path to ``telangana_ndvi_2018_2020.csv``.
    lat_col, lon_col : str
        Column names for latitude and longitude in the well DataFrames.
    years : list of int
        Years to process (default: [2018, 2019, 2020]).
    """

    def __init__(
        self,
        era5_nc_path: str,
        ndvi_csv_path: str = '',  # unused (GEE disabled)
        lat_col: str = 'lat_gis',
        lon_col: str = 'long_gis',
        years: Optional[List[int]] = None,
    ):
        self.era5_nc_path = era5_nc_path
        self.ndvi_csv_path = ndvi_csv_path  # kept for API compatibility
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.years = years or [2018, 2019, 2020]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich_all(
        self,
        data_dict: Dict[int, pd.DataFrame],
    ) -> Dict[int, pd.DataFrame]:
        """
        Add ERA5 + NDVI features to every year's DataFrame.

        Parameters
        ----------
        data_dict : dict
            ``{year: DataFrame}`` from the ingestion/harmonisation stage.
            Each DataFrame must contain ``lat_col`` and ``lon_col`` columns.

        Returns
        -------
        Same dict with 8 additional columns per DataFrame.
        Features are NaN where external data is unavailable.
        """
        enriched = {}
        era5_available = Path(self.era5_nc_path).exists()

        if not era5_available:
            logger.warning(
                f"ERA5 NetCDF not found: {self.era5_nc_path}\n"
                "  → ERA5 features will be NaN. "
                "Run: python scripts/download_era5.py"
            )

        for year, df in data_dict.items():
            enriched[year] = self._enrich_year(df, year, era5_available)
            logger.info(
                f"Year {year}: enriched {len(df)} wells — "
                f"ERA5: {'active' if era5_available else 'skipped (file missing)'}"
            )

        return enriched

    def get_external_feature_names(self) -> List[str]:
        """Return the names of all external features added."""
        return ALL_EXTERNAL_FEATURES.copy()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _enrich_year(
        self,
        df: pd.DataFrame,
        year: int,
        era5_available: bool,
    ) -> pd.DataFrame:
        """Attach ERA5 features to a single year's DataFrame."""
        df_out = df.copy()

        well_with_year = df_out.copy()
        well_with_year['year'] = year

        # --- ERA5 features ---
        if era5_available:
            era5_feats = load_era5_features(
                self.era5_nc_path,
                well_with_year,
                lat_col=self.lat_col,
                lon_col=self.lon_col,
                years=self.years,
            )
        else:
            era5_feats = pd.DataFrame(
                np.nan, index=df_out.index, columns=ERA5_FEATURES
            )

        # --- NDVI features — disabled (GEE not configured) ---
        # ndvi_feats = load_ndvi_features(...)

        df_out = pd.concat([df_out, era5_feats], axis=1)
        return df_out


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def enrich_with_external_data(
    data_dict: Dict[int, pd.DataFrame],
    era5_nc_path: str,
    ndvi_csv_path: str,
    lat_col: str = 'lat_gis',
    lon_col: str = 'long_gis',
    years: Optional[List[int]] = None,
) -> Dict[int, pd.DataFrame]:
    """
    One-call wrapper: enrich all years with ERA5 + NDVI features.

    Safely skips any external source that is unavailable (NaN-filled columns).

    Returns
    -------
    dict: {year: enriched DataFrame}
    """
    joiner = ExternalFeatureJoiner(
        era5_nc_path=era5_nc_path,
        ndvi_csv_path=ndvi_csv_path,
        lat_col=lat_col,
        lon_col=lon_col,
        years=years,
    )
    return joiner.enrich_all(data_dict)
