"""
ERA5-Land Feature Extractor
============================

Reads the downloaded ERA5-Land NetCDF file and extracts seasonal/annual
climate statistics at each groundwater monitoring well location.

Features extracted (6 total)
-----------------------------
era5_precip_annual_mm      : Total annual precipitation [mm]
era5_precip_monsoon_mm     : Jun–Sep total precipitation [mm]
era5_precip_premonsoon_mm  : Feb–May total precipitation [mm]
era5_soil_moisture_annual  : Annual mean volumetric soil water layer-1 [m³/m³]
era5_soil_moisture_monsoon : Jun–Sep mean soil moisture [m³/m³]
era5_soil_moisture_pre     : Feb–May mean soil moisture [m³/m³]

Physical rationale
------------------
* High monsoon precipitation → dilutes dissolved ions (↓EC, TDS, SAR)
* Pre-monsoon soil moisture → indicator of dry-season recharge deficit
* Annual soil moisture → proxy for groundwater recharge capacity

Spatial matching
----------------
For each well (lat, lon), the nearest ERA5 grid cell (0.1° ≈ 9 km) is
selected using minimum Euclidean distance on the lat/lon grid.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Season definitions: month ranges (inclusive)
SEASONS = {
    'monsoon':     (6, 9),     # Jun–Sep: southwest monsoon
    'premonsoon':  (2, 5),     # Feb–May: dry / pre-monsoon heat
    'postmonsoon': (10, 12),   # Oct–Dec: northeast monsoon / rabi
}

ERA5_FEATURES = [
    'era5_precip_annual_mm',
    'era5_precip_monsoon_mm',
    'era5_precip_premonsoon_mm',
    'era5_soil_moisture_annual',
    'era5_soil_moisture_monsoon',
    'era5_soil_moisture_pre',
]


class ERA5Loader:
    """
    Loads ERA5-Land NetCDF and extracts climate features for well locations.

    Parameters
    ----------
    nc_path : str or Path
        Path to ``telangana_era5_land_2018_2020.nc``.
    years : list of int
        Years to extract (must be present in the NetCDF).
    """

    def __init__(self, nc_path: str, years: Optional[List[int]] = None):
        self.nc_path = Path(nc_path)
        self.years = years or [2018, 2019, 2020]
        self._ds = None          # xarray Dataset (loaded lazily)
        self._is_loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> 'ERA5Loader':
        """Open the NetCDF dataset."""
        try:
            import xarray as xr
        except ImportError:
            raise ImportError(
                "xarray not installed. Run: pip install xarray netcdf4 h5netcdf"
            )

        if not self.nc_path.exists():
            raise FileNotFoundError(
                f"ERA5 NetCDF not found: {self.nc_path}\n"
                f"Run: python scripts/download_era5.py to download it."
            )

        logger.info(f"Loading ERA5 dataset from {self.nc_path}")
        self._ds = xr.open_dataset(self.nc_path, engine='netcdf4')
        self._is_loaded = True

        # Log available variables and time range
        time_dim = self._time_dim()
        logger.info(f"  Variables: {list(self._ds.data_vars)}")
        logger.info(
            f"  Time range: {str(self._ds[time_dim].values[0])[:10]} → "
            f"{str(self._ds[time_dim].values[-1])[:10]}"
        )
        lat_arr = self._ds['latitude'].values if 'latitude' in self._ds else self._ds['lat'].values
        lon_arr = self._ds['longitude'].values if 'longitude' in self._ds else self._ds['lon'].values
        logger.info(
            f"  Grid: lat=[{lat_arr.min():.2f},{lat_arr.max():.2f}] "
            f"lon=[{lon_arr.min():.2f},{lon_arr.max():.2f}]"
        )
        return self

    def extract_features(
        self,
        well_df: pd.DataFrame,
        lat_col: str = 'lat_gis',
        lon_col: str = 'long_gis',
    ) -> pd.DataFrame:
        """
        Extract ERA5 climate features at each well location.

        Parameters
        ----------
        well_df : DataFrame
            Must contain ``lat_col``, ``lon_col``, and a ``year`` column.
        lat_col, lon_col : str
            Column names for latitude and longitude.

        Returns
        -------
        DataFrame with ERA5_FEATURES columns aligned to well_df rows.
        Missing grid cells are filled with the Telangana-wide annual mean.
        """
        if not self._is_loaded:
            self.load()

        # Resolve coordinate names (ERA5 CDS uses 'latitude'/'longitude')
        lat_dim = 'latitude' if 'latitude' in self._ds.dims else 'lat'
        lon_dim = 'longitude' if 'longitude' in self._ds.dims else 'lon'
        lat_grid = self._ds[lat_dim].values
        lon_grid = self._ds[lon_dim].values

        # Resolve variable names
        tp_var = self._find_var(['tp', 'total_precipitation'])
        sm_var = self._find_var(['swvl1', 'volumetric_soil_water_layer_1'])

        logger.info(f"ERA5 variables: precip='{tp_var}', soil_moisture='{sm_var}'")

        # Pre-compute annual/seasonal aggregates per grid cell (fast vectorised)
        annual_stats = self._compute_grid_stats(tp_var, sm_var, lat_dim, lon_dim)

        # Map each well to nearest grid cell
        lat_vals = pd.to_numeric(well_df[lat_col], errors='coerce').values
        lon_vals = pd.to_numeric(well_df[lon_col], errors='coerce').values
        years = well_df['year'].values if 'year' in well_df.columns else np.zeros(len(well_df), dtype=int)

        results = []
        for i, (lat_w, lon_w, year) in enumerate(zip(lat_vals, lon_vals, years)):
            if np.isnan(lat_w) or np.isnan(lon_w) or year not in annual_stats:
                results.append(self._empty_row())
                continue

            # Nearest grid cell (Euclidean on lat/lon degrees, sufficient at 0.1° scale)
            i_lat = int(np.argmin(np.abs(lat_grid - lat_w)))
            i_lon = int(np.argmin(np.abs(lon_grid - lon_w)))

            stats = annual_stats[int(year)]
            row = {}
            for key in ERA5_FEATURES:
                arr = stats.get(key)
                if arr is not None:
                    val = float(arr[i_lat, i_lon])
                    row[key] = val if np.isfinite(val) else float(np.nanmean(arr))
                else:
                    row[key] = np.nan
            results.append(row)

        out = pd.DataFrame(results, index=well_df.index)
        logger.info(
            f"ERA5 features extracted for {len(out)} wells: "
            f"{out.notna().all(axis=1).sum()} fully populated"
        )
        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _time_dim(self) -> str:
        """Return the name of the time coordinate (CDS exports use 'valid_time')."""
        for name in ('time', 'valid_time'):
            if name in self._ds.coords or name in self._ds.dims:
                return name
        raise KeyError(
            f"No time coordinate found. Available coords: {list(self._ds.coords)}"
        )

    def _find_var(self, candidates: List[str]) -> str:
        """Return the first candidate that exists in the dataset."""
        for name in candidates:
            if name in self._ds:
                return name
        raise KeyError(
            f"None of {candidates} found in ERA5 dataset. "
            f"Available variables: {list(self._ds.data_vars)}"
        )

    def _compute_grid_stats(
        self,
        tp_var: str,
        sm_var: str,
        lat_dim: str,
        lon_dim: str,
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Pre-compute annual and seasonal statistics on the full ERA5 grid.

        ERA5 tp is hourly accumulation in metres → convert to mm and sum.
        ERA5 swvl1 is instantaneous m³/m³ → take mean.

        Returns dict: year → {feature_name: 2-D array (lat, lon)}
        """
        import xarray as xr

        tp = self._ds[tp_var]    # (time, lat, lon)
        sm = self._ds[sm_var]    # (time, lat, lon)

        # Build a time coordinate DataFrame for fast filtering
        times = pd.DatetimeIndex(self._ds[self._time_dim()].values)
        time_df = pd.DataFrame({'year': times.year, 'month': times.month},
                               index=np.arange(len(times)))

        result = {}
        for year in self.years:
            yr_mask = (time_df['year'] == year).values

            tp_yr = tp.values[yr_mask]   # (T, lat, lon)
            sm_yr = sm.values[yr_mask]

            stats = {}

            # Annual totals / means
            stats['era5_precip_annual_mm'] = np.nansum(tp_yr, axis=0) * 1000   # m→mm
            stats['era5_soil_moisture_annual'] = np.nanmean(sm_yr, axis=0)

            # Seasonal
            mon_yr = time_df.loc[time_df['year'] == year, 'month'].values

            for season, (m_start, m_end) in SEASONS.items():
                s_mask = (mon_yr >= m_start) & (mon_yr <= m_end)

                if season == 'monsoon':
                    stats['era5_precip_monsoon_mm'] = np.nansum(tp_yr[s_mask], axis=0) * 1000
                    stats['era5_soil_moisture_monsoon'] = np.nanmean(sm_yr[s_mask], axis=0)
                elif season == 'premonsoon':
                    stats['era5_precip_premonsoon_mm'] = np.nansum(tp_yr[s_mask], axis=0) * 1000
                    stats['era5_soil_moisture_pre'] = np.nanmean(sm_yr[s_mask], axis=0)

            result[year] = stats
            logger.debug(f"ERA5 grid stats computed for {year}")

        return result

    def _empty_row(self) -> Dict[str, float]:
        return {f: np.nan for f in ERA5_FEATURES}

    def close(self):
        if self._ds is not None:
            self._ds.close()
            self._ds = None
            self._is_loaded = False


def load_era5_features(
    nc_path: str,
    well_df: pd.DataFrame,
    lat_col: str = 'lat_gis',
    lon_col: str = 'long_gis',
    years: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper: load ERA5 and extract features for a well DataFrame.

    Parameters
    ----------
    nc_path : str
        Path to ERA5 NetCDF file.
    well_df : DataFrame
        Groundwater well data; must have ``lat_col``, ``lon_col``, ``year``.

    Returns
    -------
    DataFrame with ERA5_FEATURES columns, same index as ``well_df``.
    Returns empty DataFrame (all NaN) if ``nc_path`` does not exist.
    """
    if not Path(nc_path).exists():
        logger.warning(
            f"ERA5 file not found at '{nc_path}'. "
            "ERA5 climate features will be skipped.\n"
            "Download with: python scripts/download_era5.py"
        )
        return pd.DataFrame(
            np.nan,
            index=well_df.index,
            columns=ERA5_FEATURES
        )

    loader = ERA5Loader(nc_path, years=years)
    try:
        loader.load()
        return loader.extract_features(well_df, lat_col=lat_col, lon_col=lon_col)
    finally:
        loader.close()
