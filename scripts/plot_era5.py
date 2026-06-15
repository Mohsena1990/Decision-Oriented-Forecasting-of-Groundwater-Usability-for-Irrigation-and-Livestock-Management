"""
ERA5-Land Visualization for Telangana Groundwater Study
========================================================

Generates 8 publication-quality figures from the downloaded ERA5-Land NetCDF:

  F_era5_1  — Spatial map: mean annual precipitation 2018-2020
  F_era5_2  — Spatial map: mean annual soil moisture 2018-2020
  F_era5_3  — Monthly time series (domain-averaged) 2018-2020
  F_era5_4  — Seasonal boxplots: monsoon vs pre-monsoon by year
  F_era5_5  — Heatmap: month × year precipitation
  F_era5_6  — Heatmap: month × year soil moisture
  F_era5_7  — Year-to-year comparison bar chart
  F_era5_8  — Precipitation vs soil moisture scatter (coloured by month)

Usage
-----
    python scripts/plot_era5.py \
        --era5 /home/mohsen/data/era5/telangana_era5_land_2018_2020.nc \
        --out  outputs/paper_outputs/figures
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Telangana bounding box [West, East, South, North]
BBOX = dict(lon_min=77.2, lon_max=81.3, lat_min=15.8, lat_max=19.9)

YEAR_COLORS = {2018: '#1f77b4', 2019: '#ff7f0e', 2020: '#2ca02c'}
MONSOON_MONTHS = [6, 7, 8, 9]
PREMONSOON_MONTHS = [2, 3, 4, 5]

STYLE = 'seaborn-v0_8-whitegrid'


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_era5(nc_path: str) -> 'xr.Dataset':
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    # Rename time dim if needed (new CDS API uses 'valid_time')
    if 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
    # Precipitation: convert m → mm
    tp_var = [v for v in ds.data_vars if 'tp' in v.lower() or 'precip' in v.lower()][0]
    sm_var = [v for v in ds.data_vars if 'swvl' in v.lower() or 'soil' in v.lower()][0]
    ds = ds.rename({tp_var: 'tp', sm_var: 'swvl1'})
    # tp is hourly accumulation in m; multiply by 1000 to get mm
    ds['tp'] = ds['tp'] * 1000.0
    ds['tp'].attrs['units'] = 'mm'
    ds['swvl1'].attrs['units'] = 'm³/m³'
    return ds


def ds_to_monthly(ds: 'xr.Dataset') -> pd.DataFrame:
    """Domain-averaged monthly means."""
    tp_mean = ds['tp'].mean(dim=['latitude', 'longitude']).to_series()
    sm_mean = ds['swvl1'].mean(dim=['latitude', 'longitude']).to_series()
    df = pd.DataFrame({'tp_mm': tp_mean, 'swvl1': sm_mean})
    df.index = pd.to_datetime(df.index)
    df = df.resample('ME').agg({'tp_mm': 'sum', 'swvl1': 'mean'})
    df['year'] = df.index.year
    df['month'] = df.index.month
    return df


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def savefig(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{name}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    logger.info(f'Saved → {path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_spatial_precip(ds, out_dir: Path):
    """F_era5_1: Spatial map of mean annual precipitation."""
    annual_tp = ds['tp'].resample(time='YE').sum().mean(dim='time')
    lats = annual_tp.latitude.values
    lons = annual_tp.longitude.values
    data = annual_tp.values

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.pcolormesh(lons, lats, data, cmap='YlGnBu', shading='auto')
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        cb.set_label('Mean Annual Precipitation (mm)', fontsize=11)
        ax.set_title('ERA5-Land: Mean Annual Precipitation 2018–2020\nTelangana, India', fontsize=13, fontweight='bold')
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')
        ax.set_xlim(BBOX['lon_min'], BBOX['lon_max'])
        ax.set_ylim(BBOX['lat_min'], BBOX['lat_max'])
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f°E'))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f°N'))
        ax.set_aspect('equal')
        _add_grid_labels(ax)
    savefig(fig, out_dir, 'F_era5_1_spatial_precip')


def plot_spatial_soilmoisture(ds, out_dir: Path):
    """F_era5_2: Spatial map of mean soil moisture."""
    mean_sm = ds['swvl1'].mean(dim='time')
    lats = mean_sm.latitude.values
    lons = mean_sm.longitude.values
    data = mean_sm.values

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.pcolormesh(lons, lats, data, cmap='BrBG', shading='auto')
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        cb.set_label('Mean Soil Moisture (m³/m³)', fontsize=11)
        ax.set_title('ERA5-Land: Mean Volumetric Soil Moisture 2018–2020\nTelangana, India', fontsize=13, fontweight='bold')
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')
        ax.set_xlim(BBOX['lon_min'], BBOX['lon_max'])
        ax.set_ylim(BBOX['lat_min'], BBOX['lat_max'])
        ax.set_aspect('equal')
        _add_grid_labels(ax)
    savefig(fig, out_dir, 'F_era5_2_spatial_soilmoisture')


def plot_monthly_timeseries(monthly: pd.DataFrame, out_dir: Path):
    """F_era5_3: Monthly time series for all 3 years."""
    with plt.style.context(STYLE):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

        for year, grp in monthly.groupby('year'):
            ax1.plot(grp.index, grp['tp_mm'], marker='o', linewidth=2,
                     color=YEAR_COLORS[year], label=str(year))
            ax2.plot(grp.index, grp['swvl1'], marker='s', linewidth=2,
                     color=YEAR_COLORS[year], label=str(year))

        ax1.set_ylabel('Monthly Precipitation (mm)', fontsize=11)
        ax1.set_title('ERA5-Land Monthly Domain-Averaged Climate — Telangana 2018–2020', fontsize=13, fontweight='bold')
        ax1.legend(title='Year', framealpha=0.8)
        _shade_monsoon(ax1, monthly.index)

        ax2.set_ylabel('Soil Moisture (m³/m³)', fontsize=11)
        ax2.set_xlabel('Month', fontsize=11)
        ax2.legend(title='Year', framealpha=0.8)
        _shade_monsoon(ax2, monthly.index)

        fig.text(0.5, 0.01, 'Shaded region = monsoon months (Jun–Sep)', ha='center',
                 fontsize=9, color='steelblue', style='italic')
        fig.tight_layout(rect=[0, 0.03, 1, 1])
    savefig(fig, out_dir, 'F_era5_3_monthly_timeseries')


def plot_seasonal_boxplots(monthly: pd.DataFrame, out_dir: Path):
    """F_era5_4: Monsoon vs pre-monsoon boxplots per year."""
    def season_label(m):
        if m in MONSOON_MONTHS:
            return 'Monsoon\n(Jun–Sep)'
        if m in PREMONSOON_MONTHS:
            return 'Pre-monsoon\n(Feb–May)'
        return 'Other'

    monthly = monthly.copy()
    monthly['season'] = monthly['month'].map(season_label)
    df_seas = monthly[monthly['season'] != 'Other']

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        season_order = ['Pre-monsoon\n(Feb–May)', 'Monsoon\n(Jun–Sep)']
        years = sorted(df_seas['year'].unique())

        for ax, (var, label, cmap) in zip(axes, [
            ('tp_mm', 'Precipitation (mm/month)', 'Blues'),
            ('swvl1', 'Soil Moisture (m³/m³)', 'Greens'),
        ]):
            width = 0.25
            x = np.arange(len(season_order))
            for i, year in enumerate(years):
                vals = []
                positions = []
                for j, season in enumerate(season_order):
                    subset = df_seas[(df_seas['year'] == year) & (df_seas['season'] == season)][var].values
                    vals.append(subset)
                    positions.append(x[j] + (i - 1) * width)

                bp = ax.boxplot(
                    vals, positions=positions, widths=width * 0.85,
                    patch_artist=True, notch=False,
                    boxprops=dict(facecolor=YEAR_COLORS[year], alpha=0.7),
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(color=YEAR_COLORS[year]),
                    capprops=dict(color=YEAR_COLORS[year]),
                    flierprops=dict(marker='o', color=YEAR_COLORS[year], alpha=0.5, markersize=4),
                )
                bp['boxes'][0].set_label(str(year))
                if len(vals) > 1:
                    bp['boxes'][1].set_label('_nolegend_')

            ax.set_xticks(x)
            ax.set_xticklabels(season_order, fontsize=10)
            ax.set_ylabel(label, fontsize=11)
            ax.set_title(label.split('(')[0].strip(), fontsize=12, fontweight='bold')
            ax.legend(title='Year', fontsize=9)

        fig.suptitle('ERA5-Land Seasonal Distribution by Year — Telangana', fontsize=14, fontweight='bold')
        fig.tight_layout()
    savefig(fig, out_dir, 'F_era5_4_seasonal_boxplots')


def plot_heatmap_precip(monthly: pd.DataFrame, out_dir: Path):
    """F_era5_5: Month × Year heatmap for precipitation."""
    pivot = monthly.pivot(index='month', columns='year', values='tp_mm')
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(pivot.values, aspect='auto', cmap='YlGnBu',
                       norm=mcolors.PowerNorm(gamma=0.5))
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        cb.set_label('Monthly Precipitation (mm)', fontsize=11)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=12)
        ax.set_yticks(range(12))
        ax.set_yticklabels(month_names, fontsize=10)
        ax.set_title('ERA5-Land: Monthly Precipitation Heatmap\nTelangana 2018–2020', fontsize=13, fontweight='bold')
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Month', fontsize=11)

        for i in range(12):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                        fontsize=9, color='black' if val < pivot.values.max() * 0.6 else 'white')

        _highlight_monsoon_rows(ax)
    savefig(fig, out_dir, 'F_era5_5_heatmap_precip')


def plot_heatmap_soilmoisture(monthly: pd.DataFrame, out_dir: Path):
    """F_era5_6: Month × Year heatmap for soil moisture."""
    pivot = monthly.pivot(index='month', columns='year', values='swvl1')
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(pivot.values, aspect='auto', cmap='BrBG')
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        cb.set_label('Soil Moisture (m³/m³)', fontsize=11)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=12)
        ax.set_yticks(range(12))
        ax.set_yticklabels(month_names, fontsize=10)
        ax.set_title('ERA5-Land: Monthly Soil Moisture Heatmap\nTelangana 2018–2020', fontsize=13, fontweight='bold')
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Month', fontsize=11)

        for i in range(12):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=8.5, color='black')

        _highlight_monsoon_rows(ax)
    savefig(fig, out_dir, 'F_era5_6_heatmap_soilmoisture')


def plot_annual_comparison(monthly: pd.DataFrame, out_dir: Path):
    """F_era5_7: Year-to-year comparison bar chart."""
    annual = monthly.groupby('year').agg(
        total_precip=('tp_mm', 'sum'),
        monsoon_precip=('tp_mm', lambda x: x[monthly.loc[x.index, 'month'].isin(MONSOON_MONTHS)].sum()),
        mean_sm=('swvl1', 'mean'),
        monsoon_sm=('swvl1', lambda x: x[monthly.loc[x.index, 'month'].isin(MONSOON_MONTHS)].mean()),
    ).reset_index()

    years = annual['year'].tolist()
    x = np.arange(len(years))
    w = 0.35

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))

        # Precipitation bars
        ax = axes[0]
        b1 = ax.bar(x - w/2, annual['total_precip'], width=w, label='Annual total',
                    color=[YEAR_COLORS[y] for y in years], alpha=0.9, edgecolor='white', linewidth=1.2)
        b2 = ax.bar(x + w/2, annual['monsoon_precip'], width=w, label='Monsoon (Jun–Sep)',
                    color=[YEAR_COLORS[y] for y in years], alpha=0.55, edgecolor='white',
                    linewidth=1.2, hatch='///')
        for bar in b1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=10)
        for bar in b2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=10, color='#555')
        ax.set_xticks(x)
        ax.set_xticklabels(years, fontsize=12)
        ax.set_ylabel('Precipitation (mm)', fontsize=11)
        ax.set_title('Annual vs Monsoon Precipitation', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)

        # Soil moisture bars
        ax = axes[1]
        b3 = ax.bar(x - w/2, annual['mean_sm'], width=w, label='Annual mean',
                    color=[YEAR_COLORS[y] for y in years], alpha=0.9, edgecolor='white', linewidth=1.2)
        b4 = ax.bar(x + w/2, annual['monsoon_sm'], width=w, label='Monsoon mean',
                    color=[YEAR_COLORS[y] for y in years], alpha=0.55, edgecolor='white',
                    linewidth=1.2, hatch='///')
        for bar in b3:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(years, fontsize=12)
        ax.set_ylabel('Soil Moisture (m³/m³)', fontsize=11)
        ax.set_title('Annual vs Monsoon Soil Moisture', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)

        fig.suptitle('ERA5-Land Year-to-Year Comparison — Telangana', fontsize=14, fontweight='bold')
        fig.tight_layout()
    savefig(fig, out_dir, 'F_era5_7_annual_comparison')


def plot_scatter_precip_sm(monthly: pd.DataFrame, out_dir: Path):
    """F_era5_8: Precipitation vs soil moisture scatter coloured by month."""
    cmap = plt.colormaps.get_cmap('twilight_shifted')
    norm = mcolors.Normalize(vmin=1, vmax=12)

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 7))

        for _, row in monthly.iterrows():
            color = cmap(norm(row['month']))
            marker = {2018: 'o', 2019: 's', 2020: '^'}.get(int(row['year']), 'o')
            ax.scatter(row['tp_mm'], row['swvl1'], c=[color], s=90,
                       marker=marker, edgecolors='white', linewidths=0.5, zorder=3)

        # Colorbar for month
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, ticks=range(1, 13), fraction=0.035, pad=0.04)
        cb.set_ticklabels(month_names)
        cb.set_label('Month', fontsize=11)

        # Trend line
        x = monthly['tp_mm'].values
        y = monthly['swvl1'].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() > 2:
            z = np.polyfit(x[mask], y[mask], 1)
            p = np.poly1d(z)
            xfit = np.linspace(x.min(), x.max(), 100)
            ax.plot(xfit, p(xfit), 'k--', linewidth=1.5, label=f'Trend (slope={z[0]:.5f})')
            ax.legend(fontsize=10)

        # Marker legend for years
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker='o', color='gray', markersize=9, label='2018', linestyle='None'),
            Line2D([0], [0], marker='s', color='gray', markersize=9, label='2019', linestyle='None'),
            Line2D([0], [0], marker='^', color='gray', markersize=9, label='2020', linestyle='None'),
        ]
        ax.legend(handles=handles, title='Year', loc='upper left', fontsize=10)

        r = np.corrcoef(x[mask], y[mask])[0, 1]
        ax.set_xlabel('Monthly Precipitation (mm)', fontsize=11)
        ax.set_ylabel('Mean Soil Moisture (m³/m³)', fontsize=11)
        ax.set_title(
            f'ERA5-Land: Precipitation vs Soil Moisture — Telangana 2018–2020\n'
            f'(Pearson r = {r:.3f})',
            fontsize=13, fontweight='bold'
        )
    savefig(fig, out_dir, 'F_era5_8_precip_vs_soilmoisture')


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _shade_monsoon(ax, index):
    for year in [2018, 2019, 2020]:
        for month in MONSOON_MONTHS:
            matches = index[(index.year == year) & (index.month == month)]
            if len(matches):
                ax.axvspan(matches[0] - pd.Timedelta(days=15),
                           matches[0] + pd.Timedelta(days=15),
                           alpha=0.08, color='steelblue', zorder=0)


def _highlight_monsoon_rows(ax):
    for m in MONSOON_MONTHS:
        ax.axhline(m - 1 - 0.5, color='steelblue', linewidth=0.4, alpha=0.4)
        ax.axhline(m - 1 + 0.5, color='steelblue', linewidth=0.4, alpha=0.4)
    # Draw a rectangle around monsoon rows
    from matplotlib.patches import FancyBboxPatch
    rect = FancyBboxPatch((-0.5, 4.5), len([2018, 2019, 2020]) - 0.01, 4,
                          linewidth=1.5, edgecolor='steelblue', facecolor='none',
                          boxstyle='round,pad=0.02', zorder=5)
    ax.add_patch(rect)


def _add_grid_labels(ax):
    ax.grid(True, linestyle='--', alpha=0.4, color='gray')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Plot ERA5-Land data for Telangana')
    parser.add_argument('--era5', default='/home/mohsen/data/era5/telangana_era5_land_2018_2020.nc')
    parser.add_argument('--out', default='outputs/paper_outputs/figures')
    args = parser.parse_args()

    out_dir = Path(args.out)

    logger.info(f'Loading ERA5 data from {args.era5} ...')
    ds = load_era5(args.era5)
    logger.info(f'  Variables: {list(ds.data_vars)}')
    logger.info(f'  Time range: {ds.time.values[0]} → {ds.time.values[-1]}')
    logger.info(f'  Grid: {len(ds.latitude)} lat × {len(ds.longitude)} lon')

    logger.info('Computing monthly aggregates ...')
    monthly = ds_to_monthly(ds)
    logger.info(f'  {len(monthly)} monthly records')

    logger.info('Generating plots ...')
    plot_spatial_precip(ds, out_dir)
    plot_spatial_soilmoisture(ds, out_dir)
    plot_monthly_timeseries(monthly, out_dir)
    plot_seasonal_boxplots(monthly, out_dir)
    plot_heatmap_precip(monthly, out_dir)
    plot_heatmap_soilmoisture(monthly, out_dir)
    plot_annual_comparison(monthly, out_dir)
    plot_scatter_precip_sm(monthly, out_dir)

    logger.info(f'\nAll 8 figures saved to {out_dir}/')
    print('\nFigures generated:')
    for f in sorted(out_dir.glob('F_era5_*.png')):
        print(f'  {f.name}')


if __name__ == '__main__':
    main()
