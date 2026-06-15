"""
[DISABLED] Sentinel-2 NDVI Export via Google Earth Engine
==========================================================

This script requires a Google Earth Engine Cloud project registered at
https://console.cloud.google.com/earth-engine — currently not configured.

To enable NDVI features:
1. Register a GEE project at the URL above
2. Run: python -c "import ee; ee.Authenticate(auth_mode='notebook')"
3. Run: python scripts/download_ndvi_gee.py --project YOUR_PROJECT_ID
4. Un-comment the NDVI blocks in:
   - src/data/__init__.py
   - src/data/external_joiner.py
   - src/preprocessing/enhanced_pipeline.py
   - configs/main.yaml

-------------------------------------------------------------------------------
Original docstring below:
-------------------------------------------------------------------------------

Sentinel-2 NDVI Export via Google Earth Engine
===============================================

Exports mean and max NDVI (Oct-Nov, post-monsoon crop season) sampled at
every groundwater monitoring well location for 2018, 2019, 2020.

Oct-Nov is chosen because:
  * Post-monsoon groundwater surveys coincide with this window
  * Rabi crop establishment is visible (indicator of fertilizer application)
  * Cloud cover over Telangana is lowest after monsoon

Prerequisites
-------------
    pip install earthengine-api pandas

    Authenticate once:
        import ee; ee.Authenticate()

    Replace 'your-earth-engine-project-id' below with your GEE Cloud project.

Usage
-----
    python scripts/download_ndvi_gee.py \
        --wells /home/mohsen/data/archive/ground_water_quality_2018_post.csv \
        --output /home/mohsen/data/ndvi \
        --project your-earth-engine-project-id

Output
------
    /home/mohsen/data/ndvi/telangana_ndvi_2018_2020.csv
    Columns: lat, lon, year, ndvi_mean, ndvi_max, ndvi_p25, ndvi_p75

The script samples Sentinel-2 (COPERNICUS/S2_SR_HARMONIZED) at each well
centroid within a 500 m buffer and exports the Oct-Nov mean/max NDVI.
Cloud-contaminated pixels are masked using the SCL band.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Telangana bounding box [west, south, east, north]
TELANGANA_BBOX = [77.2, 15.8, 81.3, 19.9]
YEARS = [2018, 2019, 2020]
BUFFER_METERS = 500   # Buffer around well point for NDVI aggregation
MAX_CLOUD_PCT = 20    # Maximum CLOUDY_PIXEL_PERCENTAGE


def load_well_locations(wells_csv: str) -> pd.DataFrame:
    """
    Load unique (lat, lon) pairs from groundwater quality CSV.
    Deduplicates by rounding to 4 decimal places.
    """
    df = pd.read_csv(wells_csv)
    loc_cols = [c for c in ['district', 'mandal', 'village', 'lat_gis', 'long_gis'] if c in df.columns]
    wells = df[loc_cols].drop_duplicates().dropna(subset=['lat_gis', 'long_gis'])
    wells = wells.rename(columns={'lat_gis': 'lat', 'long_gis': 'lon'})
    wells['lat'] = pd.to_numeric(wells['lat'], errors='coerce')
    wells['lon'] = pd.to_numeric(wells['lon'], errors='coerce')
    wells = wells.dropna(subset=['lat', 'lon'])
    logger.info(f"Loaded {len(wells)} unique well locations from {wells_csv}")
    return wells.reset_index(drop=True)


def compute_ndvi_gee(wells: pd.DataFrame, project_id: str) -> pd.DataFrame:
    """
    For each well location and year, sample Oct-Nov mean/max NDVI
    from Sentinel-2 Surface Reflectance using GEE.
    """
    try:
        import ee
    except ImportError:
        raise ImportError(
            "earthengine-api not installed. Run: pip install earthengine-api"
        )

    creds_file = Path.home() / '.config' / 'earthengine' / 'credentials'
    if not creds_file.exists():
        ee.Authenticate(auth_mode='notebook')
    ee.Initialize(project=project_id)
    logger.info("Google Earth Engine initialised")

    def mask_clouds(image):
        scl = image.select('SCL')
        # SCL values 4 (vegetation) and 5 (non-vegetated) are cloud-free
        mask = scl.eq(4).Or(scl.eq(5))
        return image.updateMask(mask)

    def add_ndvi(image):
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)

    records = []

    for year in YEARS:
        logger.info(f"Processing NDVI for {year} Oct-Nov ...")

        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filter(ee.Filter.calendarRange(year, year, 'year'))
            .filter(ee.Filter.calendarRange(10, 11, 'month'))   # Oct-Nov
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_PCT))
            .map(mask_clouds)
            .map(add_ndvi)
            .select('NDVI')
        )

        mean_ndvi = collection.mean()
        max_ndvi = collection.max()

        # Sample at each well point
        for _, row in wells.iterrows():
            pt = ee.Geometry.Point([float(row['lon']), float(row['lat'])])
            buf = pt.buffer(BUFFER_METERS)

            try:
                mean_val = mean_ndvi.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=buf, scale=10, maxPixels=1e6
                ).get('NDVI').getInfo()

                max_val = max_ndvi.reduceRegion(
                    reducer=ee.Reducer.max(),
                    geometry=buf, scale=10, maxPixels=1e6
                ).get('NDVI').getInfo()

                record = {
                    'lat': float(row['lat']),
                    'lon': float(row['lon']),
                    'year': year,
                    'ndvi_oct_nov_mean': float(mean_val) if mean_val is not None else np.nan,
                    'ndvi_oct_nov_max': float(max_val) if max_val is not None else np.nan,
                }

                for col in ['district', 'mandal', 'village']:
                    if col in row.index:
                        record[col] = row[col]

                records.append(record)

            except Exception as e:
                logger.debug(f"GEE sampling failed for ({row['lat']:.4f},{row['lon']:.4f}) {year}: {e}")
                records.append({
                    'lat': float(row['lat']), 'lon': float(row['lon']),
                    'year': year,
                    'ndvi_oct_nov_mean': np.nan,
                    'ndvi_oct_nov_max': np.nan,
                })

        logger.info(f"  Year {year}: {len(wells)} wells sampled")

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description='Export Sentinel-2 NDVI for groundwater wells')
    parser.add_argument(
        '--wells',
        default='/home/mohsen/data/archive/ground_water_quality_2018_post.csv',
        help='Path to groundwater quality CSV (for well locations)'
    )
    parser.add_argument(
        '--output', default='/home/mohsen/data/ndvi',
        help='Output directory'
    )
    parser.add_argument(
        '--project', default='your-earth-engine-project-id',
        help='Google Earth Engine Cloud project ID'
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'telangana_ndvi_2018_2020.csv'

    if out_file.exists():
        logger.info(f"NDVI file already exists: {out_file}")
        return

    wells = load_well_locations(args.wells)
    ndvi_df = compute_ndvi_gee(wells, args.project)

    ndvi_df.to_csv(out_file, index=False)
    logger.info(f"NDVI data saved → {out_file} ({len(ndvi_df)} rows)")

    print(f"\nNDVI statistics:")
    print(ndvi_df[['ndvi_oct_nov_mean', 'ndvi_oct_nov_max']].describe())
    print(f"\nOutput: {out_file}")
    print("Next step: run the main pipeline — it will auto-load these files.")


if __name__ == '__main__':
    main()
