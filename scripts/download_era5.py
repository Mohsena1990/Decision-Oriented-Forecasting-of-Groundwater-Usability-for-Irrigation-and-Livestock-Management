"""
ERA5-Land Download Script
=========================

Downloads volumetric soil water (layer 1) and total precipitation
for Telangana, India (2018-2020) from the Copernicus Climate Data Store.

Prerequisites
-------------
    pip install cdsapi

Configure your CDS API key at ~/.cdsapirc:
    url: https://cds.climate.copernicus.eu/api/v2
    key: <your-uid>:<your-api-key>

    Your API key is available at:
    https://cds.climate.copernicus.eu/user/<username>

Usage
-----
    python scripts/download_era5.py --output /home/mohsen/data/era5

Output
------
    telangana_era5_land_2018_2020.nc — NetCDF4 file containing:
        * swvl1  (volumetric_soil_water_layer_1)  [m³/m³]
        * tp     (total_precipitation)             [m per hour, accumulation]
    at 0.1° grid resolution over Telangana (lat 15.8–19.9, lon 77.2–81.3)
    for every hour of 2018, 2019, 2020.
"""

import argparse
import logging
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Telangana approximate bounding box [North, West, South, East]
TELANGANA_BBOX = [19.9, 77.2, 15.8, 81.3]


def download_era5(output_dir: str = '/home/mohsen/data/era5') -> Path:
    """
    Download ERA5-Land hourly data for Telangana 2018-2020.

    Returns
    -------
    Path to the downloaded NetCDF file.
    """
    try:
        import cdsapi
    except ImportError:
        raise ImportError(
            "cdsapi not installed. Run: pip install cdsapi\n"
            "Then configure ~/.cdsapirc with your CDS API key."
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'telangana_era5_land_2018_2020.nc'

    if out_file.exists():
        logger.info(f"ERA5 file already exists: {out_file}")
        return out_file

    logger.info("Starting ERA5-Land download (one request per year to stay within CDS size limits)...")

    try:
        import xarray as xr
    except ImportError:
        raise ImportError("xarray not installed. Run: pip install xarray netcdf4")

    c = cdsapi.Client()
    year_files = []

    for year in ['2018', '2019', '2020']:
        year_file = out_dir / f'telangana_era5_land_{year}.nc'
        year_files.append(year_file)

        if year_file.exists():
            if zipfile.is_zipfile(str(year_file)):
                logger.info(f"  {year}: found zip on disk, extracting...")
                with zipfile.ZipFile(str(year_file)) as zf:
                    nc_names = [n for n in zf.namelist() if n.endswith('.nc')]
                    if not nc_names:
                        raise RuntimeError(f"No .nc file found inside zip for {year}: {zf.namelist()}")
                    extracted = out_dir / nc_names[0]
                    zf.extract(nc_names[0], out_dir)
                    year_file.unlink()
                    extracted.rename(year_file)
                logger.info(f"  {year}: extracted → {year_file}")
            else:
                logger.info(f"  {year}: already downloaded, skipping")
            continue

        logger.info(f"  Downloading {year}...")
        c.retrieve(
            'reanalysis-era5-land',
            {
                'variable': [
                    'volumetric_soil_water_layer_1',
                    'total_precipitation',
                ],
                'year': [year],
                'month': [
                    '01', '02', '03', '04', '05', '06',
                    '07', '08', '09', '10', '11', '12',
                ],
                'day': [
                    '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                    '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31',
                ],
                'time': ['00:00', '06:00', '12:00', '18:00'],
                'area': TELANGANA_BBOX,
                'format': 'netcdf',
            },
            str(year_file),
        )
        # CDS new API sometimes delivers a zip archive; extract the .nc inside it
        if zipfile.is_zipfile(str(year_file)):
            logger.info(f"  {year}: extracting NetCDF from zip archive...")
            with zipfile.ZipFile(str(year_file)) as zf:
                nc_names = [n for n in zf.namelist() if n.endswith('.nc')]
                if not nc_names:
                    raise RuntimeError(f"No .nc file found inside zip for {year}: {zf.namelist()}")
                extracted = out_dir / nc_names[0]
                zf.extract(nc_names[0], out_dir)
                year_file.unlink()          # remove the zip
                extracted.rename(year_file) # rename extracted file to expected path
            logger.info(f"  {year}: extracted → {year_file}")

        logger.info(f"  {year} done → {year_file}")

    logger.info("Merging yearly files...")
    datasets = [xr.open_dataset(str(f)) for f in year_files]
    ds = xr.concat(datasets, dim='valid_time') if 'valid_time' in datasets[0].dims else xr.concat(datasets, dim='time')
    ds.to_netcdf(str(out_file))
    ds.close()
    for d in datasets:
        d.close()

    logger.info(f"ERA5 download complete → {out_file}")
    return out_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download ERA5-Land data for Telangana')
    parser.add_argument(
        '--output', default='/home/mohsen/data/era5',
        help='Output directory for the NetCDF file'
    )
    args = parser.parse_args()
    path = download_era5(args.output)
    print(f"\nDownloaded: {path}")
    print("Next step: run scripts/download_ndvi_gee.py to get NDVI data.")
