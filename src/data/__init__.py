"""Data ingestion, harmonization, and external enrichment module."""

from .ingestion import DataIngestion
from .harmonization import DataHarmonizer
from .transitions import TransitionBuilder
from .era5_loader import ERA5Loader, ERA5_FEATURES, load_era5_features
# from .ndvi_loader import NDVILoader, NDVI_FEATURES, load_ndvi_features  # GEE disabled
from .external_joiner import ExternalFeatureJoiner, enrich_with_external_data, ALL_EXTERNAL_FEATURES

__all__ = [
    "DataIngestion",
    "DataHarmonizer",
    "TransitionBuilder",
    "ERA5Loader",
    "ERA5_FEATURES",
    "load_era5_features",
    # "NDVILoader",          # GEE disabled
    # "NDVI_FEATURES",       # GEE disabled
    # "load_ndvi_features",  # GEE disabled
    "ExternalFeatureJoiner",
    "enrich_with_external_data",
    "ALL_EXTERNAL_FEATURES",
]
