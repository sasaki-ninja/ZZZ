from typing import List, Tuple
from pathlib import Path

TESTNET_UID = 301
MAINNET_UID = 18

FORWARD_DELAY_SECONDS = 360

# ERA5 data loading constants
GCLOUD_ERA5_URL: str = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)
ERA5_CACHE_DIR: Path = Path.home() / ".cache" / "zeus" / "era5"
DATABASE_LOCATION: Path = Path.home() / ".cache" / "zeus" / "challenges.db"
COPERNICUS_ERA5_URL: str = "https://cds.climate.copernicus.eu/api"

ERA5_DATA_VARS: List[str] = ["2m_temperature"]

ERA5_LATITUDE_RANGE: Tuple[float, float] = (-90.0, 90.0)
ERA5_LONGITUDE_RANGE: Tuple[float, float] = (-180.0, 179.75)  # real ERA5 ranges

# how many datapoints we want. The resolution is 0.25 degrees, so 4 means 1 degree.
ERA5_AREA_SAMPLE_RANGE: Tuple[float, float] = (4,12,) 

ERA5_START_OFFSET_RANGE: Tuple[int, int] = (-119, 168)  # 4 days and 23 hours ago <---> until 7 days in future
ERA5_UNIFORM_START_OFFSET_PROB: float = 0.1
ERA5_HOURS_PREDICT_RANGE: Tuple[float, float] = (1, 24) # how many hours ahead we want to predict.

# see plot of distribution in Zeus/static/era5_start_offset_distribution.png
ERA5_START_SAMPLE_STD: float = 35 

# 1.0 would imply no difficulty scaling, should be >= 1.
REWARD_DIFFICULTY_SCALER = 3.0

# ------------------------------------------------------
# ------------------ Unused Constants ------------------
# ------------------------------------------------------
ERA5_DATE_RANGE: Tuple[str, str] = (
    "1960-01-01",
    "2024-10-31",
)  # current latest inside that Zarr archive - not used currently

# Augmentation constants - not used currently
MIN_INTERPOLATION_DISTORTIONS = 5
MAX_INTERPOLATION_DISTORTIONS = 50

