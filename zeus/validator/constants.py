from typing import List, Tuple
from pathlib import Path

TESTNET_UID = 301

FORWARD_DELAY_SECONDS = 120

# ERA5 data loading constants
GCLOUD_ERA5_URL: str = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)
ERA5_CACHE_DIR: Path = Path.home() / ".cache" / "zeus" / "era5"
DATABASE_LOCATION: Path = Path.home() / ".cache" / "zeus" / "challenges.db"
COPERNICUS_ERA5_URL: str = "https://cds.climate.copernicus.eu/api"

ERA5_DATA_VARS: List[str] = ["2m_temperature"]

ERA5_DATE_RANGE: Tuple[str, str] = (
    "1960-01-01",
    "2024-10-31",
)  # current latest inside that Zarr archive - not used currently

ERA5_LATITUDE_RANGE: Tuple[float, float] = (-90.0, 90.0)
ERA5_LONGITUDE_RANGE: Tuple[float, float] = (-180.0, 179.75)  # real ERA5 ranges
ERA5_AREA_SAMPLE_RANGE: Tuple[float, float] = (
    4,
    12,
)  # how many datapoints we want. The resolution is 0.25 degrees, so 4 means 1 degree.
ERA5_OLDEST_START_HOUR: int = -119  # 4 days and 23 hours ago.
ERA5_HOURS_PREDICT_RANGE: Tuple[float, float] = (
    1,
    24,
)  # in hours, how many hours ahead we want to predict.
ERA5_START_SAMPLE_STD: float = (
    40  # see plot of distribution in Zeus/static/era5_start_offset_distribution.png
)

# Augmentation constants - not used currently
MIN_INTERPOLATION_DISTORTIONS = 5
MAX_INTERPOLATION_DISTORTIONS = 50

# Reward constants
DIFFICULTY_OFFSET = 1.0  # twice the average of the Difficulty_grid
DIFFICULTY_MULTIPLIER = 2.0  # make it easier to get a high score initially
