from typing import List, Tuple

MAINNET_UID = 1 # TODO change
TESTNET_UID = 2


# ERA5 data loading constants
GCLOUD_ERA5_URL: str = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ERA5_DATA_VARS: List[str] = ["2m_temperature"]

ERA5_DATE_RANGE: Tuple[str, str] = ("1960-01-01", "2024-10-31") # current latest inside that Zarr archive
ERA5_LATITUDE_RANGE: Tuple[float, float] = (-90.0, 90.0)
ERA5_LONGITUDE_RANGE: Tuple[float, float] = (-180.0, 180.0)
ERA5_AREA_SAMPLE_RANGE: Tuple[float, float] = (2, 4) # in degrees, there is 4 measurements per degree.
ERA5_HOURS_SAMPLE_RANGE: Tuple[int, int] = (72, 120) # in hours
ERA5_HOURS_PREDICT_RANGE: Tuple[float, float] = (1, 24) # in hours

MIN_INTERPOLATION_DISTORTIONS = 5
MAX_INTERPOLATION_DISTORTIONS = 50

DIFFICULTY_OFFSET = 0.45
DIFFICULTY_MULTIPLIER = 1.0