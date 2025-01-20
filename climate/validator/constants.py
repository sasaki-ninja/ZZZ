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

# Difficulty per climate type: A lower value indicates an easier prediction task, 
# making it more challenging for miners to balance overall scores. 
# These values are based on the relative ratio of the mean RMSE of state-of-the-art (SOTA) AI weather forecasting models.
DIFFICULTIES = {
    'Af': 2,
    'Am': 2,
    'As': 2,
    'Aw': 2,
    'BSh': 2,
    'BSk': 2,
    'BWh': 2,
    'BWk': 2,
    'Cfa': 2,
    'Cfb': 2,
    'Cfc': 2,
    'Csa': 2,
    'Csb': 2,
    'Csc': 2,
    'Cwa': 2,
    'Cwb': 2,
    'Cwc': 2,
    'Dfa': 2,
    'Dfb': 2,
    'Dfc': 2,
    'Dfd': 2,
    'Dsa': 2,
    'Dsb': 2,
    'Dsc': 2,
    'Dwa': 2,
    'Dwb': 2,
    'Dwc': 2,
    'Dwd': 2,
    'EF': 2,
    'ET': 2,
    'UNKNOWN': 1,
    }