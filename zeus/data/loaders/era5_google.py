from typing import Tuple, Optional, List
import dask
import zarr
import math
import xarray as xr
import numpy as np
import pandas as pd
import torch

from zeus.data.era5.era5_base import Era5BaseLoader
from zeus.data.sample import Era5Sample
from zeus.validator.constants import (
    GCLOUD_ERA5_URL,
    MIN_INTERPOLATION_DISTORTIONS,
    MAX_INTERPOLATION_DISTORTIONS,
)


class ERA5GoogleLoader(Era5BaseLoader):

    def __init__(
        self,
        gcloud_url: str = GCLOUD_ERA5_URL,
        **kwargs,

    ) -> None:
        super().__init__(**kwargs)
        self.gcloud_url = gcloud_url
    
    def load_dataset(self) -> xr.Dataset:
        dataset = xr.open_zarr(self.gcloud_url, chunks=None) # don't chunk yet, that takes a lot of time.
        dataset = dataset[list(self.data_vars)] # slice out anything we won't use.
        return dataset

    def sample_time_range(self) -> Tuple[pd.Timestamp, pd.Timestamp, int]:
        """
        Sample a random time range from the dataset in the form of "Y-m-d" strings and a number of hours to predict.
        The start will always be the start of day,
        but the end could be any hour on a day after start.
        The hours to be predicted are also included in this range, but are not send to the miner.

        Returns:
         - start_timestamp (pd.Timestamp): The start of the time range.
         - end_timestamp (pd.Timestamp): The end of the time range.
         - num_predict_hours (int): The number of hours to be predicted.
        """
        latest_day = (
            (self.date_range[1] - self.date_range[0]).days 
            - math.floor(self.time_sample_range[1] / 24) 
            - math.floor(self.predict_sample_range[1] / 24)
        )
        start_timestamp = pd.to_timedelta(np.random.randint(0, latest_day), unit="D") + self.date_range[0]
        num_predict_hours = np.random.randint(*self.predict_sample_range)
        end_timestamp = pd.to_timedelta(np.random.randint(*self.time_sample_range) + num_predict_hours, unit="h") + start_timestamp
        return start_timestamp, end_timestamp, num_predict_hours
    
    def get_sample(self) -> Era5Sample:
        """
        Get a random sample from the dataset. This sample includes a tiny bit of noise on the input to prevent hashing lookups.

        Returns:
        - sample (Era5Sample): The sample containing the input and output data.
        """
        # get a random rectangular bounding box
        lat_start, lat_end, lon_start, lon_end = self.sample_bbox()
        start_time, end_time, predict_hours = self.sample_time_range()

        data = self.get_data(
            lat_start=lat_start,
            lat_end=lat_end,
            lon_start=lon_start,
            lon_end=lon_end,
            start_time=start_time,
            end_time=end_time
        )

        input_data = data[:-predict_hours]
        output_data = data[-predict_hours:]

        num_distortions = np.random.randint(MIN_INTERPOLATION_DISTORTIONS, MAX_INTERPOLATION_DISTORTIONS)
        input_data = interp_distort(input_data, num_distortions)
        # Add noise only to the the variables data
        noise = torch.randn(*input_data.shape[:-1], input_data.shape[-1] - 2) * self.noise_factor
        input_data[..., -1:] += noise

        # Slice off the latitude and longitude for the output
        output_data = output_data[..., 2:].squeeze()

        return Era5Sample(
            start_timestamp=start_time.timestamp(),
            end_timestamp=end_time.timestamp(),
            input_data=input_data,
            output_data=output_data
        )
    

def interp_distort(matrix: torch.Tensor, num_distortions: int) -> torch.Tensor:
    """
    Interpolate the input data with some random noise.
    """
    for _ in range(num_distortions):
        t = np.random.randint(1, matrix.shape[0] - 2)
        lat = np.random.randint(1, matrix.shape[1] - 2)
        lon = np.random.randint(1, matrix.shape[2] - 2)
        offset_t, offset_lat, offset_lon = np.random.choice([-1, 1], size=3, replace=True)
        alpha = np.random.uniform(0.0, 0.1)

        matrix[t, lat, lon, 2:] = (
            (1 - alpha) * matrix[t, lat, lon, 2:] + 
            alpha * matrix[t + offset_t, lat + offset_lat, lon + offset_lon, 2:]
        )

    return matrix