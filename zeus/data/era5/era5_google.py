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
    """
    A dataloader based on historical data from the ERA5 dataset stored on Google Cloud.
    Currently this dataset is NOT USED, as it would be too easy to lookup the correct answer for miners.
    The dataloader is provided mostly for reference, and a modified version might be implemented into the subnet in the future.
    """

    def __init__(
        self,
        gcloud_url: str = GCLOUD_ERA5_URL,
        **kwargs,
    ) -> None:
        self.gcloud_url = gcloud_url
        super().__init__(**kwargs)

    def load_dataset(self) -> xr.Dataset:
        dataset = xr.open_zarr(
            self.gcloud_url, chunks=None
        )  # don't chunk yet, that takes a lot of time.
        dataset = dataset[list(self.data_vars)]  # slice out anything we won't use.
        return dataset

    def sample_time_range(self) -> Tuple[pd.Timestamp, pd.Timestamp, int]:
        """
        Sample a random time range from the dataset in the form of pandas datetimes and a number of hours to predict.
        The start will always be the start of day,
        but the end could be any hour on a day after start.

        Returns:
         - start_timestamp (pd.Timestamp): The start of the time range.
         - end_timestamp (pd.Timestamp): The end of the time range.
         - num_predict_hours (int): The number of hours to be predicted (difference between start and end).
        """
        latest_day = (self.date_range[1] - self.date_range[0]).days - math.floor(
            self.predict_sample_range[1] / 24
        )
        start_timestamp = self.date_range[0] + pd.Timedelta(
            days=np.random.randint(0, latest_day)
        )
        num_predict_hours = np.random.randint(*self.predict_sample_range)
        end_timestamp = start_timestamp + pd.Timedelta(hours=num_predict_hours)
        return start_timestamp, end_timestamp, num_predict_hours

    def get_sample(self) -> Era5Sample:
        """
        Get a random sample from the dataset.

        Returns:
        - sample (Era5Sample): The sample containing the input and output data.
        """
        # get a random rectangular bounding box
        lat_start, lat_end, lon_start, lon_end = self.sample_bbox()
        start_time, end_time, _ = self.sample_time_range()

        data = self.get_data(
            lat_start=lat_start,
            lat_end=lat_end,
            lon_start=lon_start,
            lon_end=lon_end,
            start_time=start_time,
            end_time=end_time,
        )

        x_grid = data[0, ..., :2]
        # Slice off the latitude and longitude for the output
        output_data = data[..., 2:].squeeze()

        return Era5Sample(
            start_timestamp=start_time.timestamp(),
            end_timestamp=end_time.timestamp(),
            x_grid=x_grid,
            output_data=output_data,
        )


def interp_distort(matrix: torch.Tensor, num_distortions: int) -> torch.Tensor:
    """
    Interpolate the input data with some random noise.
    """
    for _ in range(num_distortions):
        t = np.random.randint(1, matrix.shape[0] - 2)
        lat = np.random.randint(1, matrix.shape[1] - 2)
        lon = np.random.randint(1, matrix.shape[2] - 2)
        offset_t, offset_lat, offset_lon = np.random.choice(
            [-1, 1], size=3, replace=True
        )
        alpha = np.random.uniform(0.0, 0.1)

        matrix[t, lat, lon, 2:] = (1 - alpha) * matrix[
            t, lat, lon, 2:
        ] + alpha * matrix[t + offset_t, lat + offset_lat, lon + offset_lon, 2:]

    return matrix
