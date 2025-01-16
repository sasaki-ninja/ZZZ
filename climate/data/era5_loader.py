from typing import Tuple, Optional, List
import dask
import zarr
import math
import xarray as xr
import numpy as np
import pandas as pd
import torch

from climate.data.sample import Era5Sample
from climate.validator.constants import (
    GCLOUD_ERA5_URL,
    ERA5_DATA_VARS,
    ERA5_LATITUDE_RANGE,
    ERA5_LONGITUDE_RANGE,
    ERA5_DATE_RANGE,
    ERA5_AREA_SAMPLE_RANGE,
    ERA5_HOURS_SAMPLE_RANGE,
    ERA5_HOURS_PREDICT_RANGE,
)


class ERA5DataLoader:

    def __init__(
        self,
        gcloud_url: str = GCLOUD_ERA5_URL,
        data_vars: List[str] = ERA5_DATA_VARS,
        lat_range: Tuple[float, float] = ERA5_LATITUDE_RANGE,
        lon_range: Tuple[float, float] = ERA5_LONGITUDE_RANGE,
        date_range: Tuple[str, str] = ERA5_DATE_RANGE,
        area_sample_range: Tuple[float, float] = ERA5_AREA_SAMPLE_RANGE, # in degrees, there is 4 measurements per degree.
        time_sample_range: Tuple[int, int] = ERA5_HOURS_SAMPLE_RANGE,
        predict_sample_range: Tuple[float, float] = ERA5_HOURS_PREDICT_RANGE,
        noise_factor: float = 1e-6,

    ) -> None:
        self.data_vars = data_vars
        self.noise_factor = noise_factor

        self.lat_range = sorted(lat_range)
        self.lon_range = sorted(lon_range)
        self.date_range = list(map(pd.to_datetime, sorted(date_range)))

        self.area_sample_range = sorted(area_sample_range)
        self.time_sample_range = sorted(time_sample_range)
        self.predict_sample_range = sorted(predict_sample_range)

        dataset = xr.open_zarr(gcloud_url, chunks=None) # don't chunk yet, that takes a lot of time.
        dataset = dataset[list(data_vars)] # slice out anything we won't use.

        # ensure the coordinates are (-90, 90) and (-180, 180) for latitude and longitude respectively.
        if dataset["longitude"].max() > 180:
            dataset = dataset.assign_coords(longitude=(dataset["longitude"].values + 180) % 360 - 180)
        if dataset["latitude"].max() > 90:
            dataset = dataset.assign_coords(latitude=dataset["latitude"].values - 90)

        self.dataset = dataset.sortby(["latitude", "longitude"])

    def _sample_time_range(self) -> Tuple[pd.Timestamp, pd.Timestamp, int]:
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


    def get_data(
            self, 
            lat_start: float,
            lat_end: float,
            lon_start: float,
            lon_end: float,
            start_time: pd.Timestamp, 
            end_time: pd.Timestamp,
    ) -> torch.Tensor:
        """
        Get a sample from the dataset for a specific location and time range.

        Returns:
         - sample (torch.Tensor): The sample containing the input and output data as a 4D tensor.
        """

        subset = self.dataset.sel(
            latitude=slice(lat_start, lat_end), 
            longitude=slice(lon_start, lon_end),
            time=slice(start_time, end_time)
        ).chunk()

        subset = subset.compute() # heavy loading - fetch the actual data here.

        x_grid = torch.stack(
                    torch.meshgrid(
                        *[
                            torch.as_tensor(
                                subset[v].data, dtype=torch.float
                            )
                            for v in ("latitude", "longitude")
                        ],
                        indexing="ij",
                    ),
                    dim=-1,
                ) # (lat, lon, 2)
        x_grid = x_grid.expand(len(subset.time), *x_grid.shape) # (time, lat, lon, 2)
        
        y_grid = torch.stack(
                    [
                        torch.as_tensor(
                            subset[var].data, dtype=torch.float
                        )
                        for var in self.data_vars
                    ], 
                    dim=-1
                ) # (time, lat, lon, data_vars)
        data = torch.cat([x_grid, y_grid], dim=-1)

        return data


    def get_random_sample(self) -> Era5Sample:
        """
        Get a random sample from the dataset. This sample includes a tiny bit of noise on the input to prevent hashing lookups.

        Returns:
         - sample (Era5Sample): The sample containing the input and output data.
        """
        # get a random rectangular bounding box
        lat_start = np.random.uniform(self.lat_range[0], self.lat_range[1] - self.area_sample_range[1])
        lat_end = lat_start + np.random.uniform(*self.area_sample_range)
        lon_start = np.random.randint(self.lon_range[0], self.lon_range[1] - self.area_sample_range[1])
        lon_end = lon_start + np.random.uniform(*self.area_sample_range)

        start_time, end_time, predict_hours = self._sample_time_range()

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

        # add noise to the input to make it impossible to hash-lookup the entire dataset
        noise = torch.randn_like(input_data) * self.noise_factor
        input_data = input_data + noise

        # slice off the latitude and longitude, miner's don't need to return that
        output_data = output_data[..., 2:].squeeze()
        
        return Era5Sample(
            start_timestamp=start_time.timestamp(),
            end_timestamp=end_time.timestamp(),
            input_data=input_data,
            output_data=output_data
        )