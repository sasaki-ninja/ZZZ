from typing import Tuple, Optional, List
import dask
import zarr
import math
import xarray as xr
import numpy as np
import pandas as pd
import torch


class ERA5DataLoader:

    def __init__(
        self,
        gcloud_url: str = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        data_vars: List[str] = ["2m_temperature"],
        lat_range: Tuple[float, float] = (-90.0, 90.0),
        lon_range: Tuple[float, float] = (-180.0, 180.0),
        date_range: Tuple[str, str] = ("1960-01-01", "2023-01-01"),
        area_sample_range: Tuple[float, float] = (2, 4), # in degrees, there is 4 measurements per degree.
        time_sample_range: Tuple[int, int] = (72, 120), # in hours
        predict_sample_range: Tuple[float, float] = (1, 24), # in hours
        noise_factor: float = 1e-6,

    ) -> None:
        self.data_vars = data_vars
        self.noise_factor = noise_factor

        self.lat_range = sorted(lat_range)
        self.lon_range = sorted(lon_range)
        self.date_range = list(map(pd.to_datetime, sorted(date_range))) # store datetimes

        self.area_sample_range = sorted(area_sample_range)
        self.time_sample_range = sorted(time_sample_range)
        self.predict_sample_range = sorted(predict_sample_range)

        dataset = xr.open_zarr(gcloud_url, chunks=None) # don't chunk yet, that takes a lot of time.
        dataset = dataset[list(data_vars)]

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
        but the end could be any hour.
        The hours to be predicted are also included in this range, but are not send to the miner.
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


    def get_random_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a random sample from the dataset. This sample includes a tiny bit of noise to prevent hashing lookups.

        Returns:
         - Data to be send to miner as a 4D grid (hours, latitudes, longitudes, variables), 
           where 'variables' contains both the lat and lon values and the data variable(s).
         - Data the miner should aim to predict. Also a 4D grid of same dimensions. 
           Note that miners should only predict the data variables, not the lat and lon values.
        """
        
        lat_start = np.random.uniform(self.lat_range[0], self.lat_range[1] - self.area_sample_range[1])
        lat_end = lat_start + np.random.uniform(*self.area_sample_range)
        lon_start = np.random.randint(self.lon_range[0], self.lon_range[1] - self.area_sample_range[1])
        lon_end = lon_start + np.random.uniform(*self.area_sample_range)

        start_time, end_time, predict_hours = self._sample_time_range()

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
                ) # (time, lat, lon, vars)
        noise = torch.randn_like(y_grid) * self.noise_factor
        y_grid = y_grid + noise

        data = torch.cat([x_grid, y_grid], dim=-1)

        return data[:-predict_hours], data[-predict_hours:]