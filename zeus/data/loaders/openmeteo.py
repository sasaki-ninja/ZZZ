import os
import openmeteo_requests

import numpy as np
import torch

from zeus.data.sample import Era5Sample
from zeus.data.converter import get_converter
from zeus.utils.time import to_timestamp
from zeus.validator.constants import (
    OPEN_METEO_URL
)

class OpenMeteoLoader:

    def __init__(
        self,
        open_meteo_url = OPEN_METEO_URL,
    ) -> None:
        
        self.api_key = os.getenv("OPEN_METEO_API_KEY")
        self.open_meteo_url = open_meteo_url
        self.open_meteo_api = openmeteo_requests.Client()

    def get_output(self, sample: Era5Sample) -> torch.Tensor:
        start_time = to_timestamp(sample.start_timestamp)
        end_time = to_timestamp(sample.end_timestamp)

        latitudes, longitudes = sample.x_grid.view(-1, 2).T
        converter = get_converter(sample.variable)
        params = {
            "latitude": latitudes.tolist(),
            "longitude": longitudes.tolist(),
            "hourly": converter.om_name,
            "start_hour": start_time.isoformat(timespec="minutes"),
            "end_hour": end_time.isoformat(timespec="minutes"),
            "apikey": self.api_key
        }

        responses = self.open_meteo_api.weather_api(
            self.open_meteo_url, params=params
        )
        # get output as grid of [time, lat, lon]
        output = np.stack(
            [r.Hourly().Variables(0).ValuesAsNumpy() for r in responses], axis=1
        ).reshape(-1, sample.x_grid.shape[0], sample.x_grid.shape[1])
        # OpenMeteo does not use SI units, so convert back
        output = converter.om_to_era5(output)

        assert output.shape[0] == sample.predict_hours, f"Invalid OpenMeteo response shape"
        return output
