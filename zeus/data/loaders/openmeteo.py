import os
import openmeteo_requests

import numpy as np
import torch

from zeus.data.sample import Era5Sample
from zeus.utils.time import to_timestamp
from zeus.utils.misc import celcius_to_kelvin
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
        params = {
            "latitude": latitudes.tolist(),
            "longitude": longitudes.tolist(),
            "hourly": "temperature_2m",
            "start_hour": start_time.isoformat(timespec="minutes"),
            "end_hour": end_time.isoformat(timespec="minutes"),
            "apikey": self.api_key
        }

        responses = self.open_meteo_api.weather_api(
            self.open_meteo_url, params=params
        )
        # get temperature output as grid of [time, lat, lon]
        output = np.stack(
            [r.Hourly().Variables(0).ValuesAsNumpy() for r in responses], axis=1
        ).reshape(-1, sample.x_grid.shape[0], sample.x_grid.shape[1])
        # OpenMeteo does Celcius, scoring is based on Kelvin
        output = celcius_to_kelvin(output)

        assert output.shape[0] == sample.predict_hours, f"Invalid OpenMeteo response shape"
        return output
