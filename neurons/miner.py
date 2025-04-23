# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Eric (Ørpheus A.I.)
# Copyright © 2025 Ørpheus A.I.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import torch
import typing
import bittensor as bt

import openmeteo_requests

import numpy as np
from zeus.utils.misc import celcius_to_kelvin
from zeus.utils.config import get_device_str
from zeus.utils.time import get_timestamp
from zeus.protocol import TimePredictionSynapse, HistoricPredictionSynapse
from zeus.base.miner import BaseMinerNeuron
from zeus import __version__ as zeus_version


class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior.
    In particular, you should replace the forward function with your own logic.

    Currently the base miner does a request to OpenMeteo (https://open-meteo.com/) for current/future ('live') predictions.
    You are encouraged to attempt to improve over this by changing the forward function.

    For historic predictions, the base miner simply repeats the last hour of input data to match the required shape.
    To be competitive, you will need to implement your own intelligent forecasting model for this reduced-data setting.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        bt.logging.info("Attaching forward functions to miner axon.")
        self.axon.attach(
            forward_fn=self.forward_live,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        ).attach(
            forward_fn=self.forward_historic,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )
        
        # TODO(miner): Anything specific to your use case you can do here
        self.device: torch.device = torch.device(get_device_str())
        self.openmeteo_api = openmeteo_requests.Client()

    async def forward_live(self, synapse: TimePredictionSynapse) -> TimePredictionSynapse:
        """
        Processes the incoming TimePredictionSynapse for a current/future prediction.

        Args:
            synapse (TimePredictionSynapse): The synapse object containing the time range and coordinates

        Returns:
            TimePredictionSynapse: The synapse object with the 'predictions' field set".
        """
        # shape (lat, lon, 2) so a grid of locations
        coordinates = torch.Tensor(synapse.locations)
        start_time = get_timestamp(synapse.start_time)
        end_time = get_timestamp(synapse.end_time)
        bt.logging.info(
            f"Received current/future request! Predicting {synapse.requested_hours} hours for grid of shape {coordinates.shape}."
        )

        ##########################################################################################################
        # TODO (miner) you likely want to improve over this baseline of calling OpenMeteo by changing this section
        latitudes, longitudes = coordinates.view(-1, 2).T
        params = {
            "latitude": latitudes.tolist(),
            "longitude": longitudes.tolist(),
            "hourly": "temperature_2m",
            "start_date": start_time.strftime("%Y-%m-%d"),
            "end_date": end_time.strftime("%Y-%m-%d"),
        }
        responses = self.openmeteo_api.weather_api(
            "https://api.open-meteo.com/v1/forecast", params=params
        )
        # get temperature output as grid of [time, lat, lon]
        output = np.stack(
            [r.Hourly().Variables(0).ValuesAsNumpy() for r in responses], axis=1
        ).reshape(-1, coordinates.shape[0], coordinates.shape[1])
        # OpenMeteo does Celcius, scoring is based on Kelvin
        output = celcius_to_kelvin(output)

        # OpenMeteo returns full days, so slice, make sure end hour is included only for up-to-date validators.
        output = output[start_time.hour : (-23 + end_time.hour)]
        ##########################################################################################################
        bt.logging.info(f"Output shape is {output.shape}")

        synapse.predictions = output.tolist()
        synapse.version = zeus_version
        return synapse
    

    async def forward_historic(self, synapse: HistoricPredictionSynapse) -> HistoricPredictionSynapse:
        """
        Processes the incoming HistoricPredictionSynapse based on its input data and optionally the location.
        # NOTE: the location is likely to be near the middle of the box, but there is some gaussian noise in its location.

        Args:
            synapse (HistoricPredictionSynapse): The synapse object containing the input data.

        Returns:
            HistoricPredictionSynapse: The synapse object with the 'predictions' field set".
        """

        bt.logging.info(
            f"Received historic request! Predicting {synapse.requested_hours} near {synapse.location}."
        )
        # shape (time, lat, lon) containing input temperature data
        input_data = torch.Tensor(synapse.input_data)

        ##########################################################################################################
        # TODO (miner) you might want to do something more intelligent than repeating the last measurement
        output = input_data[-1]
        output = output.expand(synapse.requested_hours, *output.shape)
        ##########################################################################################################
        bt.logging.info(f"Output shape is {output.shape}")

        synapse.predictions = output.tolist()
        synapse.version = zeus_version
        return synapse


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"Miner running | uid {miner.uid} | {time.time()}")
            time.sleep(30)
