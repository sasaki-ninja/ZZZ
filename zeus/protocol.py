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

import typing
import bittensor as bt
import torch

from pydantic import Field
from typing import List, Tuple


# This is the protocol for the dummy miner and validator.
# It is a simple request-response protocol where the validator sends a request
# to the miner, and the miner responds with a dummy response.

# ---- miner ----
# Example usage:
#   def dummy( synapse: Dummy ) -> Dummy:
#       synapse.dummy_output = synapse.dummy_input + 1
#       return synapse
#   axon = bt.axon().attach( dummy ).serve(netuid=...).start()

# ---- validator ---
# Example usage:
#   dendrite = bt.dendrite()
#   dummy_output = dendrite.query( Dummy( dummy_input = 1 ) )
#   assert dummy_output == 2


class TimePredictionSynapse(bt.Synapse):
    """
    A protocol representation which uses bt.Synapse as its base.
    This protocol helps in handling request and response communication between
    the miner and the validator.
    """

    version: str = Field(
        title="Validator/Miner codebase version",
        description="Version matches the version-string of the SENDER, either validator or miner",
        default = "",
        frozen = False,
    )

    # Required request input, filled by sending dendrite caller.
    locations: List[List[Tuple[float, float]]] = Field(
        title="Locations to predict",
        description="Locations to predict. Represents a grid of (latitude, longitude) pairs.",
        default=[],
        frozen=False,
    )

    start_time: float = Field(
        title="start timestamp",
        description="Starting timestamp in GMT+0 as a float",
        default=0.0,
        frozen=False,
    )

    end_time: float = Field(
        title="end timestamp",
        description="Ending timestamp in GMT+0 as a float",
        default=0.0,
        frozen=False,
    )

    requested_hours: int = Field(
        title="Number of hours",
        description="The number of desired output hours for the prediction.",
        default=1,
        frozen=False,
    )

    # Optional request output, filled by receiving axon.
    predictions: List[List[List[float]]] = Field(
        title="Prediction",
        description="The output tensor to be scored.",
        default=[],
        frozen=False,
    )

    def deserialize(self) -> torch.Tensor:
        """
        Deserialize the output. This method retrieves the response from
        the miner, deserializes it and returns it as the output of the dendrite.query() call.

        Returns:
        - np.ndarray: The deserialized response
        """
        return torch.tensor(self.predictions)
