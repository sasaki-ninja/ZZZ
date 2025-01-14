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
import bittensor as bt

import torch
from climate.protocol import TimePredictionSynapse
from climate.validator.reward import get_rewards
from climate.utils.uids import get_random_uids


async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    # Let's sample some data
    bt.logging.info(f"Sampling data...")
    input_data, output_data = self.data_loader.get_random_sample()
    output_data = output_data[..., 2:].squeeze() # slice off the latitude and longitude, miner's don't need to return that it

    bt.logging.success(f"Data sampled. Input shape: {input_data.shape} | Output shape: {output_data.shape}")	

    # get_random_uids is an example method, but you can replace it with your own.
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    axons=[self.metagraph.axons[uid] for uid in miner_uids]

    synapse = TimePredictionSynapse(input_data=input_data.tolist(), requested_hours=output_data.shape[0])

    # The dendrite client queries the network.
    bt.logging.info(f"Querying {len(miner_uids)} miners..")
    start = time.time()
    responses = await self.dendrite(
        axons=axons,
        synapse=synapse,
        deserialize=True,
        timeout=self.config.neuron.timeout,
    )

    bt.logging.success(f"Responses received in {time.time() - start}s")
    # Score miners
    rewards = get_rewards(correct_outputs=output_data, responses=responses)
     # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    self.update_scores(rewards, miner_uids)

    for uid, response, reward in zip(miner_uids, responses, rewards):
        if len(response) != 0:
            bt.logging.success(f"UID: {uid} | Predicted shape: {response.shape} | Reward: {reward}")

    if not self.config.wandb.off:
        # TODO, actually log WandB, maybe cool to log plots as well?
        pass
