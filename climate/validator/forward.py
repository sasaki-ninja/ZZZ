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
import wandb

from climate.data.sample import Era5Sample
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
    sample: Era5Sample = self.data_loader.get_random_sample()

    bt.logging.success(f"Data sampled. Input shape: {sample.input_data.shape} | Output shape: {sample.output_data.shape}")	

    # get some miners
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    axons=[self.metagraph.axons[uid] for uid in miner_uids]

    # The dendrite client queries the network.
    bt.logging.info(f"Querying {len(miner_uids)} miners..")
    start = time.time()
    responses = await self.dendrite(
        axons=axons,
        synapse=sample.get_synapse(),
        deserialize=True,
        timeout=self.config.neuron.timeout,
    )

    bt.logging.success(f"Responses received in {time.time() - start}s")
    # Score miners
    rewards, metrics = get_rewards(correct_outputs=sample.output_data, responses=responses)
     # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    self.update_scores(rewards, miner_uids)

    miners_scores = {}
    for uid, response, reward in zip(miner_uids, responses, rewards):
        if len(response) != 0:
            miners_scores[uid] = reward
            bt.logging.success(f"UID: {uid} | Predicted shape: {response.shape} | Reward: {reward}")
    # store best miners for the Proxy
    self.last_responding_miner_uids = sorted(miners_scores, key=miners_scores.get, reverse=True)

    if not self.config.wandb.off:
        for miner_uid, metric_dict in zip(miner_uids, metrics):
            wandb.log(
                {
                    f"miner_{miner_uid}_{key}": val 
                    for key, val in metric_dict.items()
                },
                commit=False # All logging should be the same commit
            )

        wandb.log(
            {
                "start_timestamp": sample.start_timestamp,
                "end_timestamp": sample.end_timestamp,
                "predict_hours": sample.predict_hours,
                "lat_lon_bbox": [
                    sample.input_data[...,0].min(), 
                    sample.input_data[...,0].max(),
                    sample.input_data[...,1].min(),
                    sample.input_data[...,1].max(),
                ]
            },
        )

    # Introduce a delay to prevent overloading the miner, as excessive requests can sometimes lead to an invalid miner response.
    time.sleep(5)