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

from typing import List, Optional, Tuple
from functools import partial
import time
import bittensor as bt
import wandb
import numpy as np
import torch

from zeus.data.sample import Era5Sample
from zeus.data.era5.era5_cds import Era5CDSLoader
from zeus.utils.time import timestamp_to_str
from zeus.utils.coordinates import bbox_to_str
from zeus.validator.reward import get_rewards
from zeus.validator.miner_data import MinerData
from zeus.utils.uids import get_random_uids
from zeus.validator.constants import FORWARD_DELAY_SECONDS


async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    # based on the block, we decide if we should score old stored predictions.
    if self.database.should_score(self.block):
        bt.logging.info(f"Scoring all stored predictions for live ERA5 data.")
        self.database.score_and_prune(score_func=partial(complete_challenge, self))
        return

    data_loader: Era5CDSLoader = self.cds_loader
    if not data_loader.is_ready():
        bt.logging.info(
            "Data loader is not ready yet... Waiting until ERA5 data is downloaded."
        )
        time.sleep(10)  # Don't need to spam this message
        return

    # Let's sample some data
    bt.logging.info(f"Sampling data...")
    sample = data_loader.get_sample()
    bt.logging.success(
        f"Data sampled with bounding box {bbox_to_str(sample.get_bbox())}"
    )
    bt.logging.success(
        f"Data sampled starts from {timestamp_to_str(sample.start_timestamp)} | Asked to predict {sample.predict_hours} hours ahead."
    )

    # get some miners
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    axons = [self.metagraph.axons[uid] for uid in miner_uids]
    miner_hotkeys: List[str] = list([axon.hotkey for axon in axons])

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

    # Create a dummy output with the same shape as the sample's prediction grid to check for penalties
    sample.output_data = torch.zeros(
        sample.predict_hours, sample.x_grid.shape[0], sample.x_grid.shape[1]
    )
    # Identify miners who should receive a penalty and return the actual good predictions
    good_miners = complete_challenge(
        self, sample, miner_hotkeys, responses, for_penalty=True
    )

    if len(good_miners) > 0:
        bt.logging.success(
            "Storing challenge and sensible miner responses in SQLite database"
        )
        self.database.insert(
            sample,
            [miner.hotkey for miner in good_miners],
            [miner.prediction for miner in good_miners],
        )
    # Introduce a delay to prevent spamming requests - and so miners should stay under free tier API request limit
    time.sleep(FORWARD_DELAY_SECONDS - (time.time() - start))


def complete_challenge(
    self,
    sample: Era5Sample,
    hotkeys: List[str],
    predictions: List[torch.Tensor],
    for_penalty: bool = False,
) -> Optional[List[MinerData]]:
    """
    Handle miner responses. Based on hotkeys to also work for delayed rewarding.
    If for_penalty is True, we only score miners that received a penalty (i.e. wrong shape/no response).
    This penalty can be calculated without knowning the actual output yet.
    If for_penalty is True, we return MinerData for each miner that did NOT receive a penalty, as these need to be scored.
    """

    lookup = {axon.hotkey: uid for uid, axon in enumerate(self.metagraph.axons)}

    # Make miner data for each miner that is still alive
    miners_data = []
    for hotkey, prediction in zip(hotkeys, predictions):
        uid = lookup.get(hotkey, None)
        if uid is not None:
            miners_data.append(MinerData(uid=uid, hotkey=hotkey, prediction=prediction))

    # score and reward just those miners
    miners_data = get_rewards(
        output_data=sample.output_data,
        miners_data=miners_data,
        difficulty_grid=self.difficulty_loader.get_difficulty_grid(sample),
    )

    # filter out only the miners that got a penalty.
    final_miners_data = miners_data
    if for_penalty:
        final_miners_data = [
            miner for miner in miners_data if miner.metrics["penalty"] > 0.0
        ]
        if len(final_miners_data) > 0:
            bt.logging.success(f"Punishing miners that did not respond immediately.")

    self.update_scores(
        [miner.reward for miner in final_miners_data],
        [miner.uid for miner in final_miners_data],
    )

    # print interesting miner predictions and store best miners for the Proxy
    miners_scores = {}
    for miner in final_miners_data:
        if len(miner.prediction) != 0:
            miners_scores[uid] = miner.reward
        bt.logging.success(
            f"UID: {miner.uid:3} | Predicted shape: {miner.prediction.shape} | Reward: {miner.reward}"
        )
    self.last_responding_miner_uids = sorted(
        miners_scores, key=miners_scores.get, reverse=True
    )

    # do W&B logging
    if not self.config.wandb.off:
        for miner in final_miners_data:
            wandb.log(
                {f"miner_{miner.uid}_{key}": val for key, val in miner.metrics.items()},
                commit=False,  # All logging should be the same commit
            )

        wandb.log(
            {
                "start_timestamp": sample.start_timestamp,
                "end_timestamp": sample.end_timestamp,
                "predict_hours": sample.predict_hours,
                "lat_lon_bbox": sample.get_bbox(),
            },
        )

    # optionally return miners that should be stored if we were doing penalty
    if for_penalty:
        return [miner for miner in miners_data if miner not in final_miners_data]
