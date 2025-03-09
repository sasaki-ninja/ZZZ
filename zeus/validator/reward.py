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
from typing import List, Tuple, Dict
import numpy as np
import torch
import bittensor as bt
from zeus.validator.miner_data import MinerData
from zeus.validator.constants import DIFFICULTY_OFFSET, DIFFICULTY_MULTIPLIER

def help_format_miner_output(correct: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
    """
    Reshape or slice miner output if it is almost the correct shape.

    Args:
        correct (torch.Tensor): The correct output tensor.
        response (torch.Tensor): The response tensor from the miner.

    Returns:
       Sliced/reshaped miner output.
    """
    if correct.shape == response.shape:
        return response
    
    if correct.ndim + 1 == response.ndim and response.shape[-1] == 1:
        # miner forgot to squeeze.
        return response.squeeze()
    
    if correct.shape[:-1] == response.shape[:-1] and (correct.shape[-1] + 2) == response.shape[-1]:
        # miner included latitude and longitude, slice those off
        return response[..., 2:]
    return response

def compute_penalty(correct: torch.Tensor, response: torch.Tensor) -> float:
    """
    Compute penalty for predictions that are incorrectly shaped or contains NaN/infinities.

    Args:
        correct (torch.Tensor): The correct output tensor.
        response (torch.Tensor): The response tensor from the miner.

    Returns:
        float: 0.0 if prediction is valid, 1.0 if invalid
    """
    valid = True
    if response.shape != correct.shape:
        valid = False
    elif not torch.isfinite(response).all():
        valid = False
    
    return 0.0 if valid else 1.0


def get_rewards(
    output_data: torch.Tensor,
    miners_data: List[MinerData],
    difficulty_grid: np.ndarray,
) -> List[MinerData]:
    """
    Returns an array of rewards for the given query and responses.

    Args:
    - sample (Era5Sample): The data sampled including input and output.
    - responses (List[torch.Tensor]): A list of responses from the miners.
    - difficulty_grid (np.ndarray): A matrix representing the difficulty of each coordinate in the sample.

    Returns:
    - np.ndarray: An array of rewards for the given query and responses.
    - List[Dict[str, float]]: A list of metrics and scores for each miner as a dictionary.
    """

    # based on historical time-derivate of variance we have calculated a difficulty per coordinate
    # with some offset (the mean difficulty) and multiplier
    z_score_grid = difficulty_grid * DIFFICULTY_MULTIPLIER + DIFFICULTY_OFFSET

    for miner_data in miners_data:
        RMSE = -1.0 # default values if penalty occurs
        score = 0.0

        prediction = help_format_miner_output(output_data, miner_data.prediction)
        penalty = compute_penalty(output_data, prediction)

        # only score if no penalty
        if penalty == 0.0:
            RMSE = (
                (
                    (prediction - output_data) / z_score_grid
                ) ** 2
            ).mean().sqrt()
             # Miners should get the lowest MSE.
            score = max(0.0, 1.0 - RMSE)

        miner_data.reward = score
        miner_data.metrics = {
            "penalty": penalty,
            "RMSE": RMSE,
            "avg_difficulty": difficulty_grid.mean(),
            "score": score,
        }
 
    # sort unpenalized/honest miners by their initial score to determine rankings.
    honest_miners = [m for m in miners_data if m.metrics["penalty"] == 0.0]
    honest_miners.sort(key=lambda x: x.metrics["initial_score"], reverse=True)
    
    # apply sigmoid reward distribution based on miners' ranking.
    for rank, miner in enumerate(honest_miners, 1):
        miner.reward = sigmoid_rank(float(rank))
        miner.metrics["rank"] = rank
        miner.metrics["score"] = miner.reward

    # set zero reward for penalized/unhonest miners.
    for miner in miners_data:
        if miner.metrics["penalty"] != 0.0:
            miner.reward = 0.0
            miner.metrics["rank"] = -1
            miner.metrics["score"] = 0.0
 
    return miners_data

def sigmoid_rank(rank: float) -> float:
    """
    Apply an S-curve distribution (sigmoid function) to rank and compute a score in the range [0,1],
    centered at rank 125 with a scaling factor of 20.
    
    Args:
        rank (float): The rank position (1 to 256)
    
    Returns:
        float: Sigmoid score between 0 and 1
    """
    return 1.0 / (1.0 + np.exp((rank - 125) / 20))