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
    Calculates rewards for miner predictions based on RMSE and relative difficulty.

    Args:
        output_data (torch.Tensor): The ground truth data.
        miners_data (List[MinerData]): List of MinerData objects containing predictions.
        difficulty_grid (np.ndarray): Difficulty grid for each coordinate. Currenly not used.

    Returns:
        List[MinerData]: List of MinerData objects with updated rewards and metrics.
    """
    rmse_values = []

    for miner_data in miners_data:
        prediction = help_format_miner_output(output_data, miner_data.prediction)
        penalty = compute_penalty(output_data, prediction)

        if penalty == 0.0:
            rmse = torch.sqrt(torch.mean((prediction - output_data) ** 2)).item()
            rmse_values.append(rmse)
        else:
            rmse = -1.0 # Using -1.0 to indicate penalty.

        miner_data.metrics = {
            "penalty": penalty,
            "RMSE": rmse,
        }

    if not rmse_values:
        for miner_data in miners_data:
            miner_data.metrics["score"] = 0.0
            miner_data.reward = 0.0
        return miners_data

    min_rmse = min(rmse_values)
    max_rmse = max(rmse_values)

    for miner_data in miners_data:
        if miner_data.metrics["RMSE"] == -1.0:
            miner_data.metrics["score"] = 0.0
        else:
            if max_rmse == min_rmse:
                miner_data.metrics["score"] = 1.0
            else:
                miner_data.metrics["score"] = (max_rmse - miner_data.metrics["RMSE"]) / (max_rmse - min_rmse)
        
        miner_data.reward = miner_data.metrics["score"]

    return miners_data
