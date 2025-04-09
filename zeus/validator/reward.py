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
from zeus.validator.miner_data import MinerData
from zeus.validator.constants import REWARD_DIFFICULTY_SCALER


def help_format_miner_output(
    correct: torch.Tensor, response: torch.Tensor
) -> torch.Tensor:
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
    
    try:
        if response.shape[0] + 1 == correct.shape[0]:
            # NOTE: temporary v0.1.1 -> v0.2.0 since end_timestamp is now included
            # so repeat last hour if miner is still running old code
            response = torch.cat((response, response[-1:]))

        if response.ndim - 1 == correct.ndim and response.shape[-1] == 1:
            # miner forgot to squeeze.
            response = response.squeeze(-1)
        
        return response
    except IndexError:
        # if miner's output is so wrong we cannot even index, do not try anymore
        return response


def get_shape_penalty(correct: torch.Tensor, response: torch.Tensor) -> bool:
    """
    Compute penalty for predictions that are incorrectly shaped or contains NaN/infinities.

    Args:
        correct (torch.Tensor): The correct output tensor.
        response (torch.Tensor): The response tensor from the miner.

    Returns:
        float: True if there is a shape penalty, False otherwise
    """
    penalty = False
    if response.shape != correct.shape:
        penalty = True
    elif not torch.isfinite(response).all():
        penalty = True

    return penalty


def set_rewards(
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

    # compute unnormalised scores and penalties
    for miner_data in miners_data:
        prediction = help_format_miner_output(output_data, miner_data.prediction)
        shape_penalty = get_shape_penalty(output_data, prediction)

        if shape_penalty:
            rmse = -1.0  # Using -1.0 to indicate penalty.
        else:
            rmse = torch.sqrt(torch.mean((prediction - output_data) ** 2)).item()
            rmse_values.append(rmse)
       
        miner_data.shape_penalty = shape_penalty
        miner_data.rmse = rmse

    # if everybody got a shape penalty, set all scores to 0
    if not rmse_values:
        for miner_data in miners_data:
            miner_data.reward = 0.0
        return miners_data

    min_rmse = min(rmse_values)
    max_rmse = max(rmse_values)

    avg_difficulty = difficulty_grid.mean()
    # make difficulty [-1, 1], then go between [1/scaler, scaler]
    gamma = np.power(REWARD_DIFFICULTY_SCALER, avg_difficulty * 2 - 1)

    for miner_data in miners_data:
        if miner_data.shape_penalty:
            miner_data.reward = 0.0
        else:
            if max_rmse == min_rmse:
                miner_data.reward = 1.0
            else:
                norm_rmse = (max_rmse - miner_data.rmse) / (max_rmse - min_rmse)
                miner_data.reward = np.power(norm_rmse, gamma) # apply gamma correction
    return miners_data
