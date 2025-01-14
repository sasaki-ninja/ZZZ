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
from typing import List
import numpy as np
import torch
import bittensor as bt

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
        # miner forgot to unsqueeze.
        return response.squeeze()
    
    if correct.shape[:-1] == response.shape[:-1] and (correct.shape[-1] + 2) == response.shape[-1]:
        # miner included lat and longitude, slice those off
        return response[..., 2:]
    return response

def compute_penalty(correct: torch.Tensor, response: torch.Tensor) -> float:
    """
    Compute penalty for predictions that are incorrectly shaped or contains NaN/infinities.

    Args:
        correct (torch.Tensor): The correct output tensor.
        response (torch.Tensor): The response tensor from the miner.

    Returns:
        float: 0.0 if prediction is invalid, 1.0 if valid
    """
    valid = True
    if response.shape != correct.shape:
        valid = False
    elif not torch.isfinite(response).any():
        valid = False
    
    return 1.0 if valid else 0.0


def reward(correct: torch.Tensor, response: torch.Tensor) -> float:
    """
    Reward the miner response to the request. This method returns a reward
    value for the miner, which is used to update the miner's score.

    Returns:
    - float: The reward value for the miner.
    """
    response = help_format_miner_output(correct, response)
    penalty = compute_penalty(correct, response)
    if penalty == 0.0:
        return penalty
    
    MSE = ((response - correct) ** 2).mean()
    # Miners should get the lowest MSE. If more than 10 MSE, just get 0 reward.
    return max(0.0, 1.0 - MSE / 10.0)


def get_rewards(
    correct_outputs: torch.Tensor,
    responses: List[torch.Tensor],
) -> torch.Tensor:
    """
    Returns an array of rewards for the given query and responses.

    Args:
    - correct_outputs (torch.Tensor): The output the miner should aim to give.
    - responses (List[torch.Tensor]): A list of responses from the miners.

    Returns:
    - np.ndarray: An array of rewards for the given query and responses.
    """
    # Get all the reward results by iteratively calling your reward() function.

    return np.array([reward(correct_outputs, response) for response in responses])
