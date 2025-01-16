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
    correct_outputs: torch.Tensor,
    responses: List[torch.Tensor],
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """
    Returns an array of rewards for the given query and responses.

    Args:
    - correct_outputs (torch.Tensor): The output the miner should aim to give.
    - responses (List[torch.Tensor]): A list of responses from the miners.

    Returns:
    - np.ndarray: An array of rewards for the given query and responses.
    - List[Dict[str, float]]: A list of metrics and scores for each miner as a dictionary.
    """
    miner_rewards = []
    miner_metrics = []
    
    for response in responses:
        RMSE = -1.0 # default values if penalty occurs
        score = 0.0

        response = help_format_miner_output(correct_outputs, response)
        penalty = compute_penalty(correct_outputs, response)

        # only score if no penalty
        if penalty == 0.0:
            RMSE = ((response - correct_outputs) ** 2).mean().sqrt()
             # Miners should get the lowest MSE. If more than 5 MSE, just get 0 reward.
            score = max(0.0, 1.0 - RMSE / 5.0)
        
        miner_rewards.append(score)
        miner_metrics.append(
            {
                "penalty": penalty,
                "RMSE": RMSE, 
                "score": score
             }
        )
 
    return np.array(miner_rewards), miner_metrics
