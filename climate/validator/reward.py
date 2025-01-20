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

def get_grid_lats_lons(sample_input_data: torch.Tensor) -> List[Tuple[float, float]]:
    # Extract latitude and longitude
    lat_lon_points = data[..., :2]  # Select only the latitude and longitude
    unique_lat_lon = lat_lon_points.view(-1, 2).unique(dim=0)  # Flatten and get unique points

    # Convert to a list of tuples if desired
    unique_lat_lon_list = [tuple(point.tolist()) for point in unique_lat_lon]

    return unique_lat_lon_list  

def count_climate_types_grid(grid_lats_lons: List[Tuple[float, float]], climate_grid: dict) -> dict:
    climate_counts = dict()
    for lat, lon in grid_lats_lons:
        climate_counts[climate_grid[lat][lon]] = climate_counts.get(climate_grid[lat][lon], 0) + 1
    return climate_counts

def get_difficulty(climate_counts: dict, difficulties: dict) -> float:
    total_difficulty = 0
    total_grid_points = 0
    for climate in climate_counts:
        total_difficulty += difficulties[climate] * climate_counts[climate]
        total_grid_points += climate_counts[climate]

    return total_difficulty / total_grid_points

def get_rewards(
    input_data: torch.Tensor,
    correct_outputs: torch.Tensor,
    responses: List[torch.Tensor],
    difficulties: dict,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """
    Returns an array of rewards for the given query and responses.

    Args:
    - input_data (torch.Tensor): The input data to the miner.
    - correct_outputs (torch.Tensor): The output the miner should aim to give.
    - responses (List[torch.Tensor]): A list of responses from the miners.
    - difficulties (dict): A dictionary of difficulties for each climate type.

    Returns:
    - np.ndarray: An array of rewards for the given query and responses.
    - List[Dict[str, float]]: A list of metrics and scores for each miner as a dictionary.
    """
    miner_rewards = []
    miner_metrics = []

    # based on the climate types, we set a max RMSE for the task, 
    # reflecting the difficulty of the forecast in that region
    lat_lon_points = get_grid_lats_lons(input_data)
    climate_counts_grid = count_climate_types_grid(lat_lon_points, climate_grid)
    max_allowed_RMSE = get_difficulty(climate_counts_grid, difficulties)
    
    for response in responses:
        RMSE = -1.0 # default values if penalty occurs
        score = 0.0

        response = help_format_miner_output(correct_outputs, response)
        penalty = compute_penalty(correct_outputs, response)

        # only score if no penalty
        if penalty == 0.0:
            RMSE = ((response - correct_outputs) ** 2).mean().sqrt()
             # Miners should get the lowest MSE. If more than 5 MSE, just get 0 reward.
            score = max(0.0, 1.0 - RMSE / difficulty)
        
        miner_rewards.append(score)
        miner_metrics.append(
            {
                "penalty": penalty,
                "RMSE": RMSE, 
                "max_allowed_RMSE": max_allowed_RMSE,
                "score": score
             }
        )
 
    return np.array(miner_rewards), miner_metrics
