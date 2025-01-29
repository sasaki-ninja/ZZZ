from typing import Tuple, Union, List
import numpy as np
import torch

def bbox_to_str(bbox: Tuple[Union[float, np.number]]) -> str:
    """
    Convert the bounding box to a string.
    """
    return f"[lat_start, lat_end, lon_start, lon_end]=[{float(bbox[0]):.2f}, {float(bbox[1]):.2f}, {float(bbox[2]):.2f}, {float(bbox[3]):.2f}]"

def get_bbox(tensor: Union[np.ndarray, torch.Tensor]) -> Tuple[float, float, float, float]:
    """
    Returns the bounding box of the given tensor, which should be 3 or 4 dimensional (optional time, lat, lon, variables). 
    Variables should be in order of latitude, longitude, and then the rest.

    Returns:
    - bbox (Tuple[float, float, float, float]): The bounding box of the tensor: Latitude start, latitude end, longitude start, longitude end.
    """
    if tensor.ndim == 4:
        # so it works regardless of time dimension
        tensor = tensor[0]
    lat_start = tensor[0, 0, 0].item()
    lat_end = tensor[-1, 0, 0].item()
    lon_start = tensor[0, 0, 1].item()
    lon_end = tensor[0, -1, 1].item()
    return lat_start, lat_end, lon_start, lon_end

def slice_bbox(matrix: Union[np.ndarray, torch.Tensor], bbox: Tuple[float, float, float, float]) -> Union[np.ndarray, torch.Tensor]:
    """
    Slice the matrix to the given lat-lon bounding box. This assumes that the matrix is of shape (180 * fidelity + 1, 360 * fidelity, ...).
    NOTE: it is also assumed that coordinates are in the range of -90 to 90 for latitude and -180 to 179.75 for longitude.
    """

    fidelity = matrix.shape[1] // 360

    lat_start, lat_end, lon_start, lon_end = bbox
    lat_start_idx = int((90 + lat_start) * fidelity)
    lat_end_idx = int((90 + lat_end) * fidelity)
    lon_start_idx = int((180 + lon_start) * fidelity)
    lon_end_idx = int((180 + lon_end) * fidelity)

    return matrix[lat_start_idx:lat_end_idx+1, lon_start_idx:lon_end_idx+1, ...]

def get_grid(lat_start: float, lat_end: float, lon_start: float, lon_end: float, fidelity: int = 4) -> torch.Tensor:
    """
    Get a grid of lat-lon points in the given bounding box.
    """
    return torch.stack(
                    torch.meshgrid(
                        *[
                            torch.linspace(
                                start,
                                end, 
                                int((end - start) * fidelity) + 1
                            )
                            for start,end in [
                                (lat_start, lat_end), 
                                (lon_start, lon_end)
                            ]
                        ],
                        indexing="ij",
                    ),
                    dim=-1,
                ) # (lat, lon, 2)