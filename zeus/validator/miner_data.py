from dataclasses import dataclass
from typing import Dict, Optional
import torch


@dataclass
class MinerData:
    uid: int
    hotkey: str
    prediction: torch.Tensor
    reward: Optional[float] = None  # not set initially
    metrics: Optional[Dict[str, float]] = None  # not set initially
