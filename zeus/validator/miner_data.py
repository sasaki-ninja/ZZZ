from dataclasses import dataclass, field
from typing import Dict, Optional
import torch


@dataclass
class MinerData:
    uid: int
    hotkey: str
    prediction: torch.Tensor
    reward: Optional[float] = None # all below are not set initially
    rmse: Optional[float] = None
    shape_penalty: Optional[bool] = None

    @property
    def metrics(self):
        return {
             "RMSE": self.rmse,
             "score": self.reward,
             "shape_penalty": self.shape_penalty,
         }