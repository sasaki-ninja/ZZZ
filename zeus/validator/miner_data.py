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
    baseline_improvement: Optional[float] = None
    _shape_penalty: Optional[bool] = None

    @property
    def metrics(self):
        return {
             "RMSE": self.rmse,
             "score": self.reward,
             "shape_penalty": self.shape_penalty,
         }

    @property
    def shape_penalty(self):
        return self._shape_penalty
    
    @shape_penalty.setter
    def shape_penalty(self, value: bool):
        self._shape_penalty = value
        if value:
            self.rmse = -1.0
            self.reward = 0