from dataclasses import dataclass
from typing import Dict, Optional
import torch


@dataclass
class MinerData:
    uid: int
    hotkey: str
    prediction: torch.Tensor
    reward: Optional[float] = None  # not set initially
    _metrics: Dict[str, float] = {}

    @property
    def metrics(self):
        return self._metrics | {"score": self.reward}