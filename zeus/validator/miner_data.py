from dataclasses import dataclass, field
from typing import Dict, Optional
import torch


@dataclass
class MinerData:
    uid: int
    hotkey: str
    prediction: torch.Tensor
    reward: Optional[float] = None  # not set initially
    penalty: Optional[float] = None
    _metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def metrics(self):
        return self._metrics | {
            "score": self.reward,
            "penalty": self.penalty
        }